import pathlib

import re
import xml.etree.ElementTree as xml

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import functional as F

import cv2 as cv
import shapely
import kornia

from libs.unionfind import UnionFind


def _fill_poly(img, poly, color):
    shape = map(lambda a: np.array(a, dtype=np.int32), [poly.exterior.coords] + [r.coords for r in poly.interiors])
    cv.fillPoly(img, list(shape), color)



def _stroke_poly(img, poly, color, thickness=2):
    cv.polylines(img, [np.array(poly.exterior.coords, dtype=np.int32)], True, color, thickness)



def _fix_polyline(polyline):
    # shapely.Polygon requires at least 3 points per contour
    while len(polyline) < 3:
        polyline = np.insert(polyline, -1, polyline[-1], axis=0)
    return polyline



def setup_eval(fake_masks, label_tile, filter_area=35):
    _, tracks, vias = fake_masks.split(1, dim=0)
    tracks = torch.max(tracks, vias).squeeze()

    track_img = np.zeros(tracks.shape, dtype=np.uint8)
    track_img[tracks >= 0.5] = 255

    track_polys = extract_polygons(track_img)
    track_polys = list(filter(lambda p: p.area >= filter_area, track_polys))

    h, w = tracks.shape
    track_eval = TrackEval(track_polys, label_tile.tracks, (filter_area, filter_area, w-filter_area, h-filter_area))

    label_img = np.zeros_like(track_img, dtype=np.uint8)
    label_tile.fill_tracks_gs(label_img)
    tp = np.logical_and(track_img, label_img).sum()
    union = np.logical_or(track_img, label_img).sum()
    iou = tp / union

    total = h * w
    tn = total - union
    acc = (tp + tn) / total
    return track_eval, iou, acc



def extract_polygons(img, ignore_holes=False):
    CV_NEXT_CONTOUR = 0
    CV_CHILD_CONTOUR = 2
    def _cv_squeeze(polyline):
        # Remove OpenCV extra dimension for the contour points
        return np.squeeze(polyline, 1)

    # Optionally only retrieve external contours
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL if ignore_holes else cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
    if hierarchy is None or len(hierarchy) == 0:
        return []

    hierarchy = np.squeeze(hierarchy, 0)  # OpenCV wraps the hierarchy in an extra dimension
    polygons = []
    idx = 0
    while idx >= 0:
        child_idx = hierarchy[idx][CV_CHILD_CONTOUR]
        holes = []
        while child_idx >= 0:
            holes.append(_fix_polyline(_cv_squeeze(contours[child_idx])))
            child_idx = hierarchy[child_idx][CV_NEXT_CONTOUR]

        polygons.append(shapely.Polygon(_fix_polyline(_cv_squeeze(contours[idx])), holes))
        idx = hierarchy[idx][CV_NEXT_CONTOUR]
    return polygons



class Tile:
    LINE_RE = re.compile(r'[ML](\d+),(\d+) ')

    def __init__(self, svg_file):
        tree = xml.ElementTree()
        tree.parse(svg_file)

        self.name = pathlib.Path(svg_file).stem
        self.tracks = []
        self.vias = []

        for track in tree.iter('{http://www.w3.org/2000/svg}path'):
            segments = Tile.LINE_RE.findall(track.attrib['d'])
            track = [(int(x), int(y)) for x, y in segments]
            self.tracks.append(shapely.Polygon(_fix_polyline(track)))

        for via in tree.iter('{http://www.w3.org/2000/svg}circle'):
            cx = int(via.attrib['cx'])
            cy = int(via.attrib['cy'])
            r = int(via.attrib['r'])
            self.vias.append((cx, cy, r))

    def has_tracks(self):
        return len(self.tracks) > 0

    def has_vias(self):
        return len(self.vias) > 0

    def has_labels(self):
        return self.has_tracks() or self.has_vias()

    def fill_tracks(self, img, color=(0, 255, 0)):
        r, g, b = color  # Convert from RGB to (OpenCV native) BGR
        for track in self.tracks:
            _fill_poly(img, track, (b, g, r))
        return img

    def fill_tracks_gs(self, img):
        for track in self.tracks:
            _fill_poly(img, track, (255))
        return img

    def fill_vias(self, img, color=(255, 0, 0)):
        for x, y, r in self.vias:
            cv.circle(img, np.array([x, y], dtype=np.int32), np.int32(r), color, cv.FILLED, cv.LINE_4)
        return img



class TrackEval:
    class Stats:
        def __init__(self):
            self.opens = 0  # Intersects two disjoint labels
            self.shorts = 0  # Intersects a label covered by multiple tracks
            self.false_pos = 0  # Does not intersect any label
            self.false_neg = 0  # Label not intersecting any tracks
            self.total_tracks = 0


        def __add__(self, other):
            self.opens += other.opens
            self.shorts += other.shorts
            self.false_pos += other.false_pos
            self.false_neg += other.false_neg
            self.total_tracks += other.total_tracks
            return self



    class Result:
        def __init__(self, num_tracks, num_labels):
            self.track_hits = [set() for  _ in range(num_tracks)]
            self.label_hits = [set() for  _ in range(num_labels)]

            self.track_stats = [TrackEval.Stats() for _ in range(num_tracks)]
            self.false_neg_labels = set()
            self.num_false_neg = 0
            self.num_distinct_labels = 0

        def to_stats(self):
            cstats = TrackEval.Stats()
            cstats.false_neg = self.num_false_neg
            cstats.total_tracks = self.num_distinct_labels
            for stat in self.track_stats:
                cstats += stat
            return cstats



    def __init__(self, tracks, labels, bounds):
        l, t, r, b = bounds
        def _intersects_tile(poly):
            x1, y1, x2, y2 = poly.bounds
            return not (x1 >= r or x2 <= l or y1 >= b or y2 <= t)

        self.tracks = list(filter(_intersects_tile, tracks))
        self.labels = list(filter(_intersects_tile, labels))
        self.label_map = UnionFind(len(labels))
        if len(self.labels) > 0:
            self.label_tree = shapely.STRtree(self.labels)
            self._find_overlapping_labels()


    def _find_overlapping_labels(self):
        # Labels can overlap each other, which would create false ESD shorts.
        # Instead of merging overlapping label polygons, which would be inefficient,
        # we map all overlapping labels to a common index using the union-find algorithm.
        overlaps = self.label_tree.query(self.labels, predicate='intersects')
        for i in range(overlaps.shape[1]):
            l1, l2 = overlaps[:, i]
            if l1 != l2:
                self.label_map.union(l1, l2)


    def eval(self) -> Result:
        result = TrackEval.Result(len(self.tracks), len(self.labels))
        if len(self.tracks) > 0 and len(self.labels) > 0:
            overlaps = self.label_tree.query(self.tracks, predicate='intersects')
            for i in range(overlaps.shape[1]):
                t, l = overlaps[:, i]
                l = self.label_map.find(l)
                result.track_hits[t].add(l)
                result.label_hits[l].add(t)

        for i, stats in enumerate(result.track_stats):
            num_hits = len(result.track_hits[i])
            stats.shorts = max(num_hits - 1, 0)
            stats.false_pos = 0 if num_hits > 0 else 1

        for l in range(len(self.labels)):
            r = self.label_map.find(l)
            hits = result.label_hits[r]
            if len(hits) == 0:
                result.false_neg_labels.add(l)
            if r == l:
                result.num_distinct_labels += 1
                result.num_false_neg += 1 if len(hits) == 0 else 0
                for i, t in enumerate(hits):
                    result.track_stats[t].opens += 1 if i > 0 else 0  # Do not count opens twice
        return result


    def draw_result(self, img, result: Result, thickness=2) -> Stats:
        cstats = TrackEval.Stats()
        cstats.false_neg = result.num_false_neg
        cstats.total_tracks = result.num_distinct_labels
        for t, stat in enumerate(result.track_stats):
            cstats += stat
            if stat.shorts + stat.opens + stat.false_pos == 0:
                color = (255, 255, 255)
            else:
                color = (255 if stat.shorts > 0 else 0, 255 if stat.false_pos > 0 else 0, 255 if stat.opens > 0 else 0)
            _fill_poly(img, self.tracks[t], color)

        for i, label in enumerate(self.labels):
            if self.label_map.find(i) in result.false_neg_labels:
                color = (255, 255, 0)
            else:
                color = (128, 128, 128)
            _stroke_poly(img, label, color, thickness)
        return cstats

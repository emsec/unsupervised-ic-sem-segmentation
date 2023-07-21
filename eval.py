import libs.datasets as datasets
import libs.labels as labels
from libs.unet_resize_conv import UNet

import argparse
import pathlib
import os
import re

import numpy as np

import torch
from torch.cuda import amp
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.transforms.functional as V

import kornia


class EvalDataset(datasets.SemMaskDataset):
        def __init__(self, *args, **kwargs):
            self.input_patch_size = kwargs.pop('input_patch_size', 174)
            self.output_patch_size = kwargs.pop('output_patch_size', 128)
            super().__init__(*args, **kwargs)
            self.kernel = torch.ones(5, 5, device=self.device)

        def _collate(self, sample):
            sample = kornia.filters.median_blur(sample.unsqueeze(0), 5).squeeze(0)

            padding = (self.input_patch_size - self.output_patch_size) // 2
            sem_batch = kornia.contrib.extract_tensor_patches(sample.unsqueeze(0), self.input_patch_size, self.output_patch_size, padding=padding).squeeze(0)

            grad_batch = V.center_crop(sem_batch, (self.output_patch_size, self.output_patch_size))
            grad_batch = 5 * kornia.morphology.gradient(grad_batch, self.kernel, engine='convolution')
            grad_batch = grad_batch.clamp(0, 1)

            sem_batch = V.normalize(sem_batch, (0.5), (0.5))
            return sem_batch, grad_batch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--encoder', required=True, help='The encoder model to load')
    parser.add_argument('-b', '--batch_split', type=int, default=1, help='Number of chunks a single batch (SEM image) is split into to avoid CUDA OOMs')
    parser.add_argument('-f', '--fast', action='store_true', help='Use CUDA amp for faster training with reduced precision')
    parser.add_argument('-d', '--decoder', help='Load the decoder and compute the reconstruction loss')
    parser.add_argument('-o', '--out_dir', help='Save ESD visualizations of images where errors occured in this directory')
    args = parser.parse_args()

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    assert torch.cuda.is_available()
    torch.set_float32_matmul_precision('high')  # Allow faster TF float32 on Nvidia
    device = torch.device('cuda:0')

    eval_data = torch.utils.data.DataLoader(EvalDataset("dataset/sems", batch_split=args.batch_split, input_patch_size=174, output_patch_size=128), batch_size=None, shuffle=True, num_workers=8)
    labels_dir = pathlib.Path('dataset/labels')

    encoder = UNet(1, 3, padding=False, activation=nn.Softmax(dim=1)).to(device)
    encoder.load_state_dict(torch.load(pathlib.Path(args.encoder)))

    if args.decoder:
        decoder = UNet(3, 1, padding=True, activation=nn.Sigmoid()).to(device)
        decoder.load_state_dict(torch.load(pathlib.Path(args.decoder)))
        criterion = torch.nn.MSELoss()
    else:
        decoder = None

    @torch.no_grad()
    def eval(sem_batch, grad_batch):
        encoder.eval()
        decoder.eval()
        loss = 0
        if args.fast:
            with amp.autocast():
                fake_masks = encoder(sem_batch)
                if decoder:
                    rec_sems = decoder(fake_masks)
                    loss = criterion(rec_sems, grad_batch)
        else:
            fake_masks = encoder(sem_batch)
            if decoder:
                rec_sems = decoder(fake_masks)
                loss = criterion(rec_sems, grad_batch)

        return loss, fake_masks

    ignored = 0
    evaluated = 0
    track_stats = labels.TrackEval.Stats()
    mean_iou = 0
    mean_px_acc = 0
    mean_loss = 0
    fake_mask_batch = None
    i = 0
    for i, (sem_batch, grad_batch, filename) in enumerate(eval_data):
        tile_nr = int(re.match(r'sem(\d{4})', pathlib.Path(filename).stem)[1])
        print(f'Evaluating tile {i}/{len(eval_data)}: mIoU={mean_iou / max(i, 1):.4f}, mPA={mean_px_acc / max(i, 1):.4f}, mLoss={mean_loss / max(i, 1):.4f}', end='\r', flush=True)

        if not (labels_dir / f'label{tile_nr:04}.svg').exists():
            ignored += 1
            continue

        tile = labels.Tile(labels_dir / f'label{tile_nr:04}.svg')
        if len(tile.tracks) == 0:
            ignored += 1
            continue

        fake_mask_batch = torch.empty((*sem_batch.shape[:2], 3, 128, 128), device=device) if fake_mask_batch is None else fake_mask_batch
        loss, fake_masks = datasets.split_batch_apply(eval, (sem_batch, grad_batch), (fake_mask_batch,), device=device)
        fake_masks = datasets.stitch_image(fake_masks, 32)

        track_eval, iou, acc = labels.setup_eval(fake_masks, tile)
        mean_iou += iou
        mean_px_acc += acc
        mean_loss += loss

        tev = track_eval.eval()
        stats = tev.to_stats()
        track_stats += stats
        evaluated += 1

        if args.out_dir is not None and (stats.opens > 0 or stats.shorts > 0 or stats.false_pos > 0 or stats.false_neg > 0):
            out = np.zeros((*fake_masks.shape[1:], 3), dtype=np.uint8)
            track_eval.draw_result(out, tev, thickness=4)
            f = pathlib.Path(filename).stem
            print(f'ESD Tile {f}: {track_stats.shorts} shorts, {track_stats.opens} opens')
            datasets.save_numpy_image(out, pathlib.Path(args.out_dir) / f'esd_{f}.png')

    mean_iou /= evaluated
    mean_px_acc /= evaluated
    mean_loss /= evaluated

    print(f"Track ESD stats: {track_stats.shorts} shorts, {track_stats.opens} opens, {track_stats.false_pos} false positives, {track_stats.false_neg} false negatives of {track_stats.total_tracks} tracks")
    print(f"Pixel stats: mIoU: {mean_iou}, mean px acc: {mean_px_acc}")
    print(f"{ignored} tiles were ignored, {evaluated} evaluated. Mean loss: {mean_loss}")

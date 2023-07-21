from datetime import datetime
start_time = datetime.now()

import libs.datasets as datasets
import libs.labels as labels
from libs.morphsnakes import morphsnakes
from libs.unet_resize_conv import UNet

import argparse
import pathlib
import os
import re

import numpy as np

import torch
from torch.cuda import amp

import torch
import torch.nn as nn
import torchvision.transforms.functional as V

import kornia



class MorphDataset(datasets.SemMaskDataset):
        def __init__(self, *args, **kwargs):
            self.input_patch_size = kwargs.pop('input_patch_size', 174)
            self.output_patch_size = kwargs.pop('output_patch_size', 128)
            self.track_thresh = kwargs.pop('track_thresh', 67)
            self.via_thresh = kwargs.pop('via_thresh', 157)

            super().__init__(*args, **kwargs)
            self.kernel = torch.ones(5, 5, device=self.device)
            self.morph = None


        def _collate(self, sample):
            _, h, w = sample.shape
            sample = V.crop(sample, 0, 0, h - (h % self.output_patch_size), w - (w % self.output_patch_size))
            sample = kornia.filters.median_blur(sample.unsqueeze(0), 5).squeeze(0)

            padding = (self.input_patch_size - self.output_patch_size) // 2
            sem_batch = kornia.contrib.extract_tensor_patches(sample.unsqueeze(0), self.input_patch_size, self.output_patch_size, padding=padding).squeeze(0)

            grad_batch = V.center_crop(sem_batch, (self.output_patch_size, self.output_patch_size))
            grad_batch = 5 * kornia.morphology.gradient(grad_batch, self.kernel, engine='convolution')
            grad_batch = grad_batch.clamp(0, 1)

            sample = (255 * sample).squeeze().numpy()
            if self.morph is None:
                self.morph = morphsnakes.MorphACWE(None, (sample.shape[-1], sample.shape[-2]))
            tracks = torch.from_numpy(self.morph(sample, self.track_thresh, 50, smoothing=3, lambda1=1, lambda2=2))
            vias = torch.from_numpy(self.morph(sample, self.via_thresh, 50, smoothing=3, lambda1=2, lambda2=1))
            mask = torch.empty((3, *tracks.shape))
            mask[0] = 1 - tracks
            mask[1] = tracks - vias
            mask[2] = vias
            mask_batch = kornia.contrib.extract_tensor_patches(mask.unsqueeze(0), self.output_patch_size, self.output_patch_size).squeeze(0)

            sem_batch = V.normalize(sem_batch, (0.5), (0.5))
            return sem_batch, mask_batch, grad_batch



class RandThreshDataset(datasets.SemMaskDataset):
        def __init__(self, *args, **kwargs):
            self.input_patch_size = kwargs.pop('input_patch_size', 174)
            self.output_patch_size = kwargs.pop('output_patch_size', 128)
            self.track_mean = kwargs.pop('track_mean', 67/256)
            self.via_mean = kwargs.pop('via_mean', 157/256)
            self.variance = kwargs.pop('variance', 25/256)

            super().__init__(*args, **kwargs)
            self.kernel = torch.ones(5, 5, device=self.device)


        def _collate(self, sample):
            sample = kornia.filters.median_blur(sample.unsqueeze(0), 5).squeeze(0)

            padding = (self.input_patch_size - self.output_patch_size) // 2
            sem_batch = kornia.contrib.extract_tensor_patches(sample.unsqueeze(0), self.input_patch_size, self.output_patch_size, padding=padding).squeeze(0)

            grad_batch = V.center_crop(sem_batch, (self.output_patch_size, self.output_patch_size))

            if self.variance > 0:
                tthresh =  self.variance * torch.rand((grad_batch.size(0), 1, 1, 1), device=self.device).expand_as(grad_batch) + self.track_mean
                vthresh = self.variance * torch.rand((grad_batch.size(0), 1, 1, 1), device=self.device).expand_as(grad_batch) + self.via_mean
            else:
                tthresh = self.track_mean
                vthresh = self.via_mean

            mask_batch = torch.zeros((grad_batch.size(0), 3, self.output_patch_size, self.output_patch_size), dtype=grad_batch.dtype, device=self.device)
            b, t, v = mask_batch.split(1, dim=1)
            t[grad_batch > tthresh] = 1
            v[grad_batch > vthresh] = 1
            b[t == 0] = 1
            t[v == 1] = 0

            grad_batch = 5 * kornia.morphology.gradient(grad_batch, self.kernel, engine='convolution')
            grad_batch = grad_batch.clamp(0, 1)

            sem_batch = V.normalize(sem_batch, (0.5), (0.5))
            return sem_batch, mask_batch, grad_batch




class EvalDataset(datasets.SemMaskDataset):
        def __init__(self, *args, **kwargs):
            self.input_patch_size = kwargs.pop('input_patch_size', 174)
            self.output_patch_size = kwargs.pop('output_patch_size', 128)
            super().__init__(*args, **kwargs)

        def _collate(self, sample):
            sample = kornia.filters.median_blur(sample.unsqueeze(0), 5).squeeze(0)

            padding = (self.input_patch_size - self.output_patch_size) // 2
            sem_batch = kornia.contrib.extract_tensor_patches(sample.unsqueeze(0), self.input_patch_size, self.output_patch_size, padding=padding).squeeze(0)
            sem_batch = V.normalize(sem_batch, (0.5), (0.5))
            return (sem_batch,)



parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_dir', required=True)
parser.add_argument('-n', '--num_batches', type=int, required=True, help='Number of SEM images to train on')
parser.add_argument('-b', '--batch_split', type=int, default=1, help='Number of chunks a single batch (SEM image) is split into to avoid CUDA OOMs')
parser.add_argument('-w', '--write', type=int, default=50, help='Evaluate instead of train every x images and save the results')
parser.add_argument('-f', '--fast', action='store_true', help='Use CUDA amp for faster training with reduced precision')
parser.add_argument('-s', '--save_model', action='store_true', help='Save the trained encoder and decoder in the output directory')
parser.add_argument('-a', '--algorithm', required=True, choices=['morph', 'randthresh', 'fixedthresh'], help='Input segmentation algorithm to use for training')
parser.add_argument('--reconstruct_sem', action='store_true', help='Reconstruct SEM images instead of their gradients')

args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

assert torch.cuda.is_available()
torch.set_float32_matmul_precision('high')  # Allow faster TF float32 on Nvidia
device = torch.device('cuda:0')

# Disables automatic batching and splits images manually into patches.
if args.algorithm == 'morph':
    sem2thresh_data = torch.utils.data.DataLoader(MorphDataset("dataset/sems", batch_split=args.batch_split, input_patch_size=174, output_patch_size=128), batch_size=None, shuffle=True, num_workers=0)
elif args.algorithm == 'randthresh':
    sem2thresh_data = torch.utils.data.DataLoader(RandThreshDataset("dataset/sems", batch_split=args.batch_split, input_patch_size=174, output_patch_size=128), batch_size=None, shuffle=True, num_workers=0)
elif args.algorithm == 'fixedthresh':
    sem2thresh_data = torch.utils.data.DataLoader(RandThreshDataset("dataset/sems", variance=0, batch_split=args.batch_split, input_patch_size=174, output_patch_size=128), batch_size=None, shuffle=True, num_workers=0)
else:
    raise Exception('Invalid algorithm')

labels_dir = pathlib.Path('dataset/labels')

decoder = UNet(3, 1, activation=nn.Sigmoid() if not args.reconstruct_sem else nn.Tanh()).to(device)
encoder = UNet(1, 3, padding=False, activation=nn.Softmax(dim=1)).to(device)

decoder.init_weights()
encoder.init_weights()

optD = torch.optim.Adam(list(decoder.parameters()), lr=0.0002, betas=(0.9, 0.999))
optE = torch.optim.Adam(list(encoder.parameters()), lr=0.0002, betas=(0.9, 0.999))

if args.fast:
    scaleD = amp.GradScaler()
    scaleE = amp.GradScaler()

criterion = torch.nn.MSELoss()



def denormalize(img_tensor, min=-1, max=1):
    img_tensor = img_tensor.clamp(min, max) - min
    return img_tensor / (max - min)


def regularization(masks):
    """
    Force per-pixel exclusivity of b, t, v labels.
    """
    b, t, v = masks.split(1, dim=1)
    return 0.1 * (b*t + b*v + t*v).mean()


def split_batch_apply(fn, in_batches, out_batches=()):
    split_batch = in_batches[0].shape[0]
    loss = 0

    for s in range(split_batch):
        out = fn(*[b[s].to(device) for b in in_batches])
        loss += out[0]
        for i in range(len(out_batches)):
            out_batches[i][s].copy_(out[i + 1])
    loss /= split_batch

    if len(out_batches) == 0:
        return loss
    else:
        # Return continuous view of split output patches, ready to be stitched to an image
        return loss, *[b.view(1, -1, *b.shape[2:]).squeeze(0) for b in out_batches]


def train_decoder(mask_batch, grad_batch):
    decoder.train()
    decoder.requires_grad(True)
    optD.zero_grad()
    if args.fast:
        with amp.autocast():
            rec_sems = decoder(mask_batch)
            lossD = criterion(rec_sems, grad_batch)
        lossD = scaleD.scale(lossD)
        lossD.backward()
        scaleD.step(optD)
        scaleD.update()
    else:
        rec_sems = decoder(mask_batch)
        lossD = criterion(rec_sems, grad_batch)
        lossD.backward()
        optD.step()
    return lossD, rec_sems


@torch.no_grad()
def eval_decoder(mask_batch, grad_batch):
    decoder.eval()
    if args.fast:
        with amp.autocast():
            rec_sems = decoder(mask_batch)
            lossD = criterion(rec_sems, grad_batch)
        lossD = scaleD.scale(lossD)
    else:
        rec_sems = decoder(mask_batch)
        lossD = criterion(rec_sems, grad_batch)
    return lossD, rec_sems


def train_encoder(sem_batch, grad_batch):
    encoder.train()
    decoder.train()
    decoder.requires_grad(False)
    optE.zero_grad()
    if args.fast:
        with amp.autocast():
            fake_masks = encoder(sem_batch)
            rec_sems = decoder(fake_masks)
            lossE = criterion(rec_sems, grad_batch) + regularization(fake_masks)
        lossE = scaleE.scale(lossE)
        lossE.backward()
        scaleE.step(optE)
        scaleE.update()
    else:
        fake_masks = encoder(sem_batch)
        rec_sems = decoder(fake_masks)
        lossE = criterion(rec_sems, grad_batch) + regularization(fake_masks)
        lossE.backward()
        optE.step()

    return lossE, fake_masks, rec_sems


@torch.no_grad()
def eval_encoder(sem_batch, grad_batch):
    encoder.eval()
    decoder.eval()
    if args.fast:
        with amp.autocast():
            fake_masks = encoder(sem_batch)
            rec_sems = decoder(fake_masks)
            lossE = criterion(rec_sems, grad_batch) + regularization(fake_masks)
        lossE = scaleE.scale(lossE)
    else:
        fake_masks = encoder(sem_batch)
        rec_sems = decoder(fake_masks)
        lossE = criterion(rec_sems, grad_batch) + regularization(fake_masks)

    return lossE, fake_masks, rec_sems


out_dir = pathlib.Path(args.out_dir)

fake_mask_batch = None
reconstructed_batch = None
batch_nr = 0
while batch_nr < args.num_batches:
    for sem_batch, mask_batch, grad_batch, filename in sem2thresh_data:
        if batch_nr < args.num_batches:
            batch_nr += 1
        else:
            break

        if args.reconstruct_sem:
            grad_batch = V.center_crop(sem_batch, (128, 128))

        if batch_nr % args.write != 0:  # Train
            lossD = split_batch_apply(train_decoder, (mask_batch, grad_batch))
            lossE = split_batch_apply(train_encoder, (sem_batch, grad_batch))
            print(f'Train loss for batch {batch_nr}/{args.num_batches}: D: {lossD.item():.4f}, E: {lossE.item():.4f}')
        else:  # Eval
            fake_mask_batch = torch.empty((*grad_batch.shape[:2], 3, *grad_batch.shape[3:])) if fake_mask_batch is None else fake_mask_batch
            reconstructed_batch = torch.empty(grad_batch.shape) if reconstructed_batch is None else reconstructed_batch

            lossD = split_batch_apply(eval_decoder, (mask_batch, grad_batch))
            lossE, fake_masks, recs = split_batch_apply(eval_encoder, (sem_batch, grad_batch), (fake_mask_batch, reconstructed_batch))

            print(f'Eval loss for batch {batch_nr}/{args.num_batches}: D: {lossD.item():.4f}, E: {lossE.item():.4f}')
            recs = datasets.stitch_image(recs, 32)
            if args.reconstruct_sem:
                recs = denormalize(recs)
            fake_masks = datasets.stitch_image(fake_masks, 32)

            datasets.save_image(recs, out_dir / f'batch{batch_nr:04}_rec.png')

            tile_nr = int(re.match(r'sem(\d{4})', pathlib.Path(filename).stem)[1])
            if (labels_dir / f'label{tile_nr:04}.svg').exists():
                tile = labels.Tile(labels_dir / f'label{tile_nr:04}.svg')
                track_eval, _, _ = labels.setup_eval(fake_masks, tile)

                out = np.zeros((*fake_masks.shape[1:], 3), dtype=np.uint8)
                track_stats = track_eval.draw_result(out, track_eval.eval())
                print(f"Track ESD stats: {track_stats.shorts} shorts, {track_stats.opens} opens, {track_stats.false_pos} false positives, {track_stats.false_neg} false negatives of {track_stats.total_tracks} tracks")
                datasets.save_numpy_image(out, out_dir / f'batch{batch_nr:04}_esd.png')
            else:
                print(f'Did not find labels for image Nr. {tile_nr}')
                datasets.save_image(fake_masks, out_dir / f'batch{batch_nr:04}_mask.png')

if args.save_model:
    torch.save(decoder.state_dict(), out_dir / 'decoder.pt')
    torch.save(encoder.state_dict(), out_dir / 'encoder.pt')

stop_time = datetime.now()
runtime = stop_time - start_time
print(f'Total training time: {runtime}')

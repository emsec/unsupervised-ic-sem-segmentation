import os
import pathlib

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as F

from torch.utils.data import DataLoader, random_split
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension


class UnlabeledImageFolder(VisionDataset):
    """
    Simpler version of `torchvision.datasets.folder.ImageFolder` for image datasets without labels.
    """
    def __init__(
        self,
        root: str,
        extensions=IMG_EXTENSIONS,
        transform=None,
    ) -> None:
        super().__init__(root, transform=transform)
        self.samples = []
        root = pathlib.Path(root)
        if root.is_file():
            self.samples.append(root)
        else:
            for root, _, fnames in sorted(os.walk(root, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if has_file_allowed_extension(path, extensions):
                            self.samples.append(path)


    def __getitem__(self, index: int):
        filename = self.samples[index]
        with open(filename, "rb") as f:
            sample = F.to_tensor(Image.open(f))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, filename


    def __len__(self) -> int:
        return len(self.samples)


    def get_filename(self, index: int) -> pathlib.Path:
        return pathlib.Path(self.samples[index])



class SemMaskDataset(UnlabeledImageFolder):
    def __init__(self, *args, **kwargs):
        self.split_batch = kwargs.pop('batch_split', 0)
        self.device = kwargs.pop('device', torch.device('cpu'))
        self.current_file = None
        super().__init__(*args, **kwargs)


    def _collate(sample):
        raise NotImplementedError()


    def __getitem__(self, index):
        """
        Returns a batch and its filename
        The batch has dimensions S, N, C, H, W
        with the batch size being split into a
        batch_split dimension (S) and a smaller batch size (N).
        """
        sample, self.current_file = super().__getitem__(index)
        batches = list(self._collate(sample.to(self.device)))
        if self.split_batch > 0:
            for i in range(len(batches)):
              batches[i] = batches[i].unsqueeze(0).view(self.split_batch, -1, *batches[i].shape[1:])
        return *batches, self.current_file



def stitch_image(tensor, nrows, nchans=None):
    """
    Adapted to gray scale images from torch.utils.make_grid()

    nchans: Desired number of output channels. Additional channels contain zeros.
    """
    if nchans is None:
        nchans = tensor.size(1)

    nmaps = tensor.size(0)
    xmaps = min(nrows, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = tensor.size(2), tensor.size(3)
    grid = torch.zeros((nchans, height * ymaps, width * xmaps), dtype=tensor.dtype)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(0, 0, tensor.size(1)).narrow(1, y * height, height).narrow(2, x * width, width).copy_(tensor[k])
            k = k + 1
    return grid



def save_image(tensor, file):
    file = pathlib.Path(file)
    os.makedirs(file.parent, exist_ok=True)
    F.to_pil_image(tensor).save(file)



def save_numpy_image(numpy, file):
    file = pathlib.Path(file)
    os.makedirs(file.parent, exist_ok=True)
    Image.fromarray(numpy).save(file)


def to_numpy(ftensor):
    itensor = (255 * ftensor).to(torch.uint8)
    if itensor.dim() == 3:  # C x H x W
        itensor = torch.permute(itensor, (1, 2, 0))
    return itensor.numpy()


def split_batch_apply(fn, in_batches, out_batches=(), device=torch.device('cpu')):
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

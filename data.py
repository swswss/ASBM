import os
from glob import glob

import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LatentFolder(Dataset):
    """Flat folder of pre-computed VAE latents (see prepare_ffhq_latents.py).

    Each `*.npy` file stores [mu; std] of the VAE posterior with shape [2C, H, W];
    every access returns a fresh sample z = mu + std * eps.
    """

    def __init__(self, folder: str):
        self.files = sorted(glob(os.path.join(folder, "*.npy")))
        if not self.files:
            raise RuntimeError(f"No .npy latents found in {folder}")
        print(f"Found {len(self.files)} latent tensors.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lat = torch.from_numpy(np.load(self.files[idx]))  # [2C, H, W]
        c = lat.size(0) // 2
        mu, std = lat[:c], lat[c:]
        z = mu + std * torch.randn_like(mu)
        return z, torch.tensor(1)


def build_dataset(opt):
    if opt.problem_name == "cifar10":
        return datasets.CIFAR10(
            opt.data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),  # [0, 1]
                transforms.Lambda(lambda t: (t * 2) - 1),  # [-1, 1]
            ]),
        )
    if opt.problem_name == "ffhq":
        return LatentFolder(opt.dataset_path)
    raise NotImplementedError(f"Unknown problem: {opt.problem_name}")


def build_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True, drop_last=False)


class DataSampler:
    """Infinite sampler returning one batch of data per `sample()` call."""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataloader = self._infinite(build_loader(dataset, batch_size, shuffle))

    @staticmethod
    def _infinite(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    def sample(self):
        return next(self.dataloader)[0]

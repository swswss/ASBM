"""Pre-compute SD-VAE latents of FFHQ-256 for latent ASBM.

Each image is resized to 256x256 (LANCZOS) if needed, encoded by the VAE, and the posterior
[mu; std] (shape [8, 32, 32]) is saved as <out_dir>/<index:08d>.npy. A latent z = mu + std * eps
is re-sampled every time it is loaded during training (see data.LatentFolder).

Usage:
    python prepare_ffhq_latents.py --image-dir /path/to/ffhq/images1024x1024 --out-dir /path/to/ffhq_latents
"""
import argparse
import os
from glob import glob

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageFolder(Dataset):
    def __init__(self, folder, image_size, exts=(".png", ".jpg", ".jpeg")):
        self.files = sorted(p for e in exts for p in glob(os.path.join(folder, "**", f"*{e}"), recursive=True))
        if not self.files:
            raise RuntimeError(f"No images found in {folder}")
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        print(f"Found {len(self.files)} images.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(self.files[idx]) as im:
            im = im.convert("RGB")
            if im.size != (self.image_size, self.image_size):
                im = im.resize((self.image_size, self.image_size), Image.LANCZOS)
        return self.to_tensor(im) * 2 - 1.0  # [-1, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True, help="FFHQ images (any resolution, searched recursively)")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = ImageFolder(args.image_dir, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device).eval()

    idx = 0
    with torch.no_grad():
        for x in tqdm(loader, desc="Encoding"):
            post = vae.encode(x.to(device)).latent_dist
            lat = torch.cat([post.mean, post.std], dim=1).cpu().numpy().astype(np.float32)  # [B, 2C, H/8, W/8]
            for i in range(lat.shape[0]):
                np.save(os.path.join(args.out_dir, f"{idx:08d}.npy"), lat[i])
                idx += 1
    print(f"Saved {idx} latents to {args.out_dir}")


if __name__ == "__main__":
    main()

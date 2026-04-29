from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms
import os
import torch
from typing import Literal

FourierMode = Literal["none", "magnitude", "phase", "complex", "concat"]


class ImageDataset(Dataset):
    def __init__(
        self,
        file_csv: Path,
        images_dir: Path,
        transform: transforms.Compose | None = None,
        data_limit: int = np.inf,
        fourier: FourierMode = "none",
    ):
        self.images_dir = images_dir
        self.df = pd.read_csv(file_csv)

        # limitar dados
        self.df = self.df if data_limit == np.inf else self.df.head(data_limit)

        # ajustar caminhos
        self.df["img_name"] = self.df["img_name"].apply(
            lambda x: os.path.join(self.images_dir, x)
        )

        # transforms separados
        self.spatial_transform = transforms.Compose([
            transforms.Resize((128, 128))
        ])

        self.to_tensor = transforms.ToTensor()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.transform = transform
        self.fourier: FourierMode = fourier

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = self.spatial_transform(img)

        except Exception as e:
            print(f"Erro na imagem: {img_path} -> {e}")
            return self.__getitem__((idx + 1) % len(self))

        label = self.df.iloc[idx, 1]

        # tensor base (sem normalizar ainda)
        img_tensor = self.to_tensor(img)  # (C, H, W)

        # imagem espacial
        if self.transform is not None:
            image = self.transform(img)
        else:
            image = self.normalize(img_tensor)

        # ===== Fourier =====
        if self.fourier == "none":
            output = image

        elif self.fourier == "magnitude":
            output = self._fft_magnitude(img_tensor)

        elif self.fourier == "phase":
            output = self._fft_phase(img_tensor)

        elif self.fourier == "complex":
            output = self._fft_complex(img_tensor)

        elif self.fourier == "concat":
            fft = self._fft_magnitude(img_tensor)

            # padroniza FFT (importante!)
            fft = (fft - fft.mean()) / (fft.std() + 1e-8)

            output = torch.cat([image, fft], dim=0)

        return output, label, idx

    # =========================
    # AUX FUNCTIONS
    # =========================

    def _to_grayscale(self, img: torch.Tensor):
        # luminância (melhor que média simples)
        if img.shape[0] == 3:
            return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        return img.squeeze(0)

    def _safe_normalize(self, arr: np.ndarray):
        den = arr.max() - arr.min()

        if den < 1e-8:
            return np.zeros_like(arr)
        return (arr - arr.min()) / den

    def _fft_magnitude(self, img: torch.Tensor):
        img_np = self._to_grayscale(img).detach().cpu().numpy()

        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)

        magnitude = np.log(np.abs(fshift) + 1)
        magnitude = self._safe_normalize(magnitude)

        tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

    def _fft_phase(self, img: torch.Tensor):
        img_np = self._to_grayscale(img).detach().cpu().numpy()

        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)

        phase = np.angle(fshift)
        phase = (phase + np.pi) / (2 * np.pi)

        tensor = torch.tensor(phase, dtype=torch.float32).unsqueeze(0)
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

    def _fft_complex(self, img: torch.Tensor):
        img_np = self._to_grayscale(img).detach().cpu().numpy()

        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)

        real = self._safe_normalize(np.real(fshift))
        imag = self._safe_normalize(np.imag(fshift))

        real = torch.tensor(real, dtype=torch.float32)
        imag = torch.tensor(imag, dtype=torch.float32)

        tensor = torch.stack([real, imag], dim=0)
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
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

        self.df = self.df if data_limit == np.inf else self.df.head(data_limit)  #limitacao dos dados para testes

        #csv genérico, para ser adaptável para o pc de cada um
        self.df["img_name"] = self.df["img_name"].apply(lambda x: os.path.join(self.images_dir, x))

        #padrão da ImageNET -> outros modelos feitos em demais bases, possuem outros
        self.spatial_transform = transforms.Compose([transforms.Resize((128, 128))])
        self.to_tensor = transforms.ToTensor()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # mantém compatibilidade: se vier transform custom, usa no modo "none"
        self.transform = transform
        self.fourier: FourierMode = fourier

    def __len__(self):
        return len(self.df)

    #crio um iterador para pegar a imagem, label e a posição do dado
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

        #1. aplica transform (sempre vira tensor)
        img_tensor = self.to_tensor(img)  # (C, H, W)
        image = self.normalize(img_tensor) if self.transform is None else self.transform(img)

        #2. tipos de fourier que podem ser aplicados
        if self.fourier == "none":
            output = image

        elif self.fourier == "magnitude":
            output = self._fft_magnitude(img_tensor)

        elif self.fourier == "phase":
            output = self._fft_phase(img_tensor)

        elif self.fourier == "complex":
            output = self._fft_complex(img_tensor)

        #3. concatenação dos tipos de fourier
        elif self.fourier == "concat":
            fft = self._fft_magnitude(img_tensor)
            output = torch.cat([image, fft], dim=0)

        return output, label, idx

    def _to_grayscale(self, img: torch.Tensor):
        # img: (C, H, W)

        #converter para ciza
        if img.shape[0] == 3:
            return img.mean(dim=0)  # (H, W)
        return img.squeeze(0)

    def _fft_magnitude(self, img: torch.Tensor):
        img_np = self._to_grayscale(img).numpy()

        # FFT
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)

        magnitude = np.log(np.abs(fshift) + 1)

        # normalização
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

        # tensor
        return torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)

    def _fft_phase(self, img: torch.Tensor):
        img_np = self._to_grayscale(img).numpy()
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        phase = np.angle(fshift)
        phase = (phase + np.pi) / (2 * np.pi)  # normaliza [-pi, pi] -> [0, 1]
        return torch.tensor(phase, dtype=torch.float32).unsqueeze(0)

    def _fft_complex(self, img: torch.Tensor):
        img_np = self._to_grayscale(img).numpy()
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)

        real = np.real(fshift)
        imag = np.imag(fshift)

        # normalização independente
        real = (real - real.min()) / (real.max() - real.min() + 1e-8)
        imag = (imag - imag.min()) / (imag.max() - imag.min() + 1e-8)

        real = torch.tensor(real, dtype=torch.float32)
        imag = torch.tensor(imag, dtype=torch.float32)
        return torch.stack([real, imag], dim=0)


        
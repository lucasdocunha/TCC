from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms
import os
import torch
from typing import Literal

# Cada modo define como o tensor de saída é montado a partir da imagem RGB.
# FFT: grayscale (luminância) → fft2 → fftshift; canais de saída variam conforme o modo.
FourierMode = Literal[
    "none",  # Só domínio espacial: RGB normalizado ImageNet (sem FFT).
    "magnitude",  # Espectro de magnitude: log(|F|+1), normalizado; 1 canal.
    "phase",  # Fase do espectro, mapeada para [0, 1]; 1 canal.
    "complex",  # Parte real e imaginária de F, cada uma normalizada; 2 canais.
    "concat",  # RGB (3) + magnitude FFT normalizada (1) empilhados; 4 canais.
    "frequency_3",  # Magnitude FFT com máscara passa-alta (enfatiza altas frequências); 1 canal.
    "concat_frequency",  # RGB (3) + magnitude + fase + passa-alta (3×1 canal); 6 canais.
]

# Ordem fixa para benchmarks (sem nomes duplicados para o mesmo tensor).
ALL_FOURIER_MODES: tuple[FourierMode, ...] = (
    "none",
    "magnitude",
    "phase",
    "complex",
    "concat",
    "frequency_3",
    "concat_frequency",
)


class ImageDataset(Dataset):
    def __init__(
        self,
        file_csv: Path,
        images_dir: Path,
        transform: transforms.Compose | None = None,
        data_limit: int = np.inf,
        fourier: FourierMode = "none",
        spatial_size: tuple[int, int] | None = (128, 128),
    ):
        self.images_dir = images_dir
        self.df = pd.read_csv(file_csv)
        self.df.columns = self.df.columns.str.strip()

        self.df = self.df if data_limit == np.inf else self.df.head(data_limit)

        self.df["img_name"] = self.df["img_name"].apply(
            lambda x: os.path.join(self.images_dir, x)
        )

        self.spatial_transform = (
            transforms.Resize(spatial_size) if spatial_size is not None else None
        )
        self.to_tensor = transforms.ToTensor()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        # Normalização típica para mapas de 1 canal vindos da FFT (após já estarem em [0,1] ou similar).
        self.frequency_normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        self.transform = transform
        self.fourier: FourierMode = fourier

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int, _skip_count: int = 0):
        img_path = self.df.iloc[idx, 0]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.spatial_transform is not None:
                    img = self.spatial_transform(img)

        except Exception as e:
            print(f"Erro na imagem: {img_path} -> {e}")
            if _skip_count + 1 >= len(self):
                raise RuntimeError(
                    "Nenhuma imagem pôde ser carregada (verifique caminhos no CSV e no disco)."
                ) from e
            return self.__getitem__((idx + 1) % len(self), _skip_count=_skip_count + 1)

        label = self.df.iloc[idx, 1]

        img_tensor = self.to_tensor(img)
        if self.transform is not None:
            image = self.transform(img)
        else:
            image = self.normalize(img_tensor)

        # --- Montagem do tensor conforme o modo (espacial vs. frequência vs. híbrido) ---
        if self.fourier == "none":
            output = image

        elif self.fourier == "magnitude":
            # Mapa de energia espectral (um canal).
            output = self.frequency_normalize(self._fft_magnitude(img_tensor))

        elif self.fourier == "phase":
            # Mapa de fases (um canal).
            output = self.frequency_normalize(self._fft_phase(img_tensor))

        elif self.fourier == "complex":
            # Dois canais: real e imaginário da FFT (sem Normalize ImageNet; já normalizados por canal).
            output = self._fft_complex(img_tensor)

        elif self.fourier == "frequency_3":
            # Conteúdo espectral fora do disco central (passa-alta em |F|).
            output = self.frequency_normalize(self._fft_highpass(img_tensor))

        elif self.fourier == "concat":
            # Espaço (3) + magnitude (1): o modelo vê cor/textura e espectro de amplitude juntos.
            fft = self.frequency_normalize(self._fft_magnitude(img_tensor))
            output = torch.cat([image, fft], dim=0)

        elif self.fourier == "concat_frequency":
            # Espaço (3) + três descritores FFT (magnitude, fase, passa-alta), cada um 1 canal.
            fft = torch.cat(
                [
                    self.frequency_normalize(self._fft_magnitude(img_tensor)),
                    self.frequency_normalize(self._fft_phase(img_tensor)),
                    self.frequency_normalize(self._fft_highpass(img_tensor)),
                ],
                dim=0,
            )
            output = torch.cat([image, fft], dim=0)

        else:
            raise ValueError(f"Modo de Fourier inválido: {self.fourier}")

        return output, label, idx

    def _to_grayscale(self, img: torch.Tensor):
        if img.shape[0] == 3:
            return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        return img.squeeze(0)

    def _safe_normalize(self, arr: np.ndarray):
        den = arr.max() - arr.min()
        if den < 1e-8:
            return np.zeros_like(arr)
        return (arr - arr.min()) / den

    def _fft_magnitude(self, img: torch.Tensor):
        # Magnitude log-compressa do espectro centrado (baixas freq. no centro após fftshift).
        img_np = self._to_grayscale(img).detach().cpu().numpy()
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        magnitude = self._safe_normalize(magnitude)
        tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

    def _fft_phase(self, img: torch.Tensor):
        # Fase em [-π, π] reescalada para [0, 1] para estabilizar escala do tensor.
        img_np = self._to_grayscale(img).detach().cpu().numpy()
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        phase = np.angle(fshift)
        phase = (phase + np.pi) / (2 * np.pi)
        tensor = torch.tensor(phase, dtype=torch.float32).unsqueeze(0)
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

    def _fft_complex(self, img: torch.Tensor):
        # Re e Im normalizados separadamente (preserva sinal mas perde escala absoluta cruzada).
        img_np = self._to_grayscale(img).detach().cpu().numpy()
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        real = self._safe_normalize(np.real(fshift))
        imag = self._safe_normalize(np.imag(fshift))
        real = torch.tensor(real, dtype=torch.float32)
        imag = torch.tensor(imag, dtype=torch.float32)
        tensor = torch.stack([real, imag], dim=0)
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

    def _fft_highpass(self, img: torch.Tensor):
        # Máscara circular: zera (atenua) o disco central de baixa frequência; mantém bordas do espectro.
        img_np = self._to_grayscale(img).detach().cpu().numpy()
        fshift = np.fft.fftshift(np.fft.fft2(img_np))
        height, width = fshift.shape
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) * 0.12
        mask = ((y - center_y) ** 2 + (x - center_x) ** 2) >= radius**2
        highpass = np.log1p(np.abs(fshift) * mask)
        highpass = self._safe_normalize(highpass)
        tensor = torch.tensor(highpass, dtype=torch.float32).unsqueeze(0)
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

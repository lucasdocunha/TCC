from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms 
import os 
import torch
from typing import Literal

class ImageDataset(Dataset):
    def __init__(self, file_csv:Path, images_dir:Path, transform:transforms.Compose|None=None, data_limit:int=np.inf, fourier:Literal["false", "true", "concat"]='false'):
        self.images_dir = images_dir
        self.df = pd.read_csv(file_csv)
        self.df = self.df if data_limit == np.inf else self.df.head(data_limit) #limitacao dos dados para testes
        
        #csv genérico, para ser adaptável para o pc de cada um 
        self.df['img_name'] = self.df['img_name'].apply(lambda x: os.path.join(self.images_dir, x))
        
        #padrão da ImageNET -> outros modelos feitos em demais bases, possuem outros
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.fourier = fourier

        
    def __len__(self):
        return len(self.df)
    
    
    #crio um iterador para pegar a imagem, label e a posição do dado
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]

        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')

        except Exception as e:
            print(f"Erro na imagem: {img_path} -> {e}")
            return self.__getitem__((idx + 1) % len(self))

        label = self.df.iloc[idx, 1]

        #1. aplica transform (sempre vira tensor)
        image = self.transform(img)  # (C, H, W)

        #2. Fourier
        if self.fourier == 'true':
            image = self._fourier(image)

        elif self.fourier == 'concat':
            fft = self._fourier(image)
            image = torch.cat([image, fft], dim=0)

        return image, label, idx

    def _fourier(self, img: torch.Tensor):
        # img: (C, H, W)

        #converter para ciza
        if img.shape[0] == 3:
            img = img.mean(dim=0)  # (H, W)
        else:
            img = img.squeeze(0)

        img_np = img.numpy()

        # FFT
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)

        magnitude = np.log(np.abs(fshift) + 1)

        # normalização
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

        # tensor
        magnitude = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)

        return magnitude


        
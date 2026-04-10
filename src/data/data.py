from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms 
import os 


class ImageDataset(Dataset):
    def __init__(self, file_csv:Path, transform:transforms.Compose|None=None):
        self.df = pd.read_csv(file_csv, sep=',')
        
        #padrão da ImageNET -> outros modelos feitos em demais bases, possuem outros
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.data_dir_images = file_csv.absolute().parent
        
    def __len__(self):
        return len(self.df)
    
    
    #crio um iterador para pegar a imagem, label e a posição do dado
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        with Image.open(os.path.join(self.data_dir_images, img_path)) as img:
            img = img.convert('RGB')
            
        label = self.df.iloc[idx, 1]
        
        image = self.transform(img)
        
        return np.array(image), label, idx
    


        
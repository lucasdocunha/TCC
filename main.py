from src.data.data import ImageDataset
from pathlib import Path
from torchvision import transforms
import os


PWD = Path.cwd()

def main():

    data_img = '/media/ssd2/lucas.ocunha/datasets/phase1/testset'

    dataset = ImageDataset(
        file_csv=f'{PWD}/data/raw/test.csv',
	images_dir=data_img
    )

    print(dataset[0])


if __name__ == "__main__":
    main()

from src.data.data import ImageDataset
from pathlib import Path
from torchvision import transforms
import os



def main():
    data_dir = '/media/ssd2/lucas.ocunha/datasets/phase1'

    data_dir = Path.absolute().parent

    dataset = ImageDataset(
        file=os.path.join(data_dir, "trainset_label.txt")
    )

    print(dataset[0])


if __name__ == "__main__":
    main()

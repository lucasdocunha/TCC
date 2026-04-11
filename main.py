from src.pipelines.xcpetion import run_xception
from pathlib import Path
from torchvision import transforms
import os


PWD = Path.cwd()

def main():

    run_xception()


if __name__ == "__main__":
    main()

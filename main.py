"""
Treina todos os pipelines × `ALL_FOURIER_MODES`.

Ajuste EPOCHS, RAW_MIN, RUN_VIT e BATCH_SIZE abaixo.

ViT: só RGB; roda uma vez ao final se RUN_VIT for True.
"""

from __future__ import annotations
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False

import logging

from src.data import ALL_FOURIER_MODES
from src.pipelines.mobilenet import run_mobilenet
from src.pipelines.resnet import run_resnet
from src.pipelines.vit import run_vit
from src.pipelines.xcpetion import run_xception
from src.pipelines.clip import run_clip

logger = logging.getLogger(__name__)

EPOCHS = 50
RAW_MIN = True
RUN_VIT = True
BATCH_SIZE = 32
NUM_WORKERS = 4
MULTI_GPU = True


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    for mode in ALL_FOURIER_MODES:
        logger.info("======== Xception | input=%s ========", mode)
        run_xception(
            mode,
            epochs=EPOCHS,
            raw_min=RAW_MIN,
            data_limit=float("inf"),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pretrained=False,
            multi_gpu=MULTI_GPU,
        )

    for mode in ALL_FOURIER_MODES:
        logger.info("======== ResNet | input=%s ========", mode)
        run_resnet(
            epochs=EPOCHS,
            raw_min=RAW_MIN,
            architecture="resnet18",
            image_size=224,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pretrained=False,
            use_weighted_sampler=True,
            use_class_weights=False,
            train_layer3=True,
            threshold_strategy="accuracy",
            fourier=mode,
            data_limit=float("inf"),
            multi_gpu=MULTI_GPU,
        )

    for mode in ALL_FOURIER_MODES:
        logger.info("======== MobileNet | input=%s ========", mode)
        run_mobilenet(
            epochs=EPOCHS,
            raw_min=RAW_MIN,
            variant="large",
            input_mode=mode,
            image_size=224,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pretrained=False,
            use_weighted_sampler=True,
            use_class_weights=False,
            last_n_blocks=4,
            learning_rate_backbone=3e-5,
            threshold_metric="accuracy",
            data_limit=None,
            multi_gpu=MULTI_GPU,
        )

    if RUN_VIT:
        logger.info(
            "======== ViT | somente RGB (pipeline sem modos Fourier) ========"
        )
        run_vit(
            epochs=EPOCHS,
            raw_min=RAW_MIN,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            image_size=128,
            patch_size=16,
            hidden_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            dropout=0.25,
            threshold_metric="f1",
            mixup_alpha=0.2,
            multi_gpu=MULTI_GPU,
        )

        logger.info(
            "======== CLIP local | somente RGB (sem pesos externos) ========"
        )
        run_clip(
            epochs=EPOCHS,
            raw_min=RAW_MIN,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            multi_gpu=MULTI_GPU,
        )


if __name__ == "__main__":
    main()

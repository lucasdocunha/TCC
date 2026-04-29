"""
Treina todos os pipelines × `ALL_FOURIER_MODES`.

Uso:
  python main.py --smoke     # rápido: raw_min=True (./min_dataset/), poucas amostras, 1 época, 2 modos, sem ViT
  python main.py             # treino completo — ajuste EPOCHS, RAW_MIN e RUN_VIT abaixo

ViT: só RGB; roda no fim do modo completo (desligado no --smoke).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from src.data import ALL_FOURIER_MODES, FourierMode
from src.pipelines.mobilenet import run_mobilenet
from src.pipelines.resnet import run_resnet
from src.pipelines.vit import run_vit
from src.pipelines.xcpetion import run_xception

logger = logging.getLogger(__name__)

# --- Modo completo (python main.py) ---
EPOCHS = 20
RAW_MIN = False
RUN_VIT = True

# --- Modo smoke (python main.py --smoke) ---
SMOKE_MODES: tuple[FourierMode, ...] = ("none", "magnitude")
SMOKE_EPOCHS = 1
SMOKE_DATA_LIMIT = 64
SMOKE_BATCH = 8
SMOKE_NUM_WORKERS = min(4, os.cpu_count() or 1)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCC: modelos × representação de imagem")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Só validação local: poucos dados, 1 época, 2 modos Fourier, sem ViT",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.smoke:
        logger.warning(
            "MODO SMOKE: modos=%s, epochs=%d, data_limit=%d, raw_min=True (min_dataset/) — só teste",
            SMOKE_MODES,
            SMOKE_EPOCHS,
            SMOKE_DATA_LIMIT,
        )
        modes = SMOKE_MODES
        epochs = SMOKE_EPOCHS
        data_limit = SMOKE_DATA_LIMIT
        run_vit_flag = False
        batch = SMOKE_BATCH
        num_workers = SMOKE_NUM_WORKERS
        raw_min = True
    else:
        modes = ALL_FOURIER_MODES
        epochs = EPOCHS
        data_limit = float("inf")
        run_vit_flag = RUN_VIT
        batch = 24
        num_workers = 4
        raw_min = RAW_MIN

    for mode in modes:
        logger.info("======== Xception | input=%s ========", mode)
        run_xception(
            mode,
            epochs=epochs,
            raw_min=raw_min,
            data_limit=data_limit,
            batch_size=batch,
            num_workers=num_workers,
        )

    for mode in modes:
        logger.info("======== ResNet | input=%s ========", mode)
        run_resnet(
            epochs=epochs,
            raw_min=raw_min,
            architecture="resnet18",
            image_size=224,
            batch_size=batch,
            num_workers=num_workers,
            use_weighted_sampler=True,
            use_class_weights=False,
            train_layer3=True,
            threshold_strategy="accuracy",
            fourier=mode,
            data_limit=data_limit,
        )

    for mode in modes:
        logger.info("======== MobileNet | input=%s ========", mode)
        run_mobilenet(
            epochs=epochs,
            raw_min=raw_min,
            variant="large",
            input_mode=mode,
            image_size=224,
            batch_size=batch,
            num_workers=num_workers,
            use_weighted_sampler=data_limit == float("inf"),
            use_class_weights=False,
            last_n_blocks=4,
            learning_rate_backbone=3e-5,
            threshold_metric="accuracy",
            data_limit=None if data_limit == float("inf") else int(data_limit),
        )

    if run_vit_flag:
        logger.info(
            "======== ViT | somente RGB (pipeline sem modos Fourier) ========"
        )
        run_vit()


if __name__ == "__main__":
    main(sys.argv[1:])

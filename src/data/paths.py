"""Raiz das imagens no disco (mesma para raw e raw_min).

`raw_min` vs `raw` altera apenas os CSV em ``data/raw_min`` ou ``data/raw``;
os arquivos `.jpg` continuam nos splits phase1 abaixo.

Override opcional: variável de ambiente ``TCC_DATASET_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

_SPLIT_TO_SUBDIR = {"train": "trainset", "val": "valset", "test": "testset"}

_DEFAULT_ROOT = Path("/media/ssd2/lucas.ocunha/datasets/phase1")


def phase1_split_root(split: str) -> Path:
    base = Path(os.environ.get("TCC_DATASET_ROOT", str(_DEFAULT_ROOT)))
    return base / _SPLIT_TO_SUBDIR[split]

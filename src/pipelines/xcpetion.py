from src.data import ImageDataset, FourierMode
from src.models import xception
from src.plots import *

import logging
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import tqdm 
from pathlib import Path
import os

logger = logging.getLogger(__name__)

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score


def run_xception(
    fourier: FourierMode = "none",
    epochs=10,
    raw_min=True,
    data_limit: int | float = np.inf,
    batch_size: int = 32,
    num_workers: int = 4,
):

    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    logger.info("Iniciando run_xception")

    PWD = Path.cwd()
    BATCH = batch_size
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_WORKERS = num_workers
    PIN_MEMORY = DEVICE.type == "cuda"
    PERSISTENT_WORKERS = NUM_WORKERS > 0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    FOURIER = fourier

    train = ImageDataset(
        f"{PWD}/data/{'raw_min' if raw_min else 'raw'}/train.csv",
        '/media/ssd2/lucas.ocunha/datasets/phase1/trainset',
        transform,
        data_limit,
        FOURIER,
    )

    val = ImageDataset(
        f"{PWD}/data/{'raw_min' if raw_min else 'raw'}/val.csv",
        '/media/ssd2/lucas.ocunha/datasets/phase1/valset',
        transform,
        data_limit,
        FOURIER,
    )

    test = ImageDataset(
        f"{PWD}/data/{'raw_min' if raw_min else 'raw'}/test.csv",
        '/media/ssd2/lucas.ocunha/datasets/phase1/testset',
        transform,
        data_limit,
        FOURIER,
    )

    train_loader = DataLoader(train, batch_size=BATCH, num_workers=NUM_WORKERS,
                              persistent_workers=PERSISTENT_WORKERS,
                              pin_memory=PIN_MEMORY, shuffle=True)

    val_loader = DataLoader(val, batch_size=BATCH, num_workers=NUM_WORKERS,
                            persistent_workers=PERSISTENT_WORKERS,
                            pin_memory=PIN_MEMORY, shuffle=False)

    test_loader = DataLoader(test, batch_size=BATCH, num_workers=NUM_WORKERS,
                             persistent_workers=PERSISTENT_WORKERS,
                             pin_memory=PIN_MEMORY, shuffle=False)

    sample_x, _, _ = train[0]
    in_channels = sample_x.shape[0]

    model = xception(pretrained=True, in_channels=in_channels)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.block12.parameters():
        param.requires_grad = True

    for param in model.conv3.parameters():
        param.requires_grad = True

    for param in model.conv4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(2048, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {'params': model.fc.parameters(), 'lr': 1e-3},
        {'params': model.block12.parameters(), 'lr': 1e-4},
        {'params': model.conv3.parameters(), 'lr': 1e-4},
        {'params': model.conv4.parameters(), 'lr': 1e-4},
    ])

    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=4
    )

    best_val_loss = float('inf')
    best_path = f"models/xception/weights/best.pth"
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    # =========================
    # TREINO
    # =========================
    for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):

        model.train()
        train_loss = 0

        for img, label, _ in train_loader:
            x = img.to(DEVICE)
            y = label.to(DEVICE)

            # 🔥 proteção
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            x = torch.clamp(x, -5, 5)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                out = model(x)
                out = torch.clamp(out, -20, 20)  # 🔥 evita explosão
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # =========================
        # VALIDAÇÃO
        # =========================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                x = torch.clamp(x, -5, 5)

                out = model(x)
                out = torch.clamp(out, -20, 20)

                loss = criterion(out, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    # =========================
    # TESTE
    # =========================
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    y_true, y_pred = [], []
    all_logits = []

    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            x = torch.clamp(x, -5, 5)

            out = model(x)
            out = torch.clamp(out, -20, 20)

            preds = out.argmax(1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # float32: softmax em float16 estoura exp(logit) para logits ~> ~11.5
            all_logits.append(out.detach().float().cpu())

    logits = torch.cat(all_logits, dim=0)

    probs = torch.softmax(logits, dim=1)

    probs = probs[:, 1].cpu().numpy()

    print("NaN em probs?", np.isnan(probs).any())
    print("Min/Max probs:", probs.min(), probs.max())

    results = {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, probs)
    }

    print(results)
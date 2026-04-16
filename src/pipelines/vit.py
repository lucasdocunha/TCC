from src.data import ImageDataset
from src.plots import *

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import transforms

import pandas as pd
import numpy as np
import tqdm
from pathlib import Path
import os

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score
from scipy.special import softmax

logger = logging.getLogger(__name__)


class ViTTransform:
    def __init__(self, pretrained: str, augment: bool = False):
        self.processor = ViTImageProcessor.from_pretrained(pretrained)
        self.augment = augment
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    def __call__(self, img):
        if self.augment:
            img = self.aug(img)
        return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


def collate_fn(batch):
    imgs, labels, idxs = zip(*batch)
    imgs = torch.stack([torch.tensor(i) if not isinstance(i, torch.Tensor) else i for i in imgs])
    labels = torch.tensor(labels, dtype=torch.long)
    idxs = torch.tensor(idxs, dtype=torch.long)
    return imgs, labels, idxs


def run_vit():

    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    logger.info("Iniciando run_vit (pipeline ViT).")

    PWD = Path.cwd()
    BATCH = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "vit"
    PRETRAINED = "google/vit-base-patch16-224"
    EARLY_STOP_PATIENCE = 15
    logger.info("Dispositivo: %s | batch_size=%d | cwd=%s", DEVICE, BATCH, PWD)

    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    logger.info("DataLoader: num_workers=%d, pin_memory=%s, persistent_workers=%s", NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS)

    vit_transform_train = ViTTransform(PRETRAINED, augment=True)
    vit_transform_eval  = ViTTransform(PRETRAINED, augment=False)

    logger.info("Carregando ImageDataset (train, val, test)")
    train = ImageDataset(
        file_csv=f"{PWD}/data/raw_min/train.csv",
        images_dir=f"{PWD}/min_dataset/train",
        transform=vit_transform_train,
    )
    val = ImageDataset(
        file_csv=f"{PWD}/data/raw_min/val.csv",
        images_dir=f"{PWD}/min_dataset/val",
        transform=vit_transform_eval,
    )
    test = ImageDataset(
        file_csv=f"{PWD}/data/raw_min/test.csv",
        images_dir=f"{PWD}/min_dataset/test",
        transform=vit_transform_eval,
    )
    logger.info("Datasets prontos: amostras train=%d, val=%d, test=%d", len(train), len(val), len(test))

    train_df = pd.read_csv(f"{PWD}/data/raw_min/train.csv")
    class_counts = train_df["target"].value_counts().sort_index()
    sample_weights = train_df["target"].map({0: 1.0 / class_counts[0], 1: 1.0 / class_counts[1]}).values
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(train_df),
        replacement=True,
    )
    logger.info("WeightedRandomSampler criado: class_counts=%s", class_counts.to_dict())

    train_loader = DataLoader(
        train,
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=PIN_MEMORY,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val,
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test,
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        collate_fn=collate_fn,
    )
    logger.info("DataLoaders criados: batches train=%d, val=%d, test=%d", len(train_loader), len(val_loader), len(test_loader))

    logger.info("Instanciando ViT pré-treinado e configurando transfer learning")
    model = ViTForImageClassification.from_pretrained(
        PRETRAINED,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.vit.encoder.layer[-1].parameters():
        param.requires_grad = True
    for param in model.vit.encoder.layer[-2].parameters():
        param.requires_grad = True
    for param in model.vit.encoder.layer[-3].parameters():
        param.requires_grad = True
    for param in model.vit.layernorm.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    logger.info("Últimos 3 encoder blocks, layernorm e classifier liberados; demais congelados.")

    model = model.to(DEVICE)
    logger.info("Modelo enviado para %s.", DEVICE)

    weights = torch.tensor(
        [1000 / class_counts[0], 1000 / class_counts[1]],
        dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    logger.info("CrossEntropyLoss com pesos: classe0=%.4f classe1=%.4f", weights[0], weights[1])

    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(),                "lr": 1e-4,  "weight_decay": 1e-2},
        {"params": model.vit.layernorm.parameters(),             "lr": 1e-5,  "weight_decay": 1e-2},
        {"params": model.vit.encoder.layer[-1].parameters(),     "lr": 1e-5,  "weight_decay": 1e-2},
        {"params": model.vit.encoder.layer[-2].parameters(),     "lr": 5e-6,  "weight_decay": 1e-2},
        {"params": model.vit.encoder.layer[-3].parameters(),     "lr": 5e-6,  "weight_decay": 1e-2},
    ])

    scaler = torch.amp.GradScaler("cuda")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )
    logger.info("AdamW, CrossEntropyLoss ponderada, AMP GradScaler e ReduceLROnPlateau configurados.")

    num_epochs = 100
    best_val_auc = 0.0
    epochs_no_improve = 0
    best_path = f"models/{MODEL_NAME}/weights/best_{MODEL_NAME}.pth"
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    logger.info("Iniciando treinamento: %d épocas. Melhor modelo salvo em %s", num_epochs, best_path)

    epoch_bar = tqdm.tqdm(range(num_epochs), desc="Epochs")
    for epoch in epoch_bar:

        # --- treino ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        train_pbar = tqdm.tqdm(train_loader, desc=f"Train ep {epoch+1}/{num_epochs}", leave=False)
        for img, label, idx in train_pbar:
            x = img.to(DEVICE)
            y = label.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                out = model(pixel_values=x).logits
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # --- validação ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_probs, val_true = [], []

        with torch.no_grad():
            val_pbar = tqdm.tqdm(val_loader, desc=f"Val ep {epoch+1}/{num_epochs}", leave=False)
            for x, y, _ in val_pbar:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                with torch.amp.autocast("cuda"):
                    out = model(pixel_values=x).logits
                    loss = criterion(out, y)

                val_loss += loss.item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

                probs = torch.softmax(out, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_true.extend(y.cpu().numpy())
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_true, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            logger.info(
                "Época %d/%d: novo melhor val_auc=%.6f — checkpoint salvo em %s",
                epoch + 1, num_epochs, val_auc, best_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping ativado na época %d.", epoch + 1)
                break

        scheduler.step(1 - val_auc)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}", train_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",    val_acc=f"{val_acc:.4f}",
            val_auc=f"{val_auc:.4f}",
        )
        logger.info(
            "Época %d/%d concluída | train_loss=%.6f train_acc=%.6f | val_loss=%.6f val_acc=%.6f | val_auc=%.6f",
            epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc, val_auc,
        )

    # --- teste ---
    logger.info("Treino finalizado. Carregando melhores pesos de %s para avaliação no teste.", best_path)
    model.load_state_dict(torch.load(best_path))
    model.eval()

    y_true, y_pred = [], []
    all_logits, all_ids = [], []

    with torch.no_grad():
        for x, y, idx in tqdm.tqdm(test_loader, desc="Teste (inferência)"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            with torch.amp.autocast("cuda"):
                out = model(pixel_values=x).logits

            preds = out.argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            all_logits.append(out.detach().cpu().numpy().astype(np.float32))
            all_ids.extend(idx.cpu().numpy())

    logits = np.concatenate(all_logits)
    probs  = softmax(logits, axis=1)[:, 1]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    test_results = {
        "acc":         accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred),
        "recall":      recall_score(y_true, y_pred),
        "f1":          f1_score(y_true, y_pred),
        "auc":         roc_auc_score(y_true, probs),
        "specificity": specificity,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "y_true":  y_true,
        "y_pred":  y_pred,
        "logits":  logits,
        "ids":     np.array(all_ids),
    }
    logger.info(
        "Métricas no teste: acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f specificity=%.4f | tp=%d fp=%d fn=%d tn=%d",
        test_results["acc"], test_results["precision"], test_results["recall"],
        test_results["f1"],  test_results["auc"],       test_results["specificity"],
        tp, fp, fn, tn,
    )

    model_dir = f"{PWD}/models/{MODEL_NAME}"
    os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)

    np.savez_compressed(
        f"{model_dir}/results/outputs.npz",
        logits=test_results["logits"],
        ids=test_results["ids"],
    )
    torch.save(model.state_dict(), f"{model_dir}/weights/{MODEL_NAME}.pth")
    logger.info("Pesos finais salvos em models/%s/weights/%s.pth", MODEL_NAME, MODEL_NAME)

    logger.info("Gerando gráficos (matriz de confusão e ROC-AUC)")
    plot_confusion_matrix(test_results, model_dir, f"{MODEL_NAME} Confusion Matrix")
    plot_roc_auc(test_results, model_dir, f"{MODEL_NAME} ROC-AUC Curve")

    extra_info = {
        "Camadas descongeladas": 3,
        "Nº épocas": num_epochs,
    }
    save_metrics_csv(test_results, model_dir, extra_info=extra_info)
    logger.info("run_vit concluído (métricas exportadas e plots salvos).")

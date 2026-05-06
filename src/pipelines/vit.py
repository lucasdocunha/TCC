from __future__ import annotations
import logging
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from src.data import FourierMode, ImageDataset
from src.data.paths import phase1_split_root
from src.plots import plot_confusion_matrix, plot_roc_auc, save_metrics_csv

logger = logging.getLogger(__name__)

PRETRAINED = "google/vit-base-patch16-224"
IMAGE_SIZE = 224


class ViTTransform:
    """Transform aplicado às imagens antes de entrar no ViT.

    Para ``fourier='none'`` delega ao ``ViTImageProcessor`` do HuggingFace.
    Para demais modos usa uma pipeline manual de torchvision, pois o processor
    não aceita tensores com número de canais diferente de 3.
    """

    _RGB_MODES: frozenset[FourierMode] = frozenset({"none"})

    def __init__(self, fourier: FourierMode, augment: bool = False) -> None:
        """Inicializa o transform conforme o modo Fourier e flag de augmentation.

        Args:
            fourier: Modo de entrada da imagem (espacial ou frequência).
            augment: Se ``True``, aplica augmentações geométricas e de cor.
        """
        self.fourier = fourier
        self.augment = augment
        if fourier in self._RGB_MODES:
            self.processor = ViTImageProcessor.from_pretrained(PRETRAINED)
            self._use_processor = True
        else:
            self.processor = None
            self._use_processor = False
            self._pil_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
            ]) if augment else None
        self._aug_pil = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]) if augment else None

    def __call__(self, img) -> torch.Tensor:
        """Transforma uma imagem PIL no tensor esperado pelo ViT.

        Args:
            img: Imagem PIL RGB.

        Returns:
            Tensor ``(C, H, W)`` pronto para o modelo.
        """
        if self._use_processor:
            if self._aug_pil is not None:
                img = self._aug_pil(img)
            return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        if self._pil_aug is not None:
            img = self._pil_aug(img)
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(img)


def _expand_to_3ch(x: torch.Tensor) -> torch.Tensor:
    """Expande tensores com canais != 3 para 3 canais por repetição.

    Args:
        x: Tensor ``(B, C, H, W)``.

    Returns:
        Tensor ``(B, 3, H, W)``.
    """
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] == 2:
        return torch.cat([x, x[:, :1, :, :]], dim=1)
    return x[:, :3, :, :]


def _collate_fn(batch) -> tuple:
    """Agrega uma lista de amostras em tensores batched.

    Args:
        batch: Lista de tuplas ``(imagem, label, idx)``.

    Returns:
        Tupla ``(imgs, labels, idxs)`` como tensores.
    """
    imgs, labels, idxs = zip(*batch)
    imgs = torch.stack([i if isinstance(i, torch.Tensor) else torch.tensor(i) for i in imgs])
    labels = torch.tensor(labels, dtype=torch.long)
    idxs = torch.tensor(idxs, dtype=torch.long)
    return imgs, labels, idxs


def _device() -> torch.device:
    """Retorna ``cuda:0`` se disponível, senão ``cpu``."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _amp_ctx(device: torch.device):
    """Retorna contexto de autocast para CUDA ou nullcontext para CPU."""
    return torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()


def _class_weights(csv_path: Path) -> tuple:
    """Calcula contagens por classe e cria um ``WeightedRandomSampler``.

    Args:
        csv_path: Caminho para o CSV com coluna ``target``.

    Returns:
        Tupla ``(class_counts, sampler)``.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    counts = df["target"].value_counts().sort_index()
    sample_w = df["target"].map({0: 1.0 / counts[0], 1: 1.0 / counts[1]}).values
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.float),
        num_samples=len(df),
        replacement=True,
    )
    return counts, sampler


def run_vit(
    fourier: FourierMode = "none",
    epochs: int = 50,
    raw_min: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    early_stop_patience: int = 15,
    output_root: str | Path | None = None,
    seed: int = 42,
) -> dict:
    """Treina e avalia o ViT-Base para um modo de entrada Fourier.

    Para modos com canais != 3, o tensor é expandido para 3 canais antes
    de entrar no modelo, preservando os pesos pré-treinados do patch embedding.
    Resultados, pesos e plots são salvos em ``models/vit/<fourier>/``.

    Args:
        fourier: Modo de entrada (``'none'``, ``'magnitude'``, ``'phase'``, etc.).
        epochs: Número máximo de épocas de treino.
        raw_min: Se ``True``, usa CSVs de ``data/raw_min/``; caso contrário ``data/raw/``.
        batch_size: Tamanho do batch.
        num_workers: Workers do DataLoader.
        early_stop_patience: Épocas sem melhora no val AUC antes de parar.
        output_root: Diretório raiz para salvar artefatos. Padrão: ``Path.cwd()``.
        seed: Semente para reprodutibilidade.

    Returns:
        Dicionário com métricas e arrays do conjunto de teste.
    """
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    torch.manual_seed(seed)
    np.random.seed(seed)
    pwd = Path.cwd()
    output_root = Path(output_root) if output_root is not None else pwd
    device = _device()
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    model_name = "vit"
    model_dir = output_root / "models" / model_name / fourier
    (model_dir / "weights").mkdir(parents=True, exist_ok=True)
    (model_dir / "results").mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "weights" / f"best_{model_name}.pth"
    data_dir = pwd / "data" / ("raw_min" if raw_min else "raw")
    spatial_size = (IMAGE_SIZE, IMAGE_SIZE) if fourier != "none" else None
    logger.info("run_vit | fourier=%s device=%s batch=%d", fourier, device, batch_size)
    tf_train = ViTTransform(fourier, augment=True)
    tf_eval = ViTTransform(fourier, augment=False)
    train_ds = ImageDataset(
        file_csv=data_dir / "train.csv", images_dir=phase1_split_root("train"),
        transform=tf_train, fourier=fourier, spatial_size=spatial_size,
    )
    val_ds = ImageDataset(
        file_csv=data_dir / "val.csv", images_dir=phase1_split_root("val"),
        transform=tf_eval, fourier=fourier, spatial_size=spatial_size,
    )
    test_ds = ImageDataset(
        file_csv=data_dir / "test.csv", images_dir=phase1_split_root("test"),
        transform=tf_eval, fourier=fourier, spatial_size=spatial_size,
    )
    sample_x, _, _ = train_ds[0]
    in_channels = sample_x.shape[0]
    logger.info("Datasets: train=%d val=%d test=%d | in_channels=%d", len(train_ds), len(val_ds), len(test_ds), in_channels)
    class_counts, sampler = _class_weights(data_dir / "train.csv")
    loader_kw = dict(
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, collate_fn=_collate_fn,
    )
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)
    model = ViTForImageClassification.from_pretrained(PRETRAINED, num_labels=2, ignore_mismatched_sizes=True)
    for param in model.parameters():
        param.requires_grad = False
    for i in [-1, -2, -3]:
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = True
    for param in model.vit.layernorm.parameters():
        param.requires_grad = True
    model.classifier = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(model.config.hidden_size, 2))
    model = model.to(device)
    w0 = float(np.sqrt(1000 / class_counts[0]))
    w1 = float(np.sqrt(1000 / class_counts[1]))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([w0, w1], dtype=torch.float).to(device))
    logger.info("CrossEntropyLoss pesos: classe0=%.4f classe1=%.4f", w0, w1)
    param_groups = [
        {"params": model.classifier.parameters(), "lr": 1e-4, "weight_decay": 1e-2},
        {"params": model.vit.layernorm.parameters(), "lr": 1e-5, "weight_decay": 1e-2},
        {"params": model.vit.encoder.layer[-1].parameters(), "lr": 1e-5, "weight_decay": 1e-2},
        {"params": model.vit.encoder.layer[-2].parameters(), "lr": 5e-6, "weight_decay": 1e-2},
        {"params": model.vit.encoder.layer[-3].parameters(), "lr": 5e-6, "weight_decay": 1e-2},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    best_val_auc = 0.0
    epochs_no_improve = 0
    for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for img, label, _ in tqdm.tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False):
            x, y = img.to(device), label.to(device)
            x = _expand_to_3ch(x)
            optimizer.zero_grad(set_to_none=True)
            with _amp_ctx(device):
                out = model(pixel_values=x).logits
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_probs, val_true = [], []
        with torch.no_grad():
            for x, y, _ in tqdm.tqdm(val_loader, desc=f"Val {epoch+1}/{epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                x = _expand_to_3ch(x)
                with _amp_ctx(device):
                    out = model(pixel_values=x).logits
                    loss = criterion(out, y)
                val_loss += loss.item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
                val_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                val_true.extend(y.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_true, val_probs)
        scheduler.step(1 - val_auc)
        logger.info(
            "Época %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_auc=%.4f",
            epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc, val_auc,
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            logger.info("Novo melhor val_auc=%.6f -> checkpoint salvo.", val_auc)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info("Early stopping na época %d.", epoch + 1)
                break
    logger.info("Carregando melhor checkpoint de %s.", best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    y_true, y_pred, all_logits, all_ids = [], [], [], []
    with torch.no_grad():
        for x, y, idx in tqdm.tqdm(test_loader, desc="Teste"):
            x, y = x.to(device), y.to(device)
            x = _expand_to_3ch(x)
            with _amp_ctx(device):
                out = model(pixel_values=x).logits
            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())
            all_logits.append(out.detach().cpu().numpy().astype(np.float32))
            all_ids.extend(idx.cpu().numpy())
    logits = np.concatenate(all_logits)
    probs = softmax(logits, axis=1)[:, 1]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    test_results = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, probs),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "y_true": y_true, "y_pred": y_pred,
        "logits": logits, "ids": np.array(all_ids),
    }
    logger.info(
        "Métricas no teste: acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f specificity=%.4f | tp=%d fp=%d fn=%d tn=%d",
        test_results["acc"], test_results["precision"], test_results["recall"], test_results["f1"],
        test_results["auc"], test_results["specificity"], tp, fp, fn, tn,
    )
    np.savez_compressed(
        model_dir / "results" / "outputs.npz",
        logits=logits, ids=test_results["ids"], probs=probs, y_true=y_true, y_pred=y_pred,
    )
    torch.save(model.state_dict(), model_dir / "weights" / f"{model_name}.pth")
    plot_confusion_matrix(test_results, str(model_dir), f"ViT [{fourier}] Confusion Matrix")
    plot_roc_auc(test_results, str(model_dir), f"ViT [{fourier}] ROC-AUC Curve")
    save_metrics_csv(
        test_results, str(model_dir),
        extra_info={
            "fourier": fourier,
            "in_channels": in_channels,
            "image_size": IMAGE_SIZE,
            "epochs": epochs,
            "best_val_auc": best_val_auc,
            "pretrained": PRETRAINED,
        },
    )
    logger.info("run_vit [%s] concluído.", fourier)
    return test_results

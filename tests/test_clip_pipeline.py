from pathlib import Path

import torch


def test_clip_classifier_tiny_config_forward_shape():
    from src.models.clip import CLIPVisionClassifier

    model = CLIPVisionClassifier(
        num_classes=2,
        image_size=64,
        patch_size=16,
        hidden_size=32,
        projection_dim=16,
        num_hidden_layers=1,
        num_attention_heads=4,
    )
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))

    assert out.shape == (2, 2)


def test_clip_pipeline_tiny_run_writes_metrics(tmp_path, tiny_short_split_dataset):
    from src.pipelines.clip import run_clip

    results = run_clip(
        epochs=1,
        raw_min=True,
        data_limit=8,
        output_root=tmp_path,
        batch_size=4,
        num_workers=0,
        image_size=64,
        patch_size=16,
        hidden_size=32,
        projection_dim=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        train_backbone=False,
        augment=False,
    )

    metrics_path = (
        Path(tmp_path)
        / "models"
        / "clip"
        / "clip_vit_scratch"
        / "none_limit8"
        / "results"
        / "metrics_summary.csv"
    )
    assert 0.0 <= results["acc"] <= 1.0
    assert metrics_path.exists()

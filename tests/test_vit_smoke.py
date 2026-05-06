import torch
from pathlib import Path


def test_vit_pipeline_import_does_not_require_transformers():
    from src.pipelines.vit import collate_fn, run_vit

    assert callable(run_vit)
    assert callable(collate_fn)


def test_vit_pipeline_has_no_external_pretrained_loading_api():
    import inspect

    from src.pipelines.vit import run_vit

    assert "pretrained_model" not in inspect.signature(run_vit).parameters


def test_vit_collate_fn_stacks_tensor_batches():
    from src.pipelines.vit import collate_fn

    batch = [
        (torch.zeros(3, 224, 224), 0, 3),
        (torch.ones(3, 224, 224), 1, 4),
    ]

    imgs, labels, idxs = collate_fn(batch)

    assert imgs.shape == (2, 3, 224, 224)
    assert labels.tolist() == [0, 1]
    assert idxs.tolist() == [3, 4]


def test_vit_model_accepts_frequency_channel_counts():
    from src.models.vit import VisionTransformerClassifier

    for channels in (1, 2, 4, 6):
        model = VisionTransformerClassifier(
            image_size=64,
            patch_size=16,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            in_channels=channels,
        )
        out = model(torch.zeros(2, channels, 64, 64))
        assert out.shape == (2, 2)


def test_vit_pipeline_tiny_run_writes_metrics(tmp_path, tiny_short_split_dataset):
    from src.pipelines.vit import run_vit

    results = run_vit(
        epochs=1,
        raw_min=True,
        data_limit=8,
        output_root=tmp_path,
        batch_size=4,
        num_workers=0,
        image_size=64,
        patch_size=16,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        train_backbone=True,
        augment=False,
        multi_gpu=False,
    )

    metrics_path = (
        Path(tmp_path)
        / "models"
        / "vit"
        / "vit_scratch"
        / "none_limit8"
        / "results"
        / "metrics_summary.csv"
    )
    assert 0.0 <= results["acc"] <= 1.0
    assert metrics_path.exists()


def test_vit_pipeline_tiny_frequency_run_writes_metrics(
    tmp_path, tiny_short_split_dataset
):
    from src.pipelines.vit import run_vit

    results = run_vit(
        fourier="frequency_3",
        epochs=1,
        raw_min=True,
        data_limit=8,
        output_root=tmp_path,
        batch_size=4,
        num_workers=0,
        image_size=64,
        patch_size=16,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        train_backbone=True,
        augment=False,
        multi_gpu=False,
    )

    metrics_path = (
        Path(tmp_path)
        / "models"
        / "vit"
        / "vit_scratch"
        / "frequency_3_limit8"
        / "results"
        / "metrics_summary.csv"
    )
    assert 0.0 <= results["f1"] <= 1.0
    assert metrics_path.exists()

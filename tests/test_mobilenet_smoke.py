from pathlib import Path

import pandas as pd
import pytest
import torch


def test_mobilenet_accepts_rgb_frequency_and_concat_channels():
    from src.models.mobilenet import mobilenetv3_small

    for in_channels in (1, 3, 6):
        model = mobilenetv3_small(num_classes=2, in_channels=in_channels, pretrained=False)
        model.eval()

        with torch.no_grad():
            out = model(torch.randn(2, in_channels, 128, 128))

        assert out.shape == (2, 2)


def test_mobilenet_rejects_external_pretrained_weights():
    from src.models.mobilenet import mobilenetv3_small

    with pytest.raises(ValueError, match="pretrained"):
        mobilenetv3_small(num_classes=2, pretrained=True)


def test_mobilenet_pipeline_tiny_run_writes_metrics(tmp_path, tiny_phase1_dataset):
    from src.pipelines.mobilenet import run_mobilenet

    results = run_mobilenet(
        input_mode="concat_frequency",
        epochs=1,
        data_limit=16,
        output_root=tmp_path,
        batch_size=8,
        num_workers=0,
        pretrained=False,
        image_size=128,
        warmup_epochs=0,
        last_n_blocks=1,
    )

    metrics_path = (
        Path(tmp_path)
        / "models"
        / "mobilenet"
        / "mobilenetv3_small"
        / "concat_frequency_limit16"
        / "results"
        / "metrics_summary.csv"
    )
    assert 0.0 <= results["acc"] <= 1.0
    assert metrics_path.exists()


def test_resnet_pipeline_accepts_six_channel_concat_frequency(
    tmp_path, tiny_phase1_dataset
):
    from src.pipelines.resnet import run_resnet

    run_resnet(
        epochs=1,
        raw_min=True,
        fourier="concat_frequency",
        data_limit=16,
        output_root=tmp_path,
        batch_size=8,
        num_workers=0,
        pretrained=False,
        architecture="resnet18",
        image_size=128,
        train_backbone=False,
    )

    metrics_path = Path(tmp_path) / "models" / "resnet" / "concat_frequency" / "results" / "metrics_summary.csv"
    metrics = pd.read_csv(metrics_path)
    assert int(metrics.loc[0, "in_channels"]) == 6

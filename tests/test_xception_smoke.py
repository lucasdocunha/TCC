from pathlib import Path

import torch


def test_xception_accepts_rgb_frequency_and_concat_channels():
    from src.models.xception import xception

    for in_channels in (1, 3, 6):
        model = xception(pretrained=False, in_channels=in_channels, num_classes=2)
        model.eval()

        with torch.no_grad():
            out = model(torch.randn(2, in_channels, 128, 128))

        assert out.shape == (2, 2)


def test_xception_pipeline_tiny_run_writes_metrics(tmp_path, tiny_phase1_dataset):
    from src.pipelines.xcpetion import run_xception

    results = run_xception(
        fourier="concat_frequency",
        epochs=1,
        raw_min=True,
        data_limit=8,
        output_root=tmp_path,
        batch_size=4,
        num_workers=0,
        pretrained=False,
        image_size=128,
    )

    metrics_path = (
        Path(tmp_path)
        / "models"
        / "xception"
        / "concat_frequency_limit8"
        / "results"
        / "metrics_summary.csv"
    )
    assert 0.0 <= results["acc"] <= 1.0
    assert metrics_path.exists()

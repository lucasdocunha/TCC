from pathlib import Path

import torch


def test_resnet_classifier_forward_shape():
    from src.models import resnet

    model = resnet(num_classes=2, pretrained=False, architecture="resnet18")
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(2, 3, 128, 128))

    assert out.shape == (2, 2)


def test_resnet_classifier_accepts_extra_input_channels():
    from src.models.resnet import resnet

    model = resnet(num_classes=2, pretrained=False, architecture="resnet18", in_channels=4)
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(2, 4, 128, 128))

    assert out.shape == (2, 2)


def test_resnet_pipeline_tiny_run_writes_metrics(tmp_path, tiny_phase1_dataset):
    from src.pipelines.resnet import run_resnet

    results = run_resnet(
        epochs=1,
        raw_min=True,
        data_limit=16,
        output_root=tmp_path,
        batch_size=8,
        num_workers=0,
        pretrained=False,
        architecture="resnet18",
        image_size=128,
        train_backbone=False,
    )

    assert 0.0 <= results["acc"] <= 1.0
    assert (Path(tmp_path) / "models" / "resnet" / "results" / "metrics_summary.csv").exists()

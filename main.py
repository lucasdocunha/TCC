from src.pipelines.mobilenet import run_mobilenet
from src.pipelines.resnet import run_resnet


if __name__ == "__main__":
    run_resnet(
        epochs=20,
        raw_min=False,
        architecture="resnet18",
        image_size=224,
        batch_size=24,
        use_weighted_sampler=True,
        use_class_weights=False,
        train_layer3=True,
        threshold_strategy="accuracy",
    )

    run_mobilenet(
        epochs=20,
        raw_min=False,
        variant="large",
        input_mode="none",
        image_size=224,
        batch_size=24,
        use_weighted_sampler=True,
        use_class_weights=False,
        last_n_blocks=4,
        learning_rate_backbone=3e-5,
        threshold_metric="accuracy",
    )

import torch.nn as nn
from torchvision import models


_ARCHITECTURES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
    "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
}


def resnet(
    num_classes: int = 2,
    pretrained: bool = True,
    architecture: str = "resnet18",
    dropout: float = 0.2,
    in_channels: int = 3,
) -> nn.Module:
    if architecture not in _ARCHITECTURES:
        valid = ", ".join(sorted(_ARCHITECTURES))
        raise ValueError(f"architecture must be one of: {valid}")

    builder, weights_enum = _ARCHITECTURES[architecture]
    weights = weights_enum if pretrained else None
    model = builder(weights=weights)

    if in_channels != 3:
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            old_weight = old_conv.weight.data
            if in_channels == 1:
                new_weight = old_weight.mean(dim=1, keepdim=True)
            elif in_channels > 3:
                repeat_factor = (in_channels + 2) // 3
                new_weight = old_weight.repeat(1, repeat_factor, 1, 1)[:, :in_channels, :, :]
                new_weight = new_weight * (3.0 / float(in_channels))
            else:
                new_weight = old_weight[:, :in_channels, :, :]
                new_weight = new_weight * (3.0 / float(in_channels))
            new_conv.weight.data.copy_(new_weight)
        model.conv1 = new_conv

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_last_blocks(model: nn.Module, train_layer3: bool = False) -> None:
    for param in model.layer4.parameters():
        param.requires_grad = True

    if train_layer3:
        for param in model.layer3.parameters():
            param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

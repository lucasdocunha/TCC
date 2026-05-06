from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


_VARIANTS = {
    "small": models.mobilenet_v3_small,
    "large": models.mobilenet_v3_large,
}


def _adapt_first_conv(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    first_conv = model.features[0][0]
    if not isinstance(first_conv, nn.Conv2d):
        raise TypeError("Expected MobileNet first layer to be Conv2d.")
    if first_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
    )

    if pretrained:
        with torch.no_grad():
            old_weight = first_conv.weight
            if in_channels == 1:
                new_weight = old_weight.mean(dim=1, keepdim=True)
            elif in_channels > 3:
                repeat_factor = (in_channels + 2) // 3
                new_weight = old_weight.repeat(1, repeat_factor, 1, 1)[:, :in_channels]
                new_weight = new_weight * (3.0 / float(in_channels))
            else:
                new_weight = old_weight[:, :in_channels]
                new_weight = new_weight * (3.0 / float(in_channels))

            new_conv.weight.copy_(new_weight)
            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

    model.features[0][0] = new_conv


def mobilenet(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    variant: str = "small",
    dropout: float | None = None,
) -> nn.Module:
    if variant not in _VARIANTS:
        valid = ", ".join(sorted(_VARIANTS))
        raise ValueError(f"variant must be one of: {valid}")

    if pretrained:
        raise ValueError("External pretrained MobileNet weights are disabled for this project.")
    builder = _VARIANTS[variant]
    model = builder(weights=None)
    _adapt_first_conv(model, in_channels, pretrained=pretrained)

    last_linear = model.classifier[-1]
    if not isinstance(last_linear, nn.Linear):
        raise TypeError("Expected MobileNet classifier to end with Linear.")
    if dropout is not None:
        for module in model.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
    model.classifier[-1] = nn.Linear(last_linear.in_features, num_classes)
    return model


def mobilenetv3_small(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    dropout: float | None = None,
) -> nn.Module:
    return mobilenet(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        variant="small",
        dropout=dropout,
    )


def mobilenetv3_large(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    dropout: float | None = None,
) -> nn.Module:
    return mobilenet(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        variant="large",
        dropout=dropout,
    )


def freeze_classifier_only(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, last_n_blocks: int = 3) -> None:
    for block in model.features[-last_n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

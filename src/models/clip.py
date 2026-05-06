from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        in_channels: int = 3,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class CLIPVisionClassifier(nn.Module):
    """CLIP-style vision encoder trained locally from scratch.

    This intentionally does not load OpenAI/Hugging Face weights. It mirrors the
    image-side CLIP structure used for classification: patch tokens, class token,
    transformer blocks, projection, and a supervised classifier head.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.2,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 256,
        projection_dim: int = 128,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.patch_embed = PatchEmbedding(image_size, patch_size, hidden_size)
        num_tokens = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_hidden_layers,
            norm=nn.LayerNorm(hidden_size),
        )
        self.visual_proj = nn.Linear(hidden_size, projection_dim)
        hidden_head = max(projection_dim // 2, num_classes)
        self.head = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, hidden_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        projected = self.visual_proj(x[:, 0])
        return self.head(projected)

    def freeze_backbone(self) -> None:
        for module in (self.patch_embed, self.encoder, self.visual_proj):
            for param in module.parameters():
                param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 2) -> None:
        layers = self.encoder.layers
        n = max(0, min(n, len(layers)))
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.encoder.norm.parameters():
            param.requires_grad = True
        for param in self.visual_proj.parameters():
            param.requires_grad = True
        self.cls_token.requires_grad = True
        self.pos_embed.requires_grad = True


def clip_safe_name() -> str:
    return "clip_vit_scratch"


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

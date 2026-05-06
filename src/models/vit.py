from __future__ import annotations

import torch
import torch.nn as nn

from src.models.clip import PatchEmbedding


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
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
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
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
        return self.classifier(x[:, 0])

    def freeze_backbone(self) -> None:
        for module in (self.patch_embed, self.encoder):
            for param in module.parameters():
                param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

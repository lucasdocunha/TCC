import torch
import torch.nn as nn
from torchvision import transforms 


class CLIPVisionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        from transformers import CLIPModel

        # Carrega só o vision encoder do CLIP
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.backbone = clip.vision_model
        self.visual_proj = clip.visual_projection  # projeta 1024 → 768

        # Congela o backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.visual_proj.parameters():
            param.requires_grad = False

        # Cabeça de classificação
        embed_dim = clip.config.projection_dim  # 768
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(), #ViT usa GELU por padro
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.pooler_output          # [B, 1024]
        projected = self.visual_proj(pooled)    # [B, 768]
        return self.head(projected)

    def unfreeze_last_n_layers(self, n=4):
        """Descongela as últimas N camadas do transformer para fine-tuning parcial"""
        layers = self.backbone.encoder.layers
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True


# 3. TRANSFORMS
# CLIP ViT-L/14 espera 224x224, normalizado com esses valores
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

import timm
import torch
import torch.nn as nn
from typing import Any
import torchvision.models as models

class DeiT(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=False,
        num_classes=10
    )
        
    def forward(self, x):
        return self.model(x)
    
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=192,
    ):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):

        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x

class MLP(nn.Module):
    def __init__(
        self,
        embed_dim,
        mlp_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(
            embed_dim,
            mlp_dim,
            dropout,
        )

    def forward(self, x):

        y = self.norm1(x)

        y, _ = self.attn(y, y, y)

        x = x + y

        x = x + self.mlp(self.norm2(x))

        return x


# --------------------------------------------------
# Vision Transformer
# --------------------------------------------------
class CiFormer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_dim=768,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size,
            patch_size,
            in_channels,
            embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(
            torch.empty(1, 1, embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.empty(1, num_patches + 1, embed_dim)
        )

        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                )
                for _ in range(depth)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self.apply(self._init_weights)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):

        B = x.size(0)

        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls, x), dim=1)

        x = x + self.pos_embed

        x = self.pos_drop(x)

        x = self.blocks(x)

        cls = x[:, 0]

        logits = self.head(cls)

        return logits
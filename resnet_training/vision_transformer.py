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
        pretrained=True,
        num_classes=10
    )
        
    def forward(self, x):
        return self.model(x)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x : (B,3,32,32)

        x = self.proj(x)
        # (B,192,8,8)

        x = x.flatten(2)
        # (B,192,64)

        x = x.transpose(1,2)
        # (B,64,192)

        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(embed_dim, mlp_dim, dropout)

    def forward(self, x):

        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x)
        )

        x = x + attn_out

        x = x + self.mlp(self.norm2(x))

        return x

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
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size,
            patch_size,
            in_channels,
            embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(
            torch.randn(1,1,embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1,num_patches+1,embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        B = x.size(0)

        x = self.patch_embed(x)

        cls = self.cls_token.expand(B,-1,-1)

        x = torch.cat((cls,x),dim=1)

        x = x + self.pos_embed

        x = self.dropout(x)

        x = self.blocks(x)

        x = self.norm(x)

        cls_token = x[:,0]

        logits = self.head(cls_token)

        return logits
    
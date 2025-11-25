"""
vit.py

This module implements the Vision Transformer architecture in PyTorch.
The model is split into Patchify and Attention- and MLP-components, which is combined into Transformer blocks.

Specific architectures with optional pretrain loading for ViT-B16 is included as a function.
"""
import torch
import torch.nn as nn
torch.manual_seed(0)
import os
path = os.path.dirname(__file__)

class Attention(nn.Module):
    """
    Multi-head self-attention Mechanism.

    # Parameters:
    - embed_dim (int): Total embedding dimension.
    - num_heads (int): Number of attention heads.
    - dropout (float): Dropout rate attention weights and output projection.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape  # Batch, Sequence (patches), Channels
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        out = self.proj(out)
        return self.proj_drop(out)

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, embed_dim * mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Patchify(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # -> (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # -> (B, N_patches, embed_dim)
        return x

class ViT(nn.Module):
    '''
    Vision transformer backbone.

    # Parameters:
    - image_size (int): Size of input image.
    - patch_size (int): Size of patches to split the image into.
    - embed_dim (int): Dimension of patch embeddings.
    - depth (int): Number of Transformer blocks.
    - num_heads (int): Number of attention heads.
    - mlp_ration (int): Expansion factor in MLP layers.
    - dropout (float): Dropout rate.
    '''
    def __init__(self, image_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, num_classes, dropout=0.0):
        super().__init__()
        self.name = ""
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2

        self.patchify = Patchify(in_channels=1, embed_dim=embed_dim, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patchify(x)  # (B, N, embed_dim)
        x = x + self.pos_embed # Add positional embedding to patch tokens
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling over patches
        return self.head(x)

def ViT_B16(image_size=128, num_classes=4, patch_size=16, pre_trained_path=None):
    """
    Base ViT with 12 layers, 12 heads, 768 embedding dim, patch size 16
    """
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=num_classes
    )
    model.name = f"ViT_B{patch_size}"
    if pre_trained_path:
        weights_path = os.path.join(path, pre_trained_path)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model

def ViT_B8(image_size=128, num_classes=4, patch_size=8, pre_trained_path=None):
    """
    Base ViT with 12 layers, 12 heads, 768 embedding dim, patch size 16
    """
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=num_classes
    )
    model.name = f"ViT_B{patch_size}"
    if pre_trained_path:
        weights_path = os.path.join(path, pre_trained_path)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model


def ViT_S16(image_size=128, num_classes=4, patch_size=16, pre_trained_path=None):
    """
    Small ViT with 12 layers, 6 heads, 384 embedding dim
    """
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_classes=num_classes
    )
    model.name = f"ViT_S{patch_size}"
    if pre_trained_path:
        weights_path = os.path.join(path, pre_trained_path)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model

def ViT_S8(image_size=128, num_classes=4, patch_size=8, pre_trained_path=False):
    """
    Small ViT with 12 layers, 6 heads, 384 embedding dim
    """
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_classes=num_classes
    )
    model.name = f"ViT_S{patch_size}"
    if pre_trained_path:
        weights_path = os.path.join(path, pre_trained_path)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model


def ViT_T16(image_size=128, num_classes=4, patch_size=16, pre_trained_path=None):
    """
    Tiny ViT with 12 layers, 3 heads, 192 embedding dim
    """
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        num_classes=num_classes
    )
    model.name = f"ViT_T{patch_size}"
    if pre_trained_path:
        weights_path = os.path.join(path, pre_trained_path)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model

def ViT_T8(image_size=128, num_classes=4, patch_size=8, pre_trained_path=None):
    """
    Tiny ViT with 12 layers, 3 heads, 192 embedding dim
    """
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        num_classes=num_classes
    )
    model.name = f"ViT_T{patch_size}"
    if pre_trained_path:
        weights_path = os.path.join(path, pre_trained_path)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model

def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    models = {"ViT_B16": ViT_B16, "ViT_S16": ViT_S16, "ViT_T16": ViT_T16}
    for name, model_func in models.items():
        model = model_func()
        print(f"{name} has {param_count(model):,} trainable parameters")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)
    # x = torch.randn((16,1,128,128)).to(device)
    # model = ViT_S16().to(device)
    # # print(model)
    # print(model(x).cpu().detach().numpy().shape)

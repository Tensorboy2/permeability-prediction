'''
convnext.py

This module implements the ConvNeXt architecture in PyTorch.

Includes:
- ConvNeXtBlock: Core residual block with depthwise convolution and MLP.
- ConvNeXtStage: A sequence of ConvNeXt blocks with downsampling.
- ConvNeXt: Full model with four stages and head.
- Factory functions: ConvNeXtTiny, ConvNeXtSmall.

Supports loading pretrained weights saved as .pth files.
'''
import torch
import torch.nn as nn
torch.manual_seed(0)
import os
path = os.path.dirname(__file__)

class ConvNeXtBlock(nn.Module):
    '''
     A single ConvNeXt block consisting of:
    - Depthwise convolution
    - LayerNorm in (H, W, C) format
    - MLP with GELU activation
    - Residual connection

    # Parameters:
    - dim (int): Number of channels in input.
    - expansion (int): Expansion factor for MLP.
    '''
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x) # (B, C, H, W)
        x = x.permute(0, 2, 3, 1) # (B, H, W, C) for LayerNorm
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
        return x + shortcut


class ConvNeXtStage(nn.Module):
    '''
    A stage in ConvNeXt made up of multiple ConvNeXt blocks.

    # Parameters:
    - in_dim (int): Number of channels in input. 
    - out_dim (int): Number of channels in output.
    - depth (int): Number of blocks in stage.
    '''
    def __init__(self, in_dim, out_dim, depth):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
        ) if in_dim != out_dim else nn.Identity()

        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_dim) for _ in range(depth)])

    def forward(self, x):
        if not isinstance(self.downsample, nn.Identity):
            x = x.permute(0, 2, 3, 1)
            x = self.downsample[0](x)
            x = x.permute(0, 3, 1, 2)
            x = self.downsample[1](x)
        return self.blocks(x)


class ConvNeXt(nn.Module):
    '''
    ConvNeXt model composed of:
    - Patchify stem (4x4 convolution + LayerNorm)
    - Four ConvNeXt stages
    - Classification head with global average pooling

    # Parameters:
    - dims (List[int]): Number of channels for each stage.
    - depth (List[int]): Number of blocks in each of the 4 stages.
    - input_channels (int): Number of input channels in image (default: 1 for binary).
    - num_classes (int): Number of output classes.
    '''
    def __init__(self,dims = [96, 192, 384, 768],depths = [3, 3, 9, 3], in_channels=1, num_classes=4):
        super().__init__()
        self.name = ""
        # Stem:
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6, elementwise_affine=True)
        )

        # Stages:
        self.stage1 = ConvNeXtStage(dims[0], dims[0], depths[0])
        self.stage2 = ConvNeXtStage(dims[0], dims[1], depths[1])
        self.stage3 = ConvNeXtStage(dims[1], dims[2], depths[2])
        self.stage4 = ConvNeXtStage(dims[2], dims[3], depths[3])

        # Output head:
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1], eps=1e-6),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.stem[0](x)
        x = x.permute(0, 2, 3, 1) # permutation to fit LayerNorm
        x = self.stem[1](x)
        x = x.permute(0, 3, 1, 2) # permutation back.
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def ConvNeXtTiny(pre_trained_path = None):
    '''
    Creates a ConvNeXtTiny model.

    # Parameter:
    - pre_trained (bool) wether or not to look for existing model weights.

    # Returns:
    - ResNet: A ConvNeXtTiny model instance.
    '''
    model = ConvNeXt(dims = [96, 192, 384, 768],depths = [3, 3, 9, 3])
    model.name = "ConvNeXtTiny"

    if pre_trained_path:
        weights_path = os.path.join(path, pre_trained_path)

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    
    return model

def ConvNeXtSmall(pre_trained_path = None):
    '''
    Creates a ConvNeXtSmall model.

    # Parameter:
    - pre_trained (bool) wether or not to look for existing model weights.

    # Returns:
    - ResNet: A ConvNeXtSmall model instance.
    '''
    model = ConvNeXt(dims = [96, 192, 384, 768],depths = [3, 3, 27, 3])
    model.name = "ConvNeXtSmall"

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
    models = {"ConvNeXtTiny": ConvNeXtTiny, "ConvNeXtSmall": ConvNeXtSmall}
    for name, model_func in models.items():
        model = model_func()
        print(f"{name} has {param_count(model):,} trainable parameters")
    # x = torch.randn((3,1,128,128)).half()
    # model = ConvNeXtSmall()
    # # print(model)
    # print(model(x).cpu().detach().numpy())
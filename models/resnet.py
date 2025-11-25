'''
resnet.py

This module implements a modified version of the ResNet architecture in PyTorch.
The model is split into ResNetBlock and ResNetDownsampleBlock. These are combined in a full structure in the ResNet module.

Specific architectures with optional pretrain loading for ResNet50 and ResNet101 is included as functions.
'''
import torch
import torch.nn as nn
torch.manual_seed(0)
import os
path = os.path.dirname(__file__)


class ResNetBlock(nn.Module):
    '''
    A standard residual block when input and output dimensions are the same.

    # Parameters:
    - channels (int): Number of input and output channels.
    - expansion (int): Compression factor for bottleneck layer (default: 4).
    '''
    def __init__(self, channels, expansion=4):
        super().__init__()
        mid_channels = channels // expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class ResNetDownsampleBlock(nn.Module):
    '''
    A residual block that downsamples the input using stride and adjusts channel depth.

    # Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels after expansion.
    - stride (int): Stride used for downsampling (default: 2).
    - expansion (int): Bottleneck expansion factor (default: 4).
    '''
    def __init__(self, in_channels, out_channels, stride=2, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.downsample(x))

class ResNet(nn.Module):
    '''
    A custom ResNet backbone supporting different depths and widths.

    # Parameters:
    - depth (List[int]): Number of blocks in each of the 4 stages.
    - width (List[int]): Number of channels for each stage.
    - num_classes (int): Number of output classes.
    - input_channels (int): Number of input channels in image (default: 1 for binary).
    '''
    def __init__(self, depth=[3,4,6,3], width=[64,256,512,1024,2048], num_classes=1, input_channels=1):
        super().__init__()
        self.name = ""
        # Stem: initial convolution and pooling to reduce spatial resolution
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width[0], kernel_size=9, stride=2, padding=4, padding_mode="circular", bias=False),
            nn.BatchNorm2d(width[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Define stages:
        self.stage1 = self._make_stage(width[0], width[1], num_blocks=depth[0], first_stride=1)
        self.stage2 = self._make_stage(width[1], width[2], num_blocks=depth[1])
        self.stage3 = self._make_stage(width[2], width[3], num_blocks=depth[2])
        self.stage4 = self._make_stage(width[3], width[4], num_blocks=depth[3])

        # Define pooling and fully-connected layer:
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(width[4], num_classes))

    def _make_stage(self, in_channels, out_channels, num_blocks, first_stride=2):
        '''
        Creates a ResNEt stage made with downsampling block followed by (num_blocks-1) standard blocks.
        '''
        layers = [ResNetDownsampleBlock(in_channels, out_channels, stride=first_stride)]
        for _ in range(num_blocks - 1):
            layers.append(ResNetBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def ResNet50(image_size=128, num_classes=4, pre_trained_path=None):
    '''
    Creates a ResNet50 model.

    # Parameter:
    - image_size (int): Image input size.
    - num_classes (int): Number of output classes.
    - pre_trained (bool) wether or not to look for existing model weights.

    # Returns:
    - ResNet: A ResNet50 model instance.
    '''
    model = ResNet(depth=[3,4,6,3], width=[64,256,512,1024,2048],num_classes=num_classes)
    model.name = "ResNet50"
    if pre_trained_path:
        weights_path = os.path.join(path, pre_trained_path)

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model

def ResNet101(image_size=128, num_classes=4, pre_trained_path = None):
    '''
    Creates a ResNet101 model.

    # Parameter:
    - image_size (int): Image input size.
    - num_classes (int): Number of output classes.
    - pre_trained (bool) wether or not to look for existing model weights.

    # Returns:
    - ResNet: A ResNet101 model instance
    '''
    model = ResNet(depth=[3,4,23,3], width=[64,256,512,1024,2048], num_classes=num_classes)
    model.name = "ResNet101"

    if pre_trained_path:
        weights_path = os.path.join(path, pre_trained_path)

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    return model

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    x = torch.randn((3,1,128,128)).to(device)
    model = ResNet101().to(device)
    # print(model)
    print(model(x).cpu().detach().numpy())


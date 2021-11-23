import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def __repr__(self):
        return f"LayerNorm({self.num_features}, eps={self.eps}, affine={self.affine})"

    def forward(self, x):
        x = F.layer_norm(x, x.shape[1:], eps=self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class Pad3d(nn.Module):
    def __init__(self, padding, pad_type='constant', value=0):
        super(Pad3d, self).__init__()
        self.padding = padding
        self.type = pad_type
        self.value = value

    def __repr__(self):
        return f"Pad3d({self.padding}, padding_type={self.type}, padding_value={self.value})"

    def forward(self, x):
        return F.pad(x, (self.padding,) * 6, self.type, self.value)


pad_dict = dict(
       none = lambda x: lambda x: x,
     zero2d = nn.ZeroPad2d,
     zero3d = lambda x: Pad3d(x, 'constant'),
  reflect2d = nn.ReflectionPad2d,
  reflect3d = lambda x: Pad3d(x, 'reflect'),
replicate2d = nn.ReplicationPad2d,
replicate3d = lambda x: Pad3d(x, 'replicate'),)

conv_dict = dict(
   conv2d = nn.Conv2d,
   conv3d = nn.Conv3d,
 deconv2d = nn.ConvTranspose2d,
 deconv3d = nn.ConvTranspose3d)

norm_dict = dict(
      none = lambda x: lambda x: x,
  spectral = lambda x: lambda x: x,
   batch2d = nn.BatchNorm2d,
   batch3d = nn.BatchNorm3d,
instance2d = nn.InstanceNorm2d,
instance3d = nn.InstanceNorm3d,
     layer = LayerNorm)

activ_dict = dict(
      none = lambda: lambda x: x,
      relu = lambda: nn.ReLU(inplace=True),
     lrelu = lambda: nn.LeakyReLU(0.2, inplace=True),
     prelu = lambda: nn.PReLU(),
      selu = lambda: nn.SELU(inplace=True),
      tanh = lambda: nn.Tanh())


class ConvolutionBlock(nn.Module):
    def __init__(self, conv='conv2d', norm='instance2d', activ='relu', pad='reflect2d', padding=0, **conv_opts):
        super(ConvolutionBlock, self).__init__()

        non_layers = [ly for ly in (conv, norm, pad) if ly != 'none' and ly != 'layer']
        for ly in non_layers[1:]:
            assert ly[-2:] == non_layers[0][-2:], f"Inconsistent layer dimension: {ly} and {non_layers[0]}"

        self.pad = pad_dict[pad](padding)
        self.conv = conv_dict[conv](**conv_opts)

        out_channels = conv_opts['out_channels']
        self.norm = norm_dict[norm](out_channels)
        if norm == "spectral": self.conv = spectral_norm(self.conv)

        self.activ = activ_dict[activ]()

    def forward(self, x):
        return self.activ(self.norm(self.conv(self.pad(x))))


class ResidualBlock(nn.Module):
    def __init__(self, channels, conv="conv2d", norm='instance2d', activ='relu', pad='reflect2d'):
        super(ResidualBlock, self).__init__()

        block = []
        block += [ConvolutionBlock(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, conv=conv, norm=norm, activ=activ, pad=pad)]
        block += [ConvolutionBlock(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, conv=conv, norm=norm, activ='none', pad=pad)]
        self.model = nn.Sequential(*block)

    def forward(self, x): return self.model(x) + x


class FullyConnectedBlock(nn.Module):
    def __init__(self, input_ch, output_ch, norm='none', activ='relu'):
        super(FullyConnectedBlock, self).__init__()

        self.fc = nn.Linear(input_ch, output_ch, bias=True)
        self.norm = norm_dict[norm](output_ch)
        if norm == "spectral": self.fc = spectral_norm(self.fc)
        self.activ = activ_dict[activ]()

    def forward(self, x): return self.activ(self.norm(self.fc(x)))

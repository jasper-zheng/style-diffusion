from torch import nn
import torch
from math import pi

from models.layers import Interpolate, ResConvBlock, ConditionedSequential

class UNet(nn.Module):
  def __init__(self, in_features, unet_config):
    super().__init__()
    down_config = unet_config['down_blocks']
    up_config = unet_config['up_blocks']

    channels = down_config['channels']
    scale_factors = down_config['scale_factors']
    depths = down_config['depths']

    ###### down blocks
    current_size = unet_config['in_res']
    self.down_layer_names = []
    for idx, (c_out, scale) in enumerate(zip(channels,scale_factors)):
      modules = []
      if not scale==1:
        modules.append(Interpolate(scale))
        current_size = int(current_size*scale)
      for i in range(depths[idx]):
        # c_in = channels[idx-1] if idx == 0 and i == 0 else c_out
        if idx == 0 and i == 0:
          c_in = c_out
        elif i == 0:
          c_in = channels[idx-1]
        else:
          c_in = c_out

        # c_in = channels[idx-1] if i==0 and not idx==0 else c_out
        modules.append(ResConvBlock(in_features, c_in, c_out))

      layer_seq = ConditionedSequential(*modules)
      name = f'DL{idx:02}_C{c_out}_R{current_size}'
      setattr(self, name, layer_seq)
      self.down_layer_names.append(name)
    
      print(f'{name} with {depths[idx]} res_conv blocks added, -> (B, {c_out}, {current_size}, {current_size})')
    
    ###### bottleneck
    self.bottleneck_res = current_size
    print(f'bottleneck: (B, {c_out}, {current_size}, {current_size})')

    channels = up_config['channels']
    scale_factors = up_config['scale_factors']
    depths = up_config['depths']

    ###### up blocks
    self.up_layer_names = []
    for idx, (c_out, scale) in enumerate(zip(channels,scale_factors)):
      modules = []
      if not scale==1:
        modules.append(Interpolate(scale))
        current_size = int(current_size*scale)
      for i in range(depths[idx]):
        # c_in = channels[idx-1]*2 if idx == 0 and i == 0 else c_out
        if idx == 0 and i == 0:
          c_in = c_out*2
        elif i == 0:
          c_in = channels[idx-1]*2
        else:
          c_in = c_out
        modules.append(ResConvBlock(in_features, c_in, c_out))

      layer_seq = ConditionedSequential(*modules)
      name = f'UL{idx:02}_C{c_out}_R{current_size}'
      setattr(self, name, layer_seq)
      self.up_layer_names.append(name)
    
      print(f'{name} with {depths[idx]} res_conv blocks added, -> (B, {c_out}, {current_size}, {current_size})')

  def forward(self, x, cond):
    skips = []
    for idx, name in enumerate(self.down_layer_names):
      x = getattr(self, name)(x, cond)
      skips.append(x)
    for idx, (name, skip) in enumerate(zip(self.up_layer_names, reversed(skips))):
      x = torch.cat([x, skip], dim=1)
      x = getattr(self, name)(x, cond)

    return x


class MappingNet(nn.Sequential):
  def __init__(self, feats_in, feats_out, n_layers=2):
    layers = []
    for i in range(n_layers):
      layers.append(nn.Linear(feats_in if i == 0 else feats_out, feats_out))
      layers.append(nn.GELU())
    super().__init__(*layers)
    for layer in self:
      if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight)

class FourierTimesteps(nn.Module):
  def __init__(self, in_features, out_features, std=1.):
    super().__init__()
    assert out_features % 2 == 0
    self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

  def forward(self, t):
    # sigmas = torch.sin(t * pi / 2) 
    f = 2 * pi * t @ self.weight.T
    return torch.cat([f.cos(), f.sin()], dim=-1)
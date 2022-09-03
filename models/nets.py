from torch import nn
import torch
from math import pi, log2

from models.layers import Interpolate, ResConvBlock,ResModConvBlock, ConditionedSequential, ConditionedModResidualBlock,ConditionedModSequential
from models.fourier_modulation import FourierInput

class UNet(nn.Module):
  def __init__(self, in_features, unet_config):
    super().__init__()
    down_config = unet_config['down_blocks']
    up_config = unet_config['up_blocks']

    channels = down_config['channels']
    scale_factors = down_config['scale_factors']
    depths = down_config['depths']
    ws_encoder_channels = unet_config['ws_encoder_channels']
    self.fourier_output = unet_config['fourier_output']
    self.fourier_sampling_rate = unet_config['fourier_sampling_rate']
    self.bandwidth = unet_config['bandwidth']
    self.down_skips = down_config['skips']
    

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
    self.skip_stages = int(log2(self.fourier_output/self.bottleneck_res))

    ###### bottleneck flatten

    modules = []
    for i in range(int(log2(current_size/4))):
      c_in = channels[-1] if i==0 else ws_encoder_channels[0]
      c_out = ws_encoder_channels[0]
      modules.append(torch.nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=True))
      modules.append(torch.nn.LeakyReLU(0.2, inplace=True))
      current_size = int(current_size*0.5)
      print(f'downsample to bottleneck: {current_size*2} -> {current_size}')

    print(f'bottleneck: (B, {c_out}, {current_size}, {current_size})')

    modules.append(torch.nn.Flatten(start_dim=1))
    computed_flatten_size = current_size * current_size * c_out
    c_in = computed_flatten_size
    c_out = ws_encoder_channels[0]
    modules.append(torch.nn.Linear(c_in, c_out, bias = True))
    modules.append(torch.nn.LeakyReLU(0.2, inplace=True))
    name = 'flatten_layer'
    layer_seq = torch.nn.Sequential(*modules)
    setattr(self, name, layer_seq)
    print(f'{name} added: {c_in} -> {c_out}')

    ###### modulate_w

    modules = []
    for idx, channel in enumerate(ws_encoder_channels):
      c_in = channel if idx==0 else ws_encoder_channels[idx-1]
      c_out = channel
      modules.append(torch.nn.Linear(c_in, c_out, bias = True))
      modules.append(torch.nn.LeakyReLU(0.2, inplace=True))
    name = 'modulate_w_encoder'
    layer_seq = torch.nn.Sequential(*modules)
    setattr(self, name, layer_seq)
    print(f'{name} added: -> (B, {c_out})')
    self.w_dim = c_out

    channels = up_config['channels']
    channels_out = up_config['channels_out']
    
    ##### fourier_input 
    self.fourier_input = FourierInput(c_out,
                                      channels[0],
                                      [self.fourier_output, self.fourier_output],
                                      sampling_rate=self.fourier_sampling_rate, 
                                      bandwidth=self.bandwidth)
    current_size = self.fourier_output
    print(f'fourier_input added: (B, {c_out}) -> (B, {channels[0]}, {self.fourier_output}, {self.fourier_output})')
    
    scale_factors = up_config['scale_factors']
    depths = up_config['depths']
    self.up_skips = up_config['skips']
    self.num_of_ws = len(channels)+1
    ###### up blocks
    self.up_layer_names = []
    for idx, (channel,channel_out, scale) in enumerate(zip(channels,channels_out,scale_factors)):
      modules = []
      if not scale==1:
        modules.append(Interpolate(scale))
        current_size = int(current_size*scale)
      
      for i in range(depths[idx]):
        if idx == 0 and i == 0:
          c_in = channel + self.up_skips[idx]
        elif i == 0:
          c_in = channels_out[idx-1] + self.up_skips[idx]
        else:
          c_in = channel
        if not i == depths[idx]-1:
          c_out = channel
        else:
          c_out = channel_out
        modules.append(ResModConvBlock(in_features, c_in, c_out, w_dim=self.w_dim))

      

      layer_seq = ConditionedModSequential(*modules)
      name = f'UL{idx:02}_C{c_out}_R{current_size}'
      setattr(self, name, layer_seq)
      self.up_layer_names.append(name)
    
      print(f'{name} with {depths[idx]} res_conv blocks added, -> (B, {c_out}, {current_size}, {current_size})')

  def forward(self, x, cond):
    skips = []
    for idx, (name, s) in enumerate(zip(self.down_layer_names, self.down_skips)):
      x = getattr(self, name)(x, cond)
      if s:
        skips.append(x)
      else:
        skips.append(None)

    mod_ws = self.flatten_layer(x)
    mod_ws = self.modulate_w_encoder(mod_ws)
    mod_ws = mod_ws.unsqueeze(1).repeat([1, self.num_of_ws, 1]).unbind(dim=1)
    x = self.fourier_input(mod_ws[0])

    for idx, (name, skip, s, w) in enumerate(zip(self.up_layer_names, reversed(skips), self.up_skips, mod_ws[1:])):
      if s:
        if skip == None:
          assert False
        x = torch.cat([x, skip], dim=1)
      # print(x.shape)
      x = getattr(self, name)(x, cond, w)
      
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
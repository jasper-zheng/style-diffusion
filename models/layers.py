from torch import nn
from torch.nn.functional import group_norm
import torch

from models.fourier_modulation import ModulatedModule, Modulated_Conv

from math import pi

class ResidualBlock(nn.Module):
  def __init__(self, main, skip=None):
    super().__init__()
    self.main = nn.Sequential(*main)
    self.skip = skip if skip else nn.Identity()

  def forward(self, input):
    return self.main(input) + self.skip(input)

class ConditionedModule(nn.Module):
  pass



class ConditionedSequential(nn.Sequential, ConditionedModule):
  def forward(self, input, cond):
    for module in self:
      # print(module)
      if isinstance(module, ConditionedModule):
        input = module(input, cond)
      else:
        input = module(input)
    return input

class ConditionedModSequential(nn.Sequential, ConditionedModule, ModulatedModule):
  def forward(self, input, cond, w):
    for module in self:
      # print(module)
      if isinstance(module, ConditionedModule) and isinstance(module, ModulatedModule):
        input = module(input, cond, w)
      elif isinstance(module, ConditionedModule):
        input = module(input, cond)
      elif isinstance(module, ModulatedModule):
        input = module(input, w)
      else:
        input = module(input)
    return input

class ConditionedResidualBlock(ConditionedModule):
  def __init__(self, main, skip=None):
    super().__init__()
    self.main = ConditionedSequential(*main)
    self.skip = skip if skip else nn.Identity()

  def forward(self, input, cond):
    skip = self.skip(input, cond) if isinstance(self.skip, ConditionedModule) else self.skip(input)
    return self.main(input, cond) + skip

class ConditionedModResidualBlock(ConditionedModule, ModulatedModule):
  def __init__(self, main, skip=None):
    super().__init__()
    self.main = ConditionedModSequential(*main)
    self.skip = skip if skip else nn.Identity()
  def forward(self, input, cond, w):
    skip = self.skip(input, cond) if isinstance(self.skip, ConditionedModule) else self.skip(input)
    return self.main(input, cond, w) + skip

class ResConvBlock(ConditionedResidualBlock):
  def __init__(self, in_features, c_in, c_out, group_norm_size = 32, is_last=False):
    skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
    super().__init__([
        
      AdaGN(in_features, c_in, max(1, c_in // group_norm_size)),
      nn.GELU(),
      nn.Conv2d(c_in, c_out, 3, stride = 1, padding = 1),

      AdaGN(in_features, c_out, max(1, c_out // group_norm_size)),
      nn.Conv2d(c_out, c_out, 3, stride = 1, padding = 1),
      nn.Dropout2d(0.1, inplace=True),
      nn.GELU()

    ], skip)

class ResModConvBlock(ConditionedModResidualBlock):
  def __init__(self, in_features, c_in, c_out, w_dim=512, group_norm_size = 32, is_last=False):
    skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
    super().__init__([
        
      AdaGN(in_features, c_in, max(1, c_in // group_norm_size)),
      nn.GELU(),
      Modulated_Conv(c_in, c_out, w_dim = w_dim, kernel_size = 3),

      AdaGN(in_features, c_out, max(1, c_out // group_norm_size)),
      Modulated_Conv(c_out, c_out, w_dim = w_dim, kernel_size = 3),
      nn.Dropout2d(0.1, inplace=True),
      nn.GELU()

    ], skip)

class AdaGN(ConditionedModule):
  # TODO: change to modulated_conv2?
  def __init__(self, feats_in, c_out, num_groups, eps=1e-5):
    super().__init__()
    self.num_groups = num_groups
    self.eps = eps
    self.mapper = nn.Linear(feats_in, c_out * 2)

  def forward(self, input, cond):
    weight, bias = self.mapper(cond).chunk(2, dim=-1) # (B, feats_in)
    input = group_norm(input, self.num_groups, eps=self.eps) # (B, C, H, W)
    return torch.addcmul(bias.unsqueeze(-1).unsqueeze(-1), input, weight.unsqueeze(-1).unsqueeze(-1) + 1)

class SelfAttention2d(nn.Module):
  def __init__(self, c_in, n_head=1, dropout_rate=0.1):
    super().__init__()
    assert c_in % n_head == 0
    self.norm = nn.GroupNorm(1, c_in)
    self.n_head = n_head
    self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
    self.out_proj = nn.Conv2d(c_in, c_in, 1)
    self.dropout = nn.Dropout2d(dropout_rate, inplace=True)

  def forward(self, input):
    n, c, h, w = input.shape
    qkv = self.qkv_proj(self.norm(input))
    qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
    q, k, v = qkv.chunk(3, dim=1)
    scale = k.shape[3]**-0.25
    att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
    y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
    return input + self.dropout(self.out_proj(y))

class SkipBlock(nn.Module):
  def __init__(self, main, skip=None):
    super().__init__()
    self.main = nn.Sequential(*main)
    self.skip = skip if skip else nn.Identity()

  def forward(self, input):
    return torch.cat([self.main(input), self.skip(input)], dim=1)

def expand_to_planes(input, shape):
  return input[..., None, None].repeat([1, 1, shape[2], shape[3]])

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

class Interpolate(nn.Module):
  def __init__(self, scale_factor):
    super().__init__()
    self.scale_factor = scale_factor
  def forward(self, x):
    return nn.functional.interpolate(x, scale_factor = self.scale_factor, mode='nearest')
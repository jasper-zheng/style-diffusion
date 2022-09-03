from torch import nn
import torch

from models.nets import FourierTimesteps, MappingNet, UNet

from math import pi

class Diffusion(nn.Module):
  def __init__(self, config):
    super().__init__()
    
    self.loss_type = config['loss_type']
    self.in_features = config['in_features']
    
    unet_config = config['u_net']
    self.in_res = unet_config['in_res']
    self.u_net_in_channel = unet_config['down_blocks']['channels'][0]
    self.u_net_out_channel = unet_config['up_blocks']['channels'][-1]

    mapping_cond_channels = config['mapping_cond_channels']
    unet_cond_channels = config['unet_cond_channels']

    self.timestep_embed = FourierTimesteps(1, self.in_features)
    if mapping_cond_channels > 0:
      # self.mapping_condition = nn.Linear(mapping_cond_channels, self.in_features, bias=False)
      self.mapping_condition = nn.Embedding(mapping_cond_channels, self.in_features)
    # time_embed + mapping_cond -> mapping_out
    self.mapping = MappingNet(self.in_features, self.in_features)

    # proj_in
    modules = []
    modules.append(nn.Conv2d((3 + unet_cond_channels), self.u_net_in_channel, 1))
    self.rgb_in = nn.Sequential(*modules)

    # proj_out 
    self.rgb_out = nn.Conv2d(self.u_net_out_channel, 3, 1)

    # u_net 
    self.u_net = UNet(self.in_features, unet_config)
    


  def forward(self, x, t, mapping_cond = None, unet_cond = None):
    '''
      x: (B, C, H, W)
      t: (B,)
      mapping_cond: (B,)
    '''
    time_embeddings = self.timestep_embed(t.unsqueeze(-1)) # (B, in_features)
    if mapping_cond:
      mapping_cond = self.mapping_condition(mapping_cond) 
      mapping_out = self.mapping(time_embeddings + mapping_cond)
    else:
      mapping_out = self.mapping(time_embeddings)
    
    # proj_in
    if unet_cond:
      x = torch.cat([x, unet_cond], dim=1)
    x = self.rgb_in(x)
    
    # u_net 
    x = self.u_net(x, mapping_out)
    
    # proj_out 
    x = self.rgb_out(x)

    return x

  def noise_schedule(self, t):
    """
    (timestep: (n,)) -> signal_rate: (n,), noise_rate: (n,)
    """
    return torch.cos(t * pi / 2), torch.sin(t * pi / 2) 

  @torch.no_grad()
  def get_components(self, noised_imgs, preds, t):
    alphas, sigmas = self.noise_schedule(t)

    if self.loss_type == 'signal':
      pred_images = preds
      pred_noises = (noised_imgs - alphas * pred_images) / sigmas
      pred_velocities = (alphas * noised_imgs - pred_images) / sigmas

    if self.loss_type == 'noise':
      pred_noises = preds
      pred_images = (noised_imgs - sigmas * pred_noises) / alphas
      pred_velocities = (pred_noises - sigmas * noised_imgs) / alphas

    if self.loss_type == 'velocity':
      pred_velocities = preds
      pred_images = alphas * noised_imgs - sigmas * pred_velocities
      pred_noise = sigmas * noised_imgs + alphas * pred_velocities

    return pred_images, pred_noise, pred_velocities

  

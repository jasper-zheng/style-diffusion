from torch.optim import Adam, AdamW
from torch.nn.functional import mse_loss
import torch

import numpy as np
from math import pi

from tqdm.notebook import trange

def make_optimizer(config,params):
  if config['type'] == 'adam':
    opt = Adam(
        params, 
        lr=config['lr'], 
        betas=tuple(config['betas'])
        )
  elif config['type'] == 'adamw':
    opt = AdamW(
        params,
        lr=config['lr'],
        betas=tuple(config['betas']),
        eps=config['eps'],
        weight_decay=config['weight_decay']
        )
  return opt

def combined_mse_loss(recons, input):
  recons_loss = mse_loss(recons, input)
  # loss = torch.clamp(recons_loss, min=-30, max=30)
  return recons_loss



@torch.no_grad()
def get_targets(real_imgs, loss_type, t):

  device = torch.device('cuda')

  alphas, sigmas = noise_schedule(t)
  alphas = alphas[:, None, None, None] # signal rates
  sigmas = sigmas[:, None, None, None] # noise rates

  noises = torch.randn_like(real_imgs).to(device)

  noised_imgs = real_imgs * alphas + noises * sigmas

  if loss_type == 'signal':
    return noised_imgs, real_imgs
  
  elif loss_type == 'noise':
    return noised_imgs, noises

  elif loss_type == 'velocity':
    velocities = noises * alphas - real_imgs * sigmas
    return noised_imgs, velocities

  # elif loss_type == 'dissipation'


  else:
    raise NotImplementedError

# @torch.no_grad()
# def get_dissipation_targets(real_imgs, t):
#   K = real_imgs.shape[-1]
#   freqs = pi * torch.linspace(0, K-1, K) / K
#   frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2
#   u_proj = dct(real_imgs)
#   u_proj = dct(u_proj)
#   u_proj = torch.exp(-frequencies_squared*t) * u_proj
#   u_reconstucted = idct(u_proj,norm ='ortho')
#   u_reconstucted = idct(u_reconstucted,norm ='ortho')
#   return u_reconstucted

@torch.no_grad()
def noise_schedule(t):
  """
  (timestep: (n,)) -> signal_rate: (n,), noise_rate: (n,)
  """
  return torch.cos(t * pi / 2), torch.sin(t * pi / 2) 

@torch.no_grad()
def sample(model, steps, x = None, mapping_cond=None, unet_cond=None, batch_num = 1, resize = 0):
  device = torch.device('cuda')
  x = torch.randn([batch_num, 3, model.in_res, model.in_res], device=device)
  t_step = x.new_ones([x.shape[0]])

  timeline = torch.linspace(1, 0, steps + 1)[:-1]
  alphas, sigmas = noise_schedule(timeline)

  for i in trange(steps):
    v = model(x, t_step * timeline[i])

    pred = x * alphas[i] - v * sigmas[i]
    
    if i < steps - 1:

      eps = x * sigmas[i] + v * alphas[i]
      next_sigma = sigmas[i + 1]
      next_alpha = alphas[i + 1]

      x = pred * next_alpha + eps * next_sigma

  if resize and not resize == model.in_res:
    pred = torch.nn.functional.interpolate(pred, size = (resize,resize))

  return pred

@torch.no_grad()
def sample_components(model, steps, x = None, x_t = None, mapping_cond=None, unet_cond=None, eta=1, batch_num = 1, resize = 0):
  device = torch.device('cuda')
  if x==None or x_t==None:
    x = torch.randn([batch_num, 3, model.in_res, model.in_res], device=device)
    x_t = 0
  t_step = x.new_ones([x.shape[0]])

  timeline = torch.linspace(x_t, 0, steps + 1)[:-1]
  alphas, sigmas = noise_schedule(timeline)

  for i in trange(steps):
    v = model(x, t_step * timeline[i])

    pred = x * alphas[i] - v * sigmas[i]
    eps = x * sigmas[i] + v * alphas[i]
    
    if i < steps - 1:
      ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
      next_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()
      # next_sigma = sigmas[i + 1]
      next_alpha = alphas[i + 1]

      x = pred * next_alpha + eps * next_sigma

  if resize and not resize == model.in_res:
    eps = torch.nn.functional.interpolate(eps, size = (resize,resize))
    pred = torch.nn.functional.interpolate(pred, size = (resize,resize))
    v = torch.nn.functional.interpolate(v, size = (resize,resize))

  return pred, eps, v
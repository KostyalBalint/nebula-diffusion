import os

import torch
from models.vae_gaussian import *
from models.vae_flow import *

ckpt = './pretrained/GEN_airplane.pt'
device = 'mps'
sample_num_points = 10000
batch_size = 1

print('Loading model')
ckpt = torch.load(ckpt, map_location=device)

if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(device)
elif ckpt['args'].model == 'flow':
    model = FlowVAE(ckpt['args']).to(device)

print('Loading state dict')
model.load_state_dict(ckpt['state_dict'])

print('Generating 3D model')
with torch.no_grad():
    z = torch.randn([batch_size, ckpt['args'].latent_dim]).to(device)
    x = model.sample(z, sample_num_points, flexibility=ckpt['args'].flexibility)

    res = x.detach().cpu()[0]

print(np.shape(res))
print(res.tolist())
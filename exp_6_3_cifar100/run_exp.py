# MIT License
#
# Copyright (c) 2018, University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

CUDA = True

import os
import time
import pickle
import math

import os
import time
import pickle
import numpy as np
cuda_id = '0'
if CUDA:
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(cuda_id)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from exp_6_3_cifar100.network import DenseNet3

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
matplotlib.rc('font', family='Latin Modern Roman')

import utils

def cm2inch(value):
  return value/2.54

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

# The bounds in NN-space
cifar_mean = [125.3/255.0, 123.0/255.0, 113.9/255.0]
cifar_std = [63.0/255.0, 62.1/255.0, 66.7/255.0]

if CUDA:
  x_min = torch.tensor([(0.0 - cifar_mean[0])/cifar_std[0], (0.0 - cifar_mean[1])/cifar_std[1], (0.0 - cifar_mean[2])/cifar_std[2]]).cuda()
  x_max = torch.tensor([(1.0 - cifar_mean[0])/cifar_std[0], (1.0 - cifar_mean[1])/cifar_std[1], (1.0 - cifar_mean[2])/cifar_std[2]]).cuda()
else:
  x_min = torch.tensor([(0.0 - cifar_mean[0])/cifar_std[0], (0.0 - cifar_mean[1])/cifar_std[1], (0.0 - cifar_mean[2])/cifar_std[2]])
  x_max = torch.tensor([(1.0 - cifar_mean[0])/cifar_std[0], (1.0 - cifar_mean[1])/cifar_std[1], (1.0 - cifar_mean[2])/cifar_std[2]])

# Fix random seed for reproducibility
seed = int(cuda_id)
np.random.seed(seed)
torch.manual_seed(seed)

test_loader = torch.utils.data.DataLoader(
      datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean=cifar_mean, std=cifar_std)
                      ])),
      batch_size=1000, shuffle=False)

# Get data into arrays for convenience
if CUDA:
  x = torch.zeros(10000, 3, 32, 32).cuda()
  y = torch.zeros(10000, dtype=torch.long).cuda()
else:
  x = torch.zeros(10000, 3, 32, 32)
  y = torch.zeros(10000, dtype=torch.long)

for idx, (data, target) in enumerate(test_loader):
  if CUDA:
    data, target = data.float().cuda(), target.long().cuda()
  else:
    data, target = data.float(), target.long()
  #data = data.view(-1, 3*32*32)
  x[(idx*1000):((idx+1)*1000),:,:,:] = data
  y[(idx*1000):((idx+1)*1000)] = target

def multilevel_uniform(
      prop,
      x_sample,
      sigma=1.,
      rho=0.1,
      count_particles=1000,
      count_mh_steps=100,
      debug=True, stats=False):

  # Calculate the mean of the normal distribution in logit space
  # We transform the input from [x_min, x_max] to [epsilon, 1 - epsilon], then to [logit(epsilon), logit(1 - epsilon)]
  # Then we can do the sampling on (-inf, inf)
  prior = dist.Uniform(low=torch.max(x_sample-sigma*(x_max-x_min).view(3,1,1), x_min.view(3,1,1)), high=torch.min(x_sample+sigma*(x_max-x_min).view(3,1,1), x_max.view(3,1,1)))

  #print((x_sample-sigma).size())
  #raise Exception()

  # Parameters
  if CUDA:
    width_proposal = sigma*torch.ones(count_particles).cuda()/30
  else:
    width_proposal = sigma*torch.ones(count_particles)/30
  count_max_levels = 500
  target_acc_ratio = 0.9
  max_width_proposal = 0.1
  min_width_proposal = 1e-8
  width_inc = 1.02
  width_dec = 0.5

  # Sample the initial particles
  # Implements parallel batched accept-reject sampling.
  x = prior.sample(torch.Size([count_particles]))

  #print(x.size())
  #raise Exception()

  L_prev = -math.inf
  L = -math.inf
  l_inf_min = math.inf
  lg_p = 0
  levels = []

  #print('Inside valid bounds', x_min, x_max)
  #utils.stats(x[0])
  #print((x >= x_min).all(dim=1) & (x <= x_max).all(dim=1))
  #raise Exception()

  # Loop over levels
  for level_idx in range(count_max_levels):
    if CUDA:
      acc_ratio = torch.zeros(count_particles).cuda()
    else:
      acc_ratio = torch.zeros(count_particles)

    if L >= 0:
      break

    # Calculate current level
    s_x = prop(x).squeeze(-1)
    s_sorted, s_idx = torch.sort(s_x)
    L = min(s_sorted[math.floor((1-rho)*count_particles)].item(), 0)
    if L == L_prev:
      L = 0
    levels.append(L)
    where_keep = s_x >= L
    where_kill = s_x < L
    count_kill = (where_kill).sum()
    count_keep = count_particles - count_kill

    # Print level
    if debug:
      print(f'Level {level_idx+1} = {L}')

    # Terminate if change in level is below some threshold
    if count_keep == 0:
      return -math.inf, None, x, levels

    lg_p += torch.log(count_keep.float()).item() - math.log(count_particles)
    
    # Early termination
    if lg_p < -90:
      return -90., None, x, levels

    # If the level is 0 then don't do last MH steps (speeds things up!)
    if L >= 0:
      break

    # Uniformly resample killed particles below the level
    new_idx = torch.randint(low=0, high=count_keep, size=(count_kill,), dtype=torch.long)
    x = x[where_keep]
    x = torch.cat((x, x[new_idx]), dim=0)
    width_proposal = width_proposal[where_keep]
    width_proposal = torch.cat((width_proposal, width_proposal[new_idx]), dim=0)
    
    #acc_ratio = torch.zeros(count_kill).cuda()
    #x_temp = x
    #while acc_ratio.mean() < 0.2:
    #  x = x_temp
    if CUDA:
      acc_ratio = torch.zeros(count_particles).cuda()
    else:
      acc_ratio = torch.zeros(count_particles)

    for mh_idx in range(count_mh_steps):
      # Propose new sample
      g_bottom = dist.Uniform(low=torch.max(x - width_proposal.view(-1,1,1,1)*(x_max-x_min).view(3,1,1), prior.low), high=torch.min(x + width_proposal.view(-1,1,1,1)*(x_max-x_min).view(3,1,1), prior.high))

      x_maybe = g_bottom.sample()
      s_x = prop(x_maybe).squeeze(-1)

      # Calculate log-acceptance ratio
      g_top = dist.Uniform(low=torch.max(x_maybe - width_proposal.view(-1,1,1,1)*(x_max-x_min).view(3,1,1), prior.low), high=torch.min(x_maybe + width_proposal.view(-1,1,1,1)*(x_max-x_min).view(3,1,1), prior.high))
      lg_alpha = (prior.log_prob(x_maybe) + g_top.log_prob(x) - prior.log_prob(x) - g_bottom.log_prob(x_maybe)).view(count_particles,-1).sum(dim=1)
      acceptance = torch.min(lg_alpha, torch.zeros_like(lg_alpha))

      # Work out which ones to accept
      log_u = torch.log(torch.rand_like(acceptance))
      acc_idx = (log_u <= acceptance) & (s_x >= L)
      acc_ratio += acc_idx.float()
      x = torch.where(acc_idx.view(-1,1,1,1), x_maybe, x)
        
    # Adapt the width proposal *for each chain individually*
    acc_ratio /= count_mh_steps

    # DEBUG: See what acceptance ratios are doing
    if stats:
      utils.stats(acc_ratio)
    #input()

    #print(acc_ratio.size())
    width_proposal = torch.where(acc_ratio > 0.124, width_proposal*width_inc, width_proposal)
    width_proposal = torch.where(acc_ratio < 0.124, width_proposal*width_dec, width_proposal)

    L_prev = L
    #input()

  return lg_p, None, x, levels

def do_run(count_runs, sample_id, sigma, rho, count_particles, count_mh_steps, debug=False, stats=True):
  lg_ps = []
  max_vals = []
  levels = []

  #print(x)

  for idx in range(count_runs):
    if CUDA:
      torch.cuda.empty_cache()

    print(f'run {idx}')
    # Create model and load trained parameters
    model = DenseNet3(num_classes=100, depth=40, growth_rate=40)
    model.load_state_dict(torch.load('./data/cifar100_40_40_0_ce_run1_best.pkl', map_location='cpu')['model'])
    #print(f'{count_parameters(model)} parameters')
    if CUDA:
      model.cuda()
    model.eval()

    x_sample = x[sample_id]
    x_class = y[sample_id]

    # DEBUG: Checking that example isn't misclassified
    #print('class', x_class.item(), 'classified as', model(x_sample.view(1,-1)).max(dim=1)[1].item())
    #raise Exception()

    def prop(x):
      y = model(x)
      y_diff = torch.cat((y[:,:x_class], y[:,(x_class+1):]),dim=1) - y[:,x_class].unsqueeze(-1)
      y_diff, _ = y_diff.max(dim=1)
      return y_diff #.max(dim=1)
    
    start = time.time()
    with torch.no_grad():
      lg_p, max_val, _, l = multilevel_uniform(prop, x_sample, sigma=sigma, rho=rho, count_particles=count_particles, count_mh_steps=count_mh_steps, debug=debug, stats=stats)
    end = time.time()
    print(f'Took {(end-start)/60} minutes...')
    #print('lg_p', lg_p, 'max_val', max_val)

    lg_ps.append(lg_p)
    max_vals.append(max_val)
    levels.append(l)
    if CUDA:
      torch.cuda.empty_cache()

    # Don't do further runs if maxes out at lower limit!
    if lg_p == -90.0:
      break

    if debug:
      print('lg_p', lg_p, 'max_val', max_val)

  with open(f'./results/cifar100/sample_{sample_id}_sigma_{sigma}_count_particles_{count_particles}_rho_{rho}_mh_{count_mh_steps}_seed_{seed}.pickle', 'wb') as handle:
    pickle.dump({'lg_ps':lg_ps, 'max_vals': max_vals, 'levels': levels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

# The sigma values that need more runs: [0.75, 0.7916667, 0.833333, 0.875, 0.9375]
sample_id = 1

with torch.no_grad():
  for count_mh_steps in [100, 250, 500, 1000, 2000]:
    for sigma in [0.02, 0.025, 0.03, 0.035, 0.04, 0.0433333, 0.0466667, 0.05, 0.055, 0.06, 0.07, 0.08]:
      print('sample', sample_id, 'sigma', sigma, 'rho', 0.1, 'count_particles', 300, 'count_mh_steps', count_mh_steps)
      do_run(count_runs=3, sample_id=sample_id, sigma=sigma, rho=0.1, count_particles=300, count_mh_steps=count_mh_steps, debug=True)
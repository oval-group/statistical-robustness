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

# Run this file from the main directory as: python -m exp_6_2.run_exp

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
if CUDA:
  cuda_id = '0'
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(cuda_id)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
matplotlib.rc('font', family='Latin Modern Roman')

from exp_6_2.network import SimpleLinear
import utils

def cm2inch(value):
  return value/2.54

# Fix random seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

def multilevel_uniform(
      model, domain, 
      rho=0.1,
      count_particles=1000,
      count_mh_steps=100,
      width_proposal=0.01,
      debug=True, stats=False):
  if CUDA:
    low_params = domain[:,0].cuda()
    high_params = domain[:,1].cuda()
  else:
    low_params = domain[:,0]
    high_params = domain[:,1]

  prior = dist.Uniform(low=low_params, high=high_params)

  # Parameters
  if CUDA:
    width_proposal = width_proposal*torch.ones(count_particles).cuda()
  else:
    width_proposal = width_proposal*torch.ones(count_particles)
  count_max_levels = 500
  target_acc_ratio = 0.9
  max_width_proposal = 0.1
  min_width_proposal = 1e-8
  width_inc = 1.02
  width_dec = 0.5

  # Sample the initial particles
  x = prior.sample(torch.Size([count_particles]))
  L_prev = -math.inf
  L = -math.inf
  lg_p = 0
  max_val = -math.inf
  levels = []

  # Loop over levels
  for level_idx in range(count_max_levels):
    if L >= 0:
      break

    # Calculate current level
    s_x = model(x).squeeze(-1)
    max_val = max(max_val, s_x.max().item())
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
      return -math.inf, max_val, x, levels

    # Update our estimate of the log-probability
    #if debug:
    #  print(f'Term for level {level_idx+1}', torch.log((s_x >= L).float().sum()).item() - math.log(count_particles))

    lg_p += torch.log(count_keep.float()).item() - math.log(count_particles)

    # Uniformly resample killed particles below the level
    new_idx = torch.randint(low=0, high=count_keep, size=(count_kill,), dtype=torch.long)
    x = x[where_keep]
    x_new = x[new_idx]
    x = torch.cat((x, x[new_idx]), dim=0)
    width_proposal = width_proposal[where_keep]
    width_new = width_proposal[new_idx]
    width_proposal = torch.cat((width_proposal, width_new), dim=0)
    
    #acc_ratio = torch.zeros(count_kill).cuda()
    if CUDA:
      acc_ratio = torch.zeros(count_particles).cuda()
    else:
      acc_ratio = torch.zeros(count_particles)

    for mh_idx in range(count_mh_steps):
      # Propose new sample
      #g_bottom = dist.Uniform(low=torch.max(low_params, x - width_proposal.unsqueeze(-1)), high=torch.min(high_params, x + width_proposal.unsqueeze(-1)))
      g_bottom = dist.Uniform(low=x - width_proposal.unsqueeze(-1), high=x + width_proposal.unsqueeze(-1))
      x_maybe = g_bottom.sample()
      s_x = model(x_maybe).squeeze(-1)

      # Work out which ones to accept
      acc_idx = (s_x >= L) & (x_maybe >= low_params).all(dim=1) & (x_maybe <= high_params).all(dim=1)
      acc_ratio += acc_idx.float()
      x = torch.where(acc_idx.unsqueeze(-1), x_maybe, x)

      s_x = model(x).squeeze(-1)
      max_val = max(max_val, s_x.max().item())
      
    if stats:
      utils.stats(s_x)

    # Adapt the width proposal *for each chain individually*
    acc_ratio /= count_mh_steps
    #print(acc_ratio.size())
    width_proposal = torch.where(acc_ratio > 0.9, width_proposal*width_inc, width_proposal)
    width_proposal = torch.where(acc_ratio < 0.9, width_proposal*width_dec, width_proposal)

    L_prev = L
    #input()

  # We return both the estimate of the log-probability of the integral and the set of adversarial examples
  s_x = model(x).squeeze(-1)
  #max_val = max(max_val, x.max().item())
  max_val = s_x.max().item()
  return lg_p, max_val, x, levels

# Open the summary because we want to pick out certain properties!
summary = pickle.load(open(f'./data/collisionDetection/summary.pickle', "rb" ))

# Load the model/property
#print(np.where(np.round(np.array(summary['lg_ps']), decimals=0) == -23))
#raise Exception()

def do_run(count_runs, property_id, rho, count_particles, count_mh_steps, debug=False):
  lg_ps = []
  max_vals = []
  levels = []

  data = pickle.load(open(f'./data/collisionDetection/property{property_id}.pickle', "rb" ))
  if CUDA:
    domain = torch.tensor(data['domain']).cuda()
  else:
    domain = torch.tensor(data['domain'])

  for idx in range(count_runs):
    print(f'run {idx}')
    model = SimpleLinear(data['layers'], data['domain'])
    if CUDA:
      model.cuda()
    
    lg_p, max_val, _, l = multilevel_uniform(model, domain, rho=rho, count_particles=count_particles, count_mh_steps=count_mh_steps, debug=debug)
    lg_ps.append(lg_p)
    max_vals.append(max_val)
    levels.append(l)
    if CUDA:
      torch.cuda.empty_cache()

    if debug:
      print('lg_p', lg_p, 'max_val', max_val)

  with open(f'./results/collisionDetection/property_{property_id}_count_particles_{count_particles}_count_mh_steps_{count_mh_steps}_rho_{rho}.pickle', 'wb') as handle:
    pickle.dump({'lg_ps':lg_ps, 'max_vals': max_vals, 'levels': levels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

# DEBUG: Seeing which properties were formally verified (i.e. counterexample found) when naive MC estimator reported p(s(x)>0)=0
"""for idx in range(500):
  if summary['lg_ps'][idx] == -math.inf and summary['SATs'][idx] == True:
    print('id', idx, 'max_val', summary['max_vals'][idx])"""

print(f'{len(summary)} properties')
for property_id in range(0, 500):
  if summary['SATs'][property_id] == False:
    with torch.no_grad():
      print(f'Property {property_id} - UNSAT')
      do_run(count_runs=30, property_id=property_id, rho=0.1, count_particles=100000, count_mh_steps=100, debug=True)
      #do_run(count_runs=30, property_id=property_id, rho=0.1, count_particles=10, count_mh_steps=100, debug=True)
  else:
    for rho in [0.1, 0.25, 0.5]:
      for count_particles in [1000, 10000, 100000]:
        for count_mh_steps in [100, 250, 1000]:
          with torch.no_grad():
            print(f'Property {property_id} - SAT')
            print(f'rho {rho}, count_particles {count_particles}, count_mh_steps {count_mh_steps}')
            do_run(count_runs=30, property_id=property_id, rho=rho, count_particles=count_particles, count_mh_steps=count_mh_steps, debug=True)
            #do_run(count_runs=30, property_id=property_id, rho=rho, count_particles=10, count_mh_steps=count_mh_steps, debug=True)
            
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

# Run this file from the main directory as: python -m collision.process_data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

CUDA = True

import os
import time
import pickle
import math
import time

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
from plnn.model import load_and_simplify2
import utils

def cm2inch(value):
  return value/2.54

# Fix random seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

from os import listdir
from os.path import isfile, join
mypath = './data/collisionDetection/'
onlyfiles = sorted([f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('rlv')])

#for f in onlyfiles:
#  print(f)

print(len(onlyfiles))

# This converts the rlv to a simple Python pickle that also includes the domain, whether the property is SAT or UNSAT,
# brute force estimation of the integral value, and the maximum value of the property found by brute-force
def save_data(filename, outputname):
  # Load model and domain
  with open(f'./data/collisionDetection/{filename}', 'r') as f:
    layers_cpu, domain_cpu = load_and_simplify2(f)
  model = SimpleLinear(layers_cpu, domain_cpu)
  if CUDA:
    model.cuda()
    domain = domain_cpu.clone().detach().cuda()
  else:
    domain = domain_cpu.clone().detach()

  lg_p, max_val, count, total = brute_force(model, domain, count_iterations=10000)

  with open(f'./data/collisionDetection/{outputname}.pickle', 'wb') as handle:
    pickle.dump({'layers':layers_cpu, 'domain':domain_cpu, 'lg_p':lg_p, 'max_val':max_val, 'count':count, 'total':total, 'SAT':not filename.endswith('UNSAT.rlv')}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def brute_force(model, domain, count_iterations=100):
  count_above = int(0)
  count_total = int(0)
  if CUDA:
    low_params = domain[:,0].cuda()
    high_params = domain[:,1].cuda()
  else:
    low_params = domain[:,0]
    high_params = domain[:,1]

  prior = dist.Uniform(low=low_params, high=high_params)

  count_particles = int(1000000)

  start = time.time()
  max_val = -math.inf
  for i in range(count_iterations):
    x = prior.sample(torch.Size([count_particles]))
    s_x = model(x).squeeze(-1)
    count_above += int((s_x >= 0).float().sum().item())
    count_total += count_particles
    max_val = max(s_x.max().item(), max_val)

    if i % 1000 == 0:
      time_per_iter = (time.time() - start) / (i+1)
      time_left = (count_iterations - i) * time_per_iter / 60
      print(f'{i/count_iterations*100}% done ({round(time_left,3)} mins left)')

  print(f'{count_above} adversarial examples observed from {count_total} samples')
  if count_above > 0:
    return math.log(count_above) - math.log(count_total), max_val, count_above, count_total
  else:
    return -math.inf, max_val, count_above, count_total

# Convert 500 properties from rlv to pickle and save useful metadata
with torch.no_grad():
  for idx, f in enumerate(onlyfiles):
    f = onlyfiles[idx]
    print(f'Property {idx}, file {f}')
    save_data(f, f'property{idx}')

# Save all the solutions into a single pickle for convenience (e.g. quickly working out which properties are SAT)
lg_ps = []
max_vals = []
SATs = []
counts = []
totals = []
for idx in range(500):
  data = pickle.load(open(f'./data/collisionDetection/property{idx}.pickle', "rb" ))
  lg_ps.append(data['lg_p'])
  max_vals.append(data['max_val'])
  SATs.append(data['SAT'])
  counts.append(data['count'])
  totals.append(data['total'])

with open(f'./data/collisionDetection/summary.pickle', 'wb') as handle:
  pickle.dump({'lg_ps':lg_ps, 'max_vals':max_vals, 'SATs':SATs, 'counts':counts, 'totals':totals}, handle, protocol=pickle.HIGHEST_PROTOCOL)

#print(lg_ps)

#summary = pickle.load(open(f'../data/collisionDetection/summary.pickle', "rb" ))
#for p, s in zip(summary['lg_ps'], summary['SATs']):
#  print((s == False and p == -math.inf) or (s == True and p != -math.inf))
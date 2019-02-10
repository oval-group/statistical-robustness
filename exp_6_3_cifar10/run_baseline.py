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
import time, math
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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from exp_6_3_cifar10.network import SimpleMlp
import utils

def cm2inch(value):
  return value/2.54

# The bounds in NN-space
if CUDA:
  x_min = torch.tensor([(0.0 - 0.4914)/0.2023, (0.0 - 0.4822)/0.1994, (0.0 - 0.4465)/0.2010]).cuda()
  x_max = torch.tensor([(1.0 - 0.4914)/0.2023, (1.0 - 0.4822)/0.1994, (1.0 - 0.4465)/0.2010]).cuda()
else:
  x_min = torch.tensor([(0.0 - 0.4914)/0.2023, (0.0 - 0.4822)/0.1994, (0.0 - 0.4465)/0.2010])
  x_max = torch.tensor([(1.0 - 0.4914)/0.2023, (1.0 - 0.4822)/0.1994, (1.0 - 0.4465)/0.2010])


# The bounds in NN-space
#x_min = (0-0.1307)/0.3081
#x_max = (1-0.1307)/0.3081

# Fixing random seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Load data
#kwargs = {'num_workers': 1, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
  #data = data.view(-1, 3, 32, 32)
  x[(idx*1000):((idx+1)*1000),:,:,:] = data
  y[(idx*1000):((idx+1)*1000)] = target
  
# Create model and load trained parameters
model = SimpleMlp()
model.load_state_dict(torch.load('./data/cifar10_simplemlp.pickle'))
if CUDA:
  model.cuda()

def run(sigma, sample_id):
  x_sample = x[sample_id]
  x_class = y[sample_id]
  print('class', x_class.item())

  # Perturbation model
  prior = dist.Uniform(low=torch.max(x_sample-sigma*(x_max-x_min).view(3,1,1), x_min.view(3,1,1)), high=torch.min(x_sample+sigma*(x_max-x_min).view(3,1,1), x_max.view(3,1,1)))

  def prop(x):
    y = model(x)
    y_diff = torch.cat((y[:,:x_class], y[:,(x_class+1):]),dim=1) - y[:,x_class].unsqueeze(-1)
    y_diff, _ = y_diff.max(dim=1)
    return y_diff

  def brute_force(prop, x_sample, count_iterations=10):
    count_above = int(0)
    count_total = int(0)
    count_particles = int(10000)

    start = time.time()
    max_val = -math.inf
    for i in range(count_iterations):
      # Perturb the input
      # Implements parallel batched accept-reject sampling.
      x = prior.sample(torch.Size([count_particles]))

      #print(x_min, x_max)
      #utils.stats(from_logit_space(x))
      #raise Exception()

      # Calculate the property, how many satisfy it, and maximum value
      s_x = prop(x.view(count_particles, -1)).squeeze(-1)
      count_above += int((s_x >= 0).float().sum().item())
      count_total += count_particles
      max_val = max(s_x.max().item(), max_val)

      #utils.stats(s_x)
      #raise Exception()

      if i % 1000 == 0:
        time_per_iter = (time.time() - start) / (i+1)
        time_left = (count_iterations - i) * time_per_iter / 60
        print(f'{i/count_iterations*100}% done ({round(time_left,3)} mins left)')

    print(f'{count_above} adversarial examples observed from {count_total} samples')
    if count_above > 0:
      return math.log(count_above) - math.log(count_total), max_val, count_above, count_total
    else:
      return -math.inf, max_val, count_above, count_total

  with torch.no_grad():
    lg_p, max_val, count, total = brute_force(prop, x_sample, count_iterations=500000)
    #lg_p, max_val, count, total = brute_force(prop, x_sample, count_iterations=100)

  with open(f'./results/cifar10/sample_{sample_id}_sigma_{sigma}_uniform_brute.pickle', 'wb') as handle:
    pickle.dump({'lg_p':lg_p, 'max_val': max_val, 'count':count, 'total':total}, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print('sample', sample_id, 'sigma', sigma, 'lg_p', lg_p, 'max_val', max_val, 'count', count, 'total', total)

for sigma in reversed([0.0366667, 0.04, 0.0433333, 0.0466667, 0.05, 0.055, 0.06, 0.07, 0.08]):
  run(sigma, 0)
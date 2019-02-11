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
#matplotlib.use('pdf')
matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
matplotlib.rc('font', family='Latin Modern Roman')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import problems as pblm
from convex_adversarial import robust_loss, robust_loss_parallel

def cm2inch(value):
  return value/2.54

def extract_exp_results():
  count_particles = 10000
  count_mh_steps = 1000
  rho = 0.1
  lg_ps = {}
  for sample_id in range(50):
    for sigma in [0.3, 0.2, 0.1]:
      print(f'sample {sample_id}')
      lg_ps[(sample_id,sigma)] = []
      for epoch in [-1,0,2,4,6,9,14,19,24,32,41,49,61,74,86,99]:
        result = pickle.load(open(f'./snapshots/mnist_robust_sample_{sample_id}_sigma_{sigma}_count_particles_{count_particles}_rho_{rho}_mh_{count_mh_steps}_epoch_{epoch}_uniform.pickle', 'rb'))

        #print(np.array(result['lg_ps']).shape)
        #raise Exception()

        lg_p = np.array(result['lg_ps'])
        if np.array(result['lg_ps']).shape[0] != 5:
          lg_p = np.tile(lg_p[0], 5)
        
        lg_ps[(sample_id,sigma)].append(lg_p)

        if lg_ps[(sample_id,sigma)][-1][0] == -250:
          if not ((lg_ps[(sample_id,sigma)][-1] == -250.).all()):
            print('Found value at limit!')
            print(lg_ps[(sample_id,sigma)][-1])
            raise Exception()

      #print(np.array(lg_ps[(sample_id,sigma)]).shape)
  
  with open(f'./snapshots/mnist_extracted_exp_results.pickle', 'wb') as handle:
    pickle.dump(lg_ps, handle, protocol=pickle.HIGHEST_PROTOCOL)

extract_exp_results()
#raise Exception()

def plot_exp_all():
  # Open the summary because we want to pick out certain properties!
  results = pickle.load(open(f'./snapshots/mnist_extracted_exp_results.pickle', 'rb' ))
  xs = np.array([-1,0,2,4,6,9,14,19,24,32,41,49,61,74,86,99])

  fig = plt.figure(figsize=(cm2inch(50.0), cm2inch(60.0)))
  for sample_id in range(50):
    #ax = fig.add_subplot(6, 6, sample_id+1, xlabel='epoch', ylabel='lg(I)', ylim=(-100.0,0.0))
    ax = fig.add_subplot(10, 5, sample_id+1) #, ylim=(-45.0,0.0))
    for sigma in [0.3, 0.2, 0.1]:
      if sigma == 0.3:
        color = 'navy'
      elif sigma == 0.2:
        color = 'seagreen'
      else:
        color = 'firebrick'

      lg_ps = np.array(results[(sample_id,sigma)])
      mean_ps = np.zeros_like(np.mean(lg_ps, axis=1))
      std_ps = np.zeros_like(np.mean(lg_ps, axis=1))
      for idx in range(lg_ps.shape[0]):
        p = np.exp(lg_ps[idx])
        p = p[p > 0]
        mean_ps[idx] = np.mean(p)
        #if math.log(mean_ps[idx]) == -100.0:
        #  mean_ps[idx] = 1e-100
        std_ps[idx] = np.std(p)/math.sqrt(5)

      for idx in range(mean_ps.shape[0]):
        x = mean_ps[idx]
        std = std_ps[idx]
        lower = x - std
        upper = min(1.0, x + std)

        if lower < 0:
          ax.scatter(np.array([xs[idx]]), np.log(np.array([x]))/math.log(10), color='red', marker='.', linewidth=2.0)
        else:
          ax.plot(np.array([xs[idx], xs[idx]]), np.log(np.array([lower, upper]))/math.log(10), color=color, linewidth=2.0)
          ax.scatter(np.array([xs[idx]]), np.log(np.array([x]))/math.log(10), color=color, marker='.', linewidth=2.0)

      ax.plot(xs, np.log(mean_ps)/math.log(10), color=color)

    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    sns.despine()

  #fig.savefig(f'mnist_test_robust_losses.pdf', bbox_inches='tight')
  fig.savefig(f'./results/robust/mnist_robust_curves.svg', bbox_inches='tight')
  plt.close(fig)

def plot_exp_summary():
  # Open the summary because we want to pick out certain properties!
  results = pickle.load(open(f'./snapshots/mnist_extracted_exp_results.pickle', 'rb' ))
  xs = np.array([-1,0,2,4,6,9,14,19,24,32,41,49,61,74,86,99])

  fig = plt.figure(figsize=(cm2inch(8.0), cm2inch(6.0)))
  ax = fig.add_subplot(1, 1, 1, xlabel='epoch', ylabel=r'$\log_{10}(\mathcal{I})$') #, ylim=(-45.0,0.0))

  for sigma in [0.3, 0.2, 0.1]:
    sigma_lg_ps = []
    if sigma == 0.3:
      color = 'navy'
    elif sigma == 0.2:
      color = 'seagreen'
    else:
      color = 'firebrick'
    
    for sample_id in range(50):
      lg_ps = np.array(results[(sample_id,sigma)])
      mean_ps = np.zeros_like(np.mean(lg_ps, axis=1))
      for idx in range(lg_ps.shape[0]):
        p = np.exp(lg_ps[idx])
        p = p[p > 0]
        mean_ps[idx] = np.mean(p)
        #if math.log(mean_ps[idx]) == -100.0: #and sigma != 0.1:
        #  mean_ps[idx] = 1e-100

      sigma_lg_ps.append(mean_ps)

    sigma_lg_ps = np.array(sigma_lg_ps)
    #print(sigma_lg_ps.shape)

    median = np.median(sigma_lg_ps, axis=0)/math.log(10)
    upper = np.percentile(sigma_lg_ps, 75, axis=0)/math.log(10)
    lower = np.percentile(sigma_lg_ps, 25, axis=0)/math.log(10)

    ax.fill_between(xs, np.log(lower)/math.log(10), np.log(upper)/math.log(10), color=color, alpha=0.3)
    ax.plot(xs, np.log(median)/math.log(10), color=color, marker='.', linewidth=1.0, label=r'$\epsilon='+str(sigma)+r'$')

    #raise Exception()

  #ax.legend() #loc='lower right')
  ax.legend(bbox_to_anchor=(0.9,0.15), loc="lower right",  bbox_transform=fig.transFigure)
  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()

  #fig.savefig(f'mnist_test_robust_losses.pdf', bbox_inches='tight')
  fig.savefig(f'./results/robust/mnist_robust_summary.svg', bbox_inches='tight')
  plt.close(fig)

def plot_certificates():
  # Load data
  # The bounds in NN-space
  x_min = 0.
  x_max = 1.

  # Fix random seed for reproducibility
  seed = 0
  np.random.seed(seed)
  torch.manual_seed(seed)

  _, test_loader = pblm.mnist_loaders(50)

  # Get data into arrays for convenience
  for idx, (data, target) in enumerate(test_loader):
    if CUDA:
      data, target = data.float().cuda(), target.long().cuda()
    else:
      data, target = data.float(), target.long()
    #print(data.size())
    #raise Exception()

    data = data.view(-1, 1, 28, 28)
    x = data
    y = target
    break

  #print(x.size(), y.size())
  #raise Exception()

  #epochs = np.array([-1,0,2,4,6,9,14,19,24,32,41,49,61,74,86,99])
  epochs = np.array([32,41,49,61,74,86,99])
  robust_errs = []
  for epoch in epochs:
    epsilon = 0.1
    
    model = pblm.mnist_model()
    if epoch == -1:
      model.load_state_dict(torch.load('./snapshots/mnist_baseline_batch_size_50_epochs_100_lr_0.001_opt_adam_real_time_False_seed_0_checkpoint_99.pth'))
    else:
      model.load_state_dict(torch.load(f'./snapshots/mnist_robustified_robust_batch_size_50_epochs_100_epsilon_0.1_l1_test_exact_l1_train_exact_lr_0.001_opt_adam_real_time_False_schedule_length_50_seed_0_starting_epsilon_0.01_checkpoint_{epoch}.pth'))
    if CUDA:
      model.cuda()

    _, robust_err = robust_loss(model, epsilon, x, y)
    robust_errs.append(robust_err)
  
  robust_errs = np.array(robust_errs)

  results = pickle.load(open(f'./snapshots/mnist_extracted_exp_results.pickle', 'rb' ))
  xs = np.array([-1,0,2,4,6,9,14,19,24,32,41,49,61,74,86,99])

  our_results = {}
  for sigma in [0.1, 0.2, 0.3]:
    sigma_lg_ps = []
    our_results[sigma] = np.zeros(16)
    
    for sample_id in range(50):
      #print('sample id', sample_id)
      #if sample_id == 24:
      #  print([r.shape for r in results[(sample_id,sigma)]])
      #  lg_ps = np.array(results[(sample_id,sigma)])
      #  print('lg_ps', lg_ps.shape)
      #  
      #  input()

      lg_ps = np.array(results[(sample_id,sigma)])

      #print('lg_ps', lg_ps.shape)  
      #input()

      #print(lg_ps.shape)
      #if len(lg_ps.shape) == 1:
      #  lg_ps = lg_ps.reshape((-1,1))
      #  #print(lg_ps.shape)
      mean_ps = np.mean(lg_ps, axis=1)

      #print(mean_ps.shape, lg_ps.shape)

      our_results[sigma] += mean_ps != -250.0

      #print(mean_ps)
      #input()
      #raise Exception()

      #print(mean_ps.shape)
      #raise Exception()
    
    our_results[sigma] /= 50.0
  
  fig = plt.figure(figsize=(cm2inch(8.0), cm2inch(6.0)))
  ax = fig.add_subplot(1, 1, 1, xlabel='epoch', ylabel='fraction certified', ylim=(-0.05,1.0))

  ax.plot(xs, 1. - our_results[0.3], color='navy', marker='.', linewidth=1.0, label=r'AMLS $\epsilon=0.3$')
  ax.plot(xs, 1. - our_results[0.2], color='seagreen', marker='.', linewidth=1.0, label=r'AMLS $\epsilon=0.2$')
  ax.plot(xs, 1. - our_results[0.1], color='firebrick', marker='.', linewidth=1.0, label=r'AMLS $\epsilon=0.1$')
  ax.plot(epochs, 1. - robust_errs, color='grey', marker='.', linestyle='--', linewidth=1.0, label=r'W\&K $\epsilon=0.1$')


  #ax.legend(loc='lower right')
  ax.legend(bbox_to_anchor=(0.9,0.15), loc="lower right",  bbox_transform=fig.transFigure)
  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()

  #fig.savefig(f'mnist_test_robust_losses.pdf', bbox_inches='tight')
  fig.savefig(f'./results/robust/mnist_certificates.svg', bbox_inches='tight')
  plt.close(fig)

plot_exp_all()
plot_exp_summary()
plot_certificates()
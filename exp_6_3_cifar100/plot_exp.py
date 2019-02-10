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

import os.path, pickle, math
import numpy as np
import matplotlib
#matplotlib.use('pdf')
matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
#matplotlib.rc('font', family='Latin Modern Roman')
#mathtext.fontset
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

viridis = plt.get_cmap('viridis').colors
viridis = [viridis[i] for i in [100, 240, 150, 0]]
tab20b = plt.get_cmap('tab20b').colors
colors = [None for _ in range(4)]
colors[0] = viridis[0]
colors[1] = plt.get_cmap('Paired').colors[3]
colors[2] = tab20b[13]
colors[3] = (0.3, 0.3, 0.3)

def cm2inch(value):
  return value/2.54

ss = [0.02, 0.025, 0.03, 0.035, 0.04, 0.0433333, 0.0466667, 0.05, 0.055, 0.06, 0.07, 0.08]

## Produce the maximum value found for each property
def extract_exp_results():
  count_particles = 300
  #count_mh_steps = 10000
  rho = 0.1
  lg_ps = {}
  naive_lg_ps = {}

  for sample_id in [1]:
    print(f'sample {sample_id}')
    for count_mh_steps in [2000, 1000, 500, 250, 100]:  
      #for sigma in [0.04, 0.0433333, 0.0466667, 0.05, 0.055, 0.06, 0.07, 0.08]:
      for sigma in ss:
        naive_lg_ps[(sample_id, count_mh_steps, sigma)] = []
        lg_ps[(sample_id, count_mh_steps, sigma)] = []
        # NOTE: Seed is set to CUDA GPU card id in run_baseline, and this code only accounts for up to 8 cards
        for seed in range(8):
          try:
            result = pickle.load(open(f'./results/cifar100/sample_{sample_id}_sigma_{sigma}_count_particles_{count_particles}_rho_{rho}_mh_{count_mh_steps}_seed_{seed}.pickle', 'rb'))
            lg_ps[(sample_id, count_mh_steps, sigma)].append(np.array(result['lg_ps'])) #/ math.log(10))
          except:
            pass
        
          try:
            result = pickle.load(open(f'./results/cifar100/sample_{sample_id}_sigma_{sigma}_seed_{seed}_uniform_brute.pickle', 'rb'))
            naive_lg_ps[(sample_id, count_mh_steps, sigma)].append(np.array(result['lg_ps'])) #/ math.log(10))
          except:
            pass
  
  with open(f'./results/cifar100/extracted_exp_results.pickle', 'wb') as handle:
    pickle.dump({'lg_ps':lg_ps, 'naive_lg_ps':naive_lg_ps}, handle, protocol=pickle.HIGHEST_PROTOCOL)

extract_exp_results()

def plot_exp_results():
  results = pickle.load(open('./results/cifar100/extracted_exp_results.pickle', 'rb'))

  for sample_id in [1]:
    for count_mh_steps in [2000]:
      lg_ps = []
      naive_lg_ps = []

      for sigma in ss:
        lg_ps.append(np.array(results['lg_ps'][(sample_id, count_mh_steps, sigma)]).reshape(-1))
        naive_lg_ps.append(np.array(results['naive_lg_ps'][(sample_id, count_mh_steps, sigma)]).reshape(-1))
      
      #print(lg_ps)
      #print(naive_lg_ps)
      #raise Exception()
      
      #naive_lg_ps = np.array(summary['lg_ps'])[ids]
      #naive_lg_ps = [-3.711534144678742, -3.9220733412816475]
      
      xs = np.array(ss)
      #xs = np.array([0.04, 0.0433333, 0.0466667, 0.05, 0.055, 0.06, 0.07, 0.08])

      # Calculate means and standard errors
      mean_ps = np.zeros((len(lg_ps)))
      std_ps = np.zeros((len(lg_ps)))
      for idx in range(len(lg_ps)):
        p = np.exp(lg_ps[idx])
        #print(p)
        p = p[p > 0]
        #if p.shape[0] < 30:
        #  print(p.shape[0])
        mean_ps[idx] = np.mean(p)
        std_ps[idx] = 1*np.std(p)/math.sqrt(p.shape[0])
        #print(p.shape[0])

      mean_naive = np.zeros((len(naive_lg_ps)))
      std_naive = np.zeros((len(naive_lg_ps)))
      for idx in range(len(naive_lg_ps)):
        p = np.exp(naive_lg_ps[idx])
        #print(p)
        p = p[p > 0]
        #if p.shape[0] < 30:
        #  print(p.shape[0])
        mean_naive[idx] = np.mean(p)
        std_naive[idx] = 3*np.std(p)/math.sqrt(p.shape[0])

        #print(p)  #np.std(p))

        #print(p.shape[0])

      #raise Exception()

      fig = plt.figure(figsize=(cm2inch(8.0), cm2inch(6.0)))
      #fig.suptitle(f'CollisionDetection, Property {property_id}, lg(p)={brute_val}')
      ax = fig.add_subplot(1, 1, 1, xlabel=r'$\epsilon$', ylabel=r'$\log_{10}(\mathcal{I})$', ylim=(-8.0,0)) #, xlim=(0.03, 0.1))
      #ax.hlines(y=0, linestyle='--', xmin=xs[0], xmax=xs[-1], linewidth=0.75, color='black') #color='red',
      #ax.hlines(y=-100, color='black', xmin=xs[0], xmax=xs[-1], linestyle='--', linewidth=0.75)

      for idx in range(mean_ps.shape[0]):
        x = mean_ps[idx]
        std = std_ps[idx]
        lower = x - std
        upper = min(1.0, x + std)

        x2 = mean_naive[idx]
        std2 = std_naive[idx]
        lower2 = x2 - std2
        upper2 = min(1.0, x2 + std2)

        print(std2)

        if lower < 0:
          ax.scatter(np.array([xs[idx]]), np.log(np.array([x]))/math.log(10), color=colors[2], marker='.', linewidth=0.5)
        else:
          ax.plot(np.array([xs[idx], xs[idx]]), np.log(np.array([lower2, upper2]))/math.log(10), color='orange', linewidth=2.5, zorder=0)
          ax.scatter(np.array([xs[idx]]), np.log(np.array([x2]))/math.log(10), color='orange', marker='.', linewidth=2.5, zorder=0)

          ax.plot(np.array([xs[idx], xs[idx]]), np.log(np.array([lower, upper]))/math.log(10), color=colors[0], linewidth=0.5)
          ax.scatter(np.array([xs[idx]]), np.log(np.array([x]))/math.log(10), color=colors[0], marker='.', linewidth=0.5)

      ax.plot(xs, np.log(mean_ps)/math.log(10), color=colors[0], linestyle='-', label=r'$M={}$'.format(count_mh_steps))

      #for idx in range(len(lg_ps)):
      #  ax.scatter(xs[idx]*np.ones_like(lg_ps[idx]), lg_ps[idx] - 10, linewidth=0.5, marker='.', color='purple')
        
      #ax.plot(xs, np.log(mean_ps), color=colors[0], linewidth=1.0)
      #print(naive_lg_ps)
      #print(naive_counts)

      ax.plot(xs, np.log(mean_naive)/math.log(10), color='orange', linewidth=2.5, linestyle='-', zorder=0, label='naive MC')

      ax.legend(loc='lower right')
      ax.xaxis.set_tick_params(width=0.5)
      ax.yaxis.set_tick_params(width=0.5)
      ax.spines['left'].set_linewidth(0.5)
      ax.spines['bottom'].set_linewidth(0.5)
      sns.despine()

      fig.savefig(f'cifar100_exp1_sample_{sample_id}_mhsteps_{count_mh_steps}.svg', bbox_inches='tight')
      plt.close(fig)

    #raise Exception()

def plot_exp_errors():
  results = pickle.load(open('./results/cifar100/extracted_exp_results.pickle', 'rb'))

  for sample_id in [1]:
    mean_ps = {}
    std_ps = {}
    fig = plt.figure(figsize=(cm2inch(8.0), cm2inch(6.0)))
    ax = fig.add_subplot(1, 1, 1, xlabel=r'$\epsilon$', ylabel=r'$\Delta\log_{10}(\mathcal{I})$') #, ylim=(-50,0))

    for count_mh_steps in [2000, 100, 250, 500, 1000]:
      lg_ps = []
      for sigma in ss:
        lg_ps.append(np.array(results['lg_ps'][(sample_id, count_mh_steps, sigma)]).reshape(-1))

      xs = np.array(ss)

      # Calculate means and standard errors
      mean_ps[count_mh_steps] = np.zeros((len(lg_ps)))
      std_ps[count_mh_steps] = np.zeros((len(lg_ps)))
      for idx in range(len(lg_ps)):
        p = np.exp(lg_ps[idx])
        #print(p)
        p = p[p > 0]
        #if p.shape[0] < 30:
        #  print(p.shape[0])
        mean_ps[count_mh_steps][idx] = np.mean(p)
        std_ps[count_mh_steps][idx] = 1.5*np.std(p)/math.sqrt(p.shape[0])
        #print(p.shape[0])

      if count_mh_steps != 2000:
        if count_mh_steps == 100:
          color = 'firebrick'
        elif count_mh_steps == 250:
          color = 'goldenrod'
        elif count_mh_steps == 500:
          color = '#bfbf00'
        elif count_mh_steps == 1000:
          color = '#556b2f'

        for idx in range(mean_ps[count_mh_steps].shape[0]):
          x = mean_ps[count_mh_steps][idx]
          std = std_ps[count_mh_steps][idx]

          #lower = max(x - 1e-200, x - std)
          lower = x - std
          upper = min(1.0, x + std)

          #if lower < 0:
          #  ax.scatter(np.array([xs[idx]]), np.log(np.array([x])), color=colors[2], marker='.', linewidth=0.5)
          #else:
          ax.plot(np.array([xs[idx], xs[idx]]), (np.log(np.array([mean_ps[2000][idx]])) - np.log(np.array([lower, upper]))), color=color, linewidth=0.5)
          #ax.scatter(np.array([xs[idx]]), np.log(np.array([x])), color=colors[0], marker='.', linewidth=0.5)

        ax.plot(xs, (np.log(mean_ps[2000]) - np.log(mean_ps[count_mh_steps])), color=color, marker='.', linestyle='-', label=r'$M={}$'.format(count_mh_steps))

    ax.legend()
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    sns.despine()

    fig.savefig(f'cifar100_exp1_sample_{sample_id}_errors.svg', bbox_inches='tight')
    plt.close(fig)

plot_exp_results()
plot_exp_errors()
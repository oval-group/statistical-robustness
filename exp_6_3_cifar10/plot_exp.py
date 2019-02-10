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

# The bounds in NN-space
#x_min = (0-0.1307)/0.3081
#x_max = (1-0.1307)/0.3081
#x_range = x_max - x_min

# Open the summary because we want to pick out certain properties!
#summary = pickle.load(open(f'./data/collisionDetection/summary.pickle', "rb" ))

## Produce the maximum value found for each property
def extract_exp_results():
  #naive_lg_ps = []

  count_particles = 10000
  #count_mh_steps = 10000
  rho = 0.1
  lg_ps = {}
  naive_lg_ps = {}
  naive_counts = {}
  naive_totals = {}
  for sample_id in range(1):
    print(f'sample {sample_id}')
    for count_mh_steps in [100, 250, 500, 1000, 2000]:
      lg_ps[(sample_id, count_mh_steps)] = []
      naive_lg_ps[(sample_id, count_mh_steps)] = []
      naive_counts[(sample_id, count_mh_steps)] = []
      naive_totals[(sample_id, count_mh_steps)] = []
      for sigma in [0.0366667, 0.04, 0.0433333, 0.0466667, 0.05, 0.055, 0.06, 0.07, 0.08]:
        result = pickle.load(open(f'./results/cifar10/sample_{sample_id}_sigma_{sigma}_count_particles_{count_particles}_rho_{rho}_mh_{count_mh_steps}_uniform.pickle', 'rb'))
        lg_ps[(sample_id, count_mh_steps)].append(np.array(result['lg_ps']))
        #print(np.array(result['lg_ps']).mean())

        if sigma > 0.05:
          result = pickle.load(open(f'./results/cifar10/sample_{sample_id}_sigma_{sigma}_uniform_brute.pickle', 'rb'))
          naive_lg_ps[(sample_id, count_mh_steps)].append(result['lg_p'])
        else:
          naive_lg_ps[(sample_id, count_mh_steps)].append(-math.inf)
        #naive_counts[(sample_id, count_mh_steps)].append(result['count'])
        #naive_totals[(sample_id, count_mh_steps)].append(result['total'])
        #print(result['lg_p'])

  
  with open(f'./results/cifar10/extracted_exp_results.pickle', 'wb') as handle:
    pickle.dump({'lg_ps':lg_ps, 'naive_lg_ps':naive_lg_ps, 'naive_counts':naive_counts, 'naive_totals':naive_totals}, handle, protocol=pickle.HIGHEST_PROTOCOL)

extract_exp_results()  

def plot_exp_results():
  results = pickle.load(open('./results/cifar10/extracted_exp_results.pickle', 'rb'))

  for sample_id in range(1):
    #for count_mh_steps in [100, 250, 500, 1000]:
    for count_mh_steps in [2000]:
      lg_ps = results['lg_ps'][(sample_id, count_mh_steps)]
      naive_lg_ps = results['naive_lg_ps'][(sample_id, count_mh_steps)]
      
      #naive_lg_ps = np.array(summary['lg_ps'])[ids]
      #xs = np.array([0.75, 0.875, 1.0, 1.125, 1.250, 1.375, 1.5])
      xs = np.array([0.0366667, 0.04, 0.0433333, 0.0466667, 0.05, 0.055, 0.06, 0.07, 0.08])

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
        std_ps[idx] = 1.5*np.std(p)/math.sqrt(p.shape[0])
        #print(p.shape[0])

      #raise Exception()

      fig = plt.figure(figsize=(cm2inch(8.0), cm2inch(6.0)))
      #fig.suptitle(f'CollisionDetection, Property {property_id}, lg(p)={brute_val}')
      ax = fig.add_subplot(1, 1, 1, xlabel=r'$\epsilon$', ylabel=r'$\log_{10}(\mathcal{I})$', ylim=(-17,0))
      #ax.hlines(y=0, linestyle='--', xmin=xs[0], xmax=xs[-1], linewidth=0.75, color='black') #color='red',
      #ax.hlines(y=-100, color='black', xmin=xs[0], xmax=xs[-1], linestyle='--', linewidth=0.75)

      for idx in range(mean_ps.shape[0]):
        x = mean_ps[idx]
        std = std_ps[idx]

        #lower = max(x - 1e-200, x - std)
        lower = x - std
        upper = min(1.0, x + std)

        if lower < 0:
          ax.scatter(np.array([xs[idx]]), np.log(np.array([x]))/math.log(10), color=colors[2], marker='.', linewidth=0.5)
        else:
          ax.plot(np.array([xs[idx], xs[idx]]), np.log(np.array([lower, upper]))/math.log(10), color=colors[0], linewidth=0.5)
          #ax.scatter(np.array([xs[idx]]), np.log(np.array([x])), color='seagreen', marker='.', linewidth=0.5)

      ax.plot(xs, np.log(mean_ps)/math.log(10), linestyle='-', marker='.', color=colors[0], label=r'$M={}$'.format(count_mh_steps))

      #for idx in range(len(lg_ps)):
      #  ax.scatter(xs[idx]*np.ones_like(lg_ps[idx]), lg_ps[idx] - 10, linewidth=0.5, marker='.', color='purple')
        
      #ax.plot(xs, np.log(mean_ps), color=colors[0], linewidth=1.0)
      #print(naive_lg_ps)
      #print(naive_counts)

      ax.plot(xs, np.array(naive_lg_ps)/math.log(10), color='orange', linewidth=2.5, linestyle='-', marker='o', zorder=0, label='naive MC')

      #ax.legend(loc='lower right')
      ax.legend()
      ax.xaxis.set_tick_params(width=0.5)
      ax.yaxis.set_tick_params(width=0.5)
      ax.spines['left'].set_linewidth(0.5)
      ax.spines['bottom'].set_linewidth(0.5)
      sns.despine()

      fig.savefig(f'cifar10_exp1_sample_{sample_id}_mhsteps_{count_mh_steps}.svg', bbox_inches='tight')
      plt.close(fig)

    #raise Exception()

def plot_exp_errors():
  results = pickle.load(open('./results/cifar10/extracted_exp_results.pickle', 'rb'))

  for sample_id in range(1):
    mean_ps = {}
    std_ps = {}
    fig = plt.figure(figsize=(cm2inch(8.0), cm2inch(6.0)))
    ax = fig.add_subplot(1, 1, 1, xlabel=r'$\epsilon$', ylabel=r'$\Delta\log_{10}(\mathcal{I})$') #, ylim=(-50,0))

    for count_mh_steps in [2000, 100, 250, 500, 1000]:
      lg_ps = results['lg_ps'][(sample_id, count_mh_steps)]
      naive_lg_ps = results['naive_lg_ps'][(sample_id, count_mh_steps)]
      
      #naive_lg_ps = np.array(summary['lg_ps'])[ids]
      #xs = np.array([0.75, 0.875, 1.0, 1.125, 1.250, 1.375, 1.5])
      xs = np.array([0.0366667, 0.04, 0.0433333, 0.0466667, 0.05, 0.055, 0.06, 0.07, 0.08])

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
          color = 'y'
        elif count_mh_steps == 1000:
          color = 'darkolivegreen'

        for idx in range(mean_ps[count_mh_steps].shape[0]):
          x = mean_ps[count_mh_steps][idx]
          std = std_ps[count_mh_steps][idx]

          #lower = max(x - 1e-200, x - std)
          lower = x - std
          upper = min(1.0, x + std)

          #if lower < 0:
          #  ax.scatter(np.array([xs[idx]]), np.log(np.array([x])), color=colors[2], marker='.', linewidth=0.5)
          #else:
          ax.plot(np.array([xs[idx], xs[idx]]), (np.log(np.array([mean_ps[2000][idx]])) - np.log(np.array([lower, upper])))/math.log(10), color=color, linewidth=0.5)
          #ax.scatter(np.array([xs[idx]]), np.log(np.array([x])), color=colors[0], marker='.', linewidth=0.5)

        ax.plot(xs, (np.log(mean_ps[2000]) - np.log(mean_ps[count_mh_steps]))/math.log(10), color=color, marker='.', linestyle='-', label=r'$M={}$'.format(count_mh_steps))

    ax.legend()
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    sns.despine()

    fig.savefig(f'cifar10_exp1_sample_{sample_id}_errors.svg', bbox_inches='tight')
    plt.close(fig)

    #raise Exception()

plot_exp_results()
plot_exp_errors()
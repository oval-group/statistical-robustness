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
matplotlib.rc('font', family='Latin Modern Roman')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def cm2inch(value):
  return value/2.54

viridis = plt.get_cmap('viridis').colors
viridis = [viridis[i] for i in [100, 240, 150, 0]]
tab20b = plt.get_cmap('tab20b').colors
colors = [None for _ in range(4)]
colors[0] = viridis[0]
colors[1] = plt.get_cmap('Paired').colors[3]
colors[2] = tab20b[13]
colors[3] = (0.3, 0.3, 0.3)

"""fig = plt.figure(figsize=(cm2inch(40), cm2inch(25)))
brute_val = round(summary['lg_ps'][property_id], 2)
fig.suptitle(f'CollisionDetection, Property {property_id}, lg(p)={brute_val}')
ax = fig.add_subplot(3, 4, idx+1, title=f'N={count_particles}', xlabel='rho', ylabel='lg(p)', ylim=lg_ylim)
ax.hlines(y=summary['lg_ps'][property_id], color='red', xmin=-10, xmax=10, linestyle='--', linewidth=1.)
ax.scatter(np.arange(1, 7), np.log(np.mean(np.exp(np.array(lg_p)), axis=1)))
ax.set_xticklabels(('0.9', '0.75', '0.5', '0.25', '0.1', '0.01'))
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
sns.despine()
fig.savefig(f'collision_exp_maxval.pdf', bbox_inches='tight')
plt.close(fig)"""

# Open the summary because we want to pick out certain properties!
summary = pickle.load(open(f'./data/collisionDetection/summary.pickle', "rb" ))

sat_idx = summary['SATs'] == True
unsat_idx = summary['SATs'] == False

# DEBUG: See which properties are misclassified by Planet as UNSAT
misverified = 0
misverified2 = 0
for idx in range(500):
  if summary['SATs'][idx] == False and summary['max_vals'][idx] >= 0: #mean_max_vals[idx] > 0:
    misverified += 1
    #print(idx)
  if summary['SATs'][idx] == True and summary['max_vals'][idx] < 0: #mean_max_vals[idx] > 0:
    misverified2 += 1
print(f'{misverified} UNSAT properties incorrectly verified as SAT (naive MC found incorrect counterexample)')
print(f'{misverified2} SAT properties incorrectly verified as UNSAT (naive MC failed to find counterexample)')

#print(f'{sat_idx.sum()} SAT, {unsat_idx.sum()} UNSAT')

## Produce the maximum value found for each property
def extract_exp1_results():
  max_vals = []
  lg_ps = []
  naive_max_vals = []
  naive_lg_ps = []
  for property_id in range(500):
    print(f'property {property_id}')
    count_particles = 100000
    count_mh_steps = 100
    rho = 0.1
    result = pickle.load(open(f'./results/collisionDetection/property_{property_id}_count_particles_{count_particles}_count_mh_steps_{count_mh_steps}_rho_{rho}.pickle', 'rb'))
    max_vals.append(np.array(result['max_vals']))
    lg_ps.append(np.array(result['lg_ps']))
    naive_max_vals.append(summary['max_vals'][property_id])
    naive_lg_ps.append(summary['lg_ps'][property_id])
  
  with open(f'./results/collisionDetection/extracted_exp1_results.pickle', 'wb') as handle:
    pickle.dump({'lg_ps':lg_ps, 'max_vals': max_vals, 'naive_lg_ps': naive_lg_ps, 'naive_max_vals': naive_max_vals}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def extract_exp2_results():
  results = {}
  for rho in [0.1, 0.25, 0.5]:
    for count_particles in [1000, 10000, 100000]:
      for count_mh_steps in [100, 250, 1000]:
        ids = []
        lg_ps = []

        print(f'rho {rho}, count_particles {count_particles}, count_mh_steps {count_mh_steps}')
        for property_id in range(500):
          # Skip UNSAT properties
          #print(f'property {property_id}')
          if summary['SATs'][property_id] == False:
            continue

          # Extract results from SAT properties
          ids.append(property_id)
          result = pickle.load(open(f'./results/collisionDetection/property_{property_id}_count_particles_{count_particles}_count_mh_steps_{count_mh_steps}_rho_{rho}.pickle', 'rb'))
          lg_ps.append(np.array(result['lg_ps']))

        results[(rho, count_particles, count_mh_steps)] = {'lg_ps':lg_ps, 'ids':ids}

  with open(f'./results/collisionDetection/extracted_exp2_results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

#extract_exp1_results()  
#extract_exp2_results()

def plot_exp1_results():
  results = pickle.load(open('./results/collisionDetection/extracted_exp1_results.pickle', 'rb'))
  # 'lg_ps':lg_ps, 'max_vals': max_vals, 'naive_lg_ps': naive_lg_ps, 'naive_max_vals': naive_max_vals

  lg_ps = np.array(results['lg_ps'])
  max_vals = np.mean(np.array(results['max_vals']), axis=1)
  mean_ps = np.zeros_like(np.mean(lg_ps, axis=1))
  std_ps = np.zeros_like(np.mean(lg_ps, axis=1))
  for idx in range(lg_ps.shape[0]):
    lg_ps[idx][lg_ps[idx] < -50.] = -50.
    p = np.exp(lg_ps[idx])
    #p = p[p > 0]
    #if p.shape[0] < 30:
    #  print(p.shape[0])
    mean_ps[idx] = np.mean(p)
    std_ps[idx] = 3. * np.std(p)/math.sqrt(p.shape[0])

  #mean_max_vals = np.mean(np.array(results['max_vals']), axis=1)
  sort_idx = np.argsort(mean_ps)
  sat_idx = np.array(summary['SATs'])[sort_idx] == True
  unsat_idx = np.array(summary['SATs'])[sort_idx] == False
  mean_lg_ps = np.log(mean_ps)

  results['naive_lg_ps'] = np.array(results['naive_lg_ps'])
  results['naive_lg_ps'][results['naive_lg_ps'] < -50.] = -50.

  # DEBUG: See which properties are misclassified by Planet as UNSAT
  #misverified = 0
  #for idx in range(500):
  #  if summary['SATs'][idx] == False and mean_max_vals[idx] > 0:  #results['naive_max_vals'][idx] > 0: #mean_max_vals[idx] > 0:
  #    misverified += 1
  #    print(idx)
  #print(f'{misverified} SAT properties incorrectly verified as UNSAT (formal verification failed to find a counterexample)')
  #print(f'{sat_idx.sum()} SAT, {unsat_idx.sum()} UNSAT')

  # olivedrab, firebrick
  # lightslategrey, goldenrod
  
  fig = plt.figure(figsize=(cm2inch(20), cm2inch(10)))
  ax = fig.add_subplot(1, 1, 1, xlabel='property', ylabel=r'$\log_{10}\mathcal{I}$', ylim=(-25., 0.))
  #ax.hlines(y=0, linestyle='--', xmin=0, xmax=unsat_idx.sum(), linewidth=0.75, color='black', zorder=0) #color='red',
  ax.scatter(np.arange(unsat_idx.sum()), mean_lg_ps[sort_idx][unsat_idx]/math.log(10), color='firebrick', marker='.', alpha=0.5, linewidth=3.0, label='AMLS (UNSAT)')
  ax.scatter(np.arange(unsat_idx.sum(),unsat_idx.sum()+sat_idx.sum()), mean_lg_ps[sort_idx][sat_idx]/math.log(10), color='olivedrab', marker='.', alpha=0.5, linewidth=3.0, label='AMLS (SAT)')

  ax.scatter(np.arange(unsat_idx.sum()), np.array(results['naive_lg_ps'])[sort_idx][unsat_idx]/math.log(10), color='goldenrod', marker='.', alpha=0.5, linewidth=0.01, label='naive MC (UNSAT)')
  ax.scatter(np.arange(unsat_idx.sum(),unsat_idx.sum()+sat_idx.sum()), np.array(results['naive_lg_ps'])[sort_idx][sat_idx]/math.log(10), color='royalblue', marker='.', alpha=0.5, linewidth=0.01, label='naive MC (SAT)')

  leg = ax.legend(loc='upper left')
  for lh in leg.legendHandles: 
    lh.set_alpha(1.0)

  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()

  fig.savefig(f'collision_exp1_lg_I_sat.svg', bbox_inches='tight')
  plt.close(fig)

  sort_idx = np.argsort(max_vals)
  fig = plt.figure(figsize=(cm2inch(20), cm2inch(10)))
  ax = fig.add_subplot(1, 1, 1, xlabel='property', ylabel=r'$\max s(\tilde{\mathbf{x}})$')  #, ylim=(-25., 0.))
  ax.hlines(y=0, linestyle='--', xmin=0, xmax=unsat_idx.sum()+sat_idx.sum(), linewidth=0.75, color='black', zorder=10) #color='red',
  ax.scatter(np.arange(unsat_idx.sum()), max_vals[sort_idx][unsat_idx], color='firebrick', marker='.', alpha=0.5, linewidth=3.0, label='AMLS (UNSAT)')
  ax.scatter(np.arange(unsat_idx.sum(),unsat_idx.sum()+sat_idx.sum()), max_vals[sort_idx][sat_idx], color='olivedrab', marker='.', alpha=0.5, linewidth=3.0, label='AMLS (SAT)')

  ax.scatter(np.arange(unsat_idx.sum()), np.array(results['naive_max_vals'])[sort_idx][unsat_idx], color='goldenrod', marker='.', alpha=0.5, linewidth=0.01, label='naive MC (UNSAT)')
  ax.scatter(np.arange(unsat_idx.sum(),unsat_idx.sum()+sat_idx.sum()), np.array(results['naive_max_vals'])[sort_idx][sat_idx], color='royalblue', marker='.', alpha=0.5, linewidth=0.01, label='naive MC (SAT)')

  leg = ax.legend(loc='lower right')
  for lh in leg.legendHandles: 
    lh.set_alpha(1.0)

  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()

  fig.savefig(f'collision_exp1_max_vals_sat.svg', bbox_inches='tight')
  plt.close(fig)

def plot_exp2_results():
  results = pickle.load(open('./results/collisionDetection/extracted_exp2_results.pickle', 'rb'))
  sort_idx = None
  count_particles = 10000

  fig = plt.figure(figsize=(cm2inch(7.0), cm2inch(6.0)))
  ax = fig.add_subplot(1, 1, 1, xlabel='property id', ylabel=r'$\log_{10}(\mathcal{I})$', ylim=(-20,0))
  ax.scatter([], [], marker='.', color='orange', linewidth=0.5, label='naive MC')

  for rho, count_mh_steps, color in [(0.1,1000,colors[0])]: # (0.5,100,'firebrick'), (0.1,1000,'seagreen')
    
    result = results[(rho, count_particles, count_mh_steps)]
    lg_ps = np.array(result['lg_ps'])
    ids = np.array(result['ids'])
    naive_lg_ps = np.array(summary['lg_ps'])[ids]

    naive_ps = np.exp(naive_lg_ps)
    naive_std = np.multiply(naive_ps, 1 - naive_ps) / (1.e10)

    #print(np.log(naive_ps - naive_std))
    #print(np.log(naive_ps + naive_std))
    #raise Exception()

    mean_ps = np.zeros_like(np.mean(lg_ps, axis=1))
    std_ps = np.zeros_like(np.mean(lg_ps, axis=1))
    for idx in range(lg_ps.shape[0]):
      p = np.exp(lg_ps[idx])
      p = p[p > 0]
      #if p.shape[0] < 30:
      #  print(p.shape[0])
      mean_ps[idx] = np.mean(p)
      std_ps[idx] = 3. * np.std(p)/math.sqrt(p.shape[0])

    if sort_idx is None:
      sort_idx = np.argsort(mean_ps)
      #continue
      
    for idx in range(mean_ps.shape[0]):
      x = mean_ps[sort_idx][idx]
      std = std_ps[sort_idx][idx]
      lower = x - std
      upper = min(1.0, x + std)

      if lower < 0:
        ax.scatter(np.array([idx]), np.log(np.array([x]))/math.log(10), color='grey', marker='.', linewidth=0.5)
      else:
        if idx != 50:
          ax.plot(np.array([idx, idx]), np.log(np.array([lower, upper]))/math.log(10), color=color, linewidth=0.5)
          ax.scatter(np.array([idx]), np.log(np.array([x]))/math.log(10), color=color, marker='.', linewidth=0.5)
        else:
          ax.plot(np.array([idx, idx]), np.log(np.array([lower, upper]))/math.log(10), color=color, linewidth=0.5)
          ax.scatter(np.array([idx]), np.log(np.array([x]))/math.log(10), color=color, marker='.', linewidth=0.5, label=r'$M={},\rho={}$'.format(count_mh_steps,rho))
    
  naive_lg_ps_idx = naive_lg_ps[sort_idx] != -np.inf
  this_naive_lg_ps = naive_lg_ps[sort_idx][naive_lg_ps_idx]
  ax.plot(np.arange(mean_ps.shape[0] - this_naive_lg_ps[1:].shape[0], mean_ps.shape[0]), this_naive_lg_ps[1:]/math.log(10), color='orange')

  ax.legend(loc='lower right')
  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()

  fig.savefig(f'collision_exp2_contrasting_results.svg', bbox_inches='tight')
  plt.close(fig)

def plot_exp2_error_rho():
  results = pickle.load(open('./results/collisionDetection/extracted_exp2_results.pickle', 'rb'))
  sort_idx = None
  count_particles = 10000
  mean_ps = {}
  std_ps = {}
  naive_ps = {}
  for rho in [0.1, 0.25, 0.5]:
    for count_mh_steps in [1000, 250, 100]:
      #print(results.keys())
      result = results[(rho, count_particles, count_mh_steps)]
      lg_ps = np.array(result['lg_ps'])
      ids = np.array(result['ids'])
      naive_lg_ps = np.array(summary['lg_ps'])[ids]
      naive_ps[(rho, count_mh_steps)] = np.exp(naive_lg_ps)

      mean_ps[(rho, count_mh_steps)] = np.zeros_like(np.mean(lg_ps, axis=1))
      std_ps[(rho, count_mh_steps)] = np.zeros_like(np.mean(lg_ps, axis=1))
      for idx in range(lg_ps.shape[0]):
        lg_ps[idx][lg_ps[idx] < -50.] = -50.
        p = np.exp(lg_ps[idx])
        #p = p[p > 0]
        #if p.shape[0] < 30:
        #  print(p.shape[0])
        mean_ps[(rho, count_mh_steps)][idx] = np.mean(p)
        std_ps[(rho, count_mh_steps)][idx] = 3. * np.std(p)/math.sqrt(p.shape[0])

      if sort_idx is None:
        sort_idx = np.argsort(mean_ps[(rho, count_mh_steps)])
        shift_idx = (naive_lg_ps[sort_idx] <= -15.).sum()
        sort_idx = sort_idx[naive_lg_ps[sort_idx] > -15.]
      
  fig = plt.figure(figsize=(cm2inch(7.0), cm2inch(6.0)))
  #fig.suptitle(f'CollisionDetection, Property {property_id}, lg(p)={brute_val}')
  ax = fig.add_subplot(1, 1, 1, xlabel='property id', ylabel=r'$\Delta\log_{10}\mathcal{I}$', title=r'$M=1000$')  #, ylim=(-40,0))
  ax.scatter(np.arange(shift_idx, sort_idx.shape[0]+shift_idx), (np.log(naive_ps[(0.1, 1000)]) - np.log(mean_ps[(0.5, 1000)]))[sort_idx]/math.log(10), color='firebrick', marker='.', linewidth=0.5, label=r'$\rho=0.5$')
  ax.scatter(np.arange(shift_idx, sort_idx.shape[0]+shift_idx), (np.log(naive_ps[(0.1, 1000)]) - np.log(mean_ps[(0.25, 1000)]))[sort_idx]/math.log(10), color='goldenrod', marker='.', linewidth=0.5, label=r'$\rho=0.25$')
  ax.scatter(np.arange(shift_idx, sort_idx.shape[0]+shift_idx), (np.log(naive_ps[(0.1, 1000)]) - np.log(mean_ps[(0.1, 1000)]))[sort_idx]/math.log(10), color='seagreen', marker='.', linewidth=0.5, label=r'$\rho=0.1$')

  for rho in [0.5, 0.25, 0.1]:
    if rho == 0.5:
      color = 'firebrick'
    elif rho == 0.25:
      color = 'goldenrod'
    else:
      color = 'seagreen'

    for idx in range(sort_idx.shape[0]):
      x = mean_ps[(rho, 1000)][sort_idx][idx]
      std = std_ps[(rho, 1000)][sort_idx][idx]

      #lower = max(x - 1e-200, x - std)
      lower = x - std
      upper = min(1.0, x + std)

      if lower < 0:
        pass
        #print(mean_ps[sort_idx][idx])
        #print(np.exp(lg_ps[sort_idx]))
        #print(lg_ps[sort_idx[idx]])
        #print(x, std)
        #print(lower)
        #print('')
        #ax.scatter(np.array([idx]), np.log(np.array([x])), color=colors[2], marker='.', linewidth=0.5)
      else:
        ax.plot(np.array([shift_idx+idx, shift_idx+idx]), (np.log(np.array([mean_ps[(0.1, 1000)][sort_idx][idx], mean_ps[(0.1, 1000)][sort_idx][idx]]))-np.log(np.array([lower, upper])))/math.log(10), color=color, linewidth=0.5)
        #ax.scatter(np.array([idx]), np.log(np.array([x])), color=colors[0], marker='.', linewidth=0.5)

  leg = ax.legend(loc='upper right')
  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()
  fig.savefig(f'collision_exp2_varying_rho_.svg', bbox_inches='tight')
  plt.close(fig)

def plot_exp2_error_M():
  results = pickle.load(open('./results/collisionDetection/extracted_exp2_results.pickle', 'rb'))
  sort_idx = None
  count_particles = 10000
  mean_ps = {}
  std_ps = {}
  naive_ps = {}
  for rho in [0.1, 0.25, 0.5]:
    for count_mh_steps in [1000, 250, 100]:
      #print(results.keys())
      result = results[(rho, count_particles, count_mh_steps)]
      lg_ps = np.array(result['lg_ps'])
      ids = np.array(result['ids'])
      naive_lg_ps = np.array(summary['lg_ps'])[ids]
      naive_ps[(rho, count_mh_steps)] = np.exp(naive_lg_ps)

      mean_ps[(rho, count_mh_steps)] = np.zeros_like(np.mean(lg_ps, axis=1))
      std_ps[(rho, count_mh_steps)] = np.zeros_like(np.mean(lg_ps, axis=1))
      for idx in range(lg_ps.shape[0]):
        lg_ps[idx][lg_ps[idx] < -50.] = -50.
        p = np.exp(lg_ps[idx])
        #p = p[p > 0]
        #if p.shape[0] < 30:
        #  print(p.shape[0])
        mean_ps[(rho, count_mh_steps)][idx] = np.mean(p)
        std_ps[(rho, count_mh_steps)][idx] = 3. * np.std(p)/math.sqrt(p.shape[0])

      if sort_idx is None:
        sort_idx = np.argsort(mean_ps[(rho, count_mh_steps)])
        shift_idx = (naive_lg_ps[sort_idx] <= -15.).sum()
        sort_idx = sort_idx[naive_lg_ps[sort_idx] > -15.]
      
  rho = 0.1
  fig = plt.figure(figsize=(cm2inch(7.0), cm2inch(6.0)))
  #fig.suptitle(f'CollisionDetection, Property {property_id}, lg(p)={brute_val}')
  ax = fig.add_subplot(1, 1, 1, xlabel='property id', ylabel=r'$\Delta\log_{10}\mathcal{I}$', title=r'$\rho={}$'.format(rho))  #, ylim=(-0.4,0.2))
  ax.scatter(np.arange(shift_idx, sort_idx.shape[0]+shift_idx), ((np.log(naive_ps[(rho, 100)]) - np.log(mean_ps[(rho, 100)]))[sort_idx])/math.log(10), color='firebrick', marker='.', linewidth=0.5, label=r'$M=100$')
  ax.scatter(np.arange(shift_idx, sort_idx.shape[0]+shift_idx), ((np.log(naive_ps[(rho, 100)]) - np.log(mean_ps[(rho, 250)]))[sort_idx])/math.log(10), color='goldenrod', marker='.', linewidth=0.5, label=r'$M=250$')
  ax.scatter(np.arange(shift_idx, sort_idx.shape[0]+shift_idx), ((np.log(naive_ps[(rho, 100)]) - np.log(mean_ps[(rho, 1000)]))[sort_idx])/math.log(10), color='seagreen', marker='.', linewidth=0.5, label=r'$M=1000$')

  for M in [100, 250, 1000]:
    if M == 100:
      color = 'firebrick'
    elif M == 250:
      color = 'goldenrod'
    else:
      color = 'seagreen'

    for idx in range(sort_idx.shape[0]):
      x = mean_ps[(rho, M)][sort_idx][idx]
      std = std_ps[(rho, M)][sort_idx][idx]

      #lower = max(x - 1e-200, x - std)
      lower = x - std
      upper = min(1.0, x + std)

      if lower < 0:
        pass
        #print(mean_ps[sort_idx][idx])
        #print(np.exp(lg_ps[sort_idx]))
        #print(lg_ps[sort_idx[idx]])
        #print(x, std)
        #print(lower)
        #print('')
        #ax.scatter(np.array([idx]), np.log(np.array([x])), color=colors[2], marker='.', linewidth=0.5)
      else:
        ax.plot(np.array([idx+shift_idx, idx+shift_idx]), (np.log(np.array([naive_ps[(rho, 100)][sort_idx][idx], naive_ps[(rho, 100)][sort_idx][idx]]))-np.log(np.array([lower, upper])))/math.log(10), color=color, linewidth=0.5)
        #ax.scatter(np.array([idx]), np.log(np.array([x])), color=colors[0], marker='.', linewidth=0.5)

  leg = ax.legend(loc='lower right')
  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()
  fig.savefig(f'collision_exp2_varying_M_rho={rho}.svg', bbox_inches='tight')
  plt.close(fig)

plot_exp1_results()
plot_exp2_error_rho()
plot_exp2_error_M()
plot_exp2_results()

#p_means = (np.mean(np.exp(np.array(lg_p)), axis=1))
#p_std = (np.std(np.exp(np.array(lg_p)), axis=1))
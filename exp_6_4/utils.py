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

import os, uuid, logging, math
import torch
import numpy as np

# Orders vertices so they go clockwise or anti-clockwise around polygon
def order_vertices(vertices):
  center = vertices.mean(axis=0)
  angles = np.arctan2(vertices[:,1] - center[1], vertices[:,0] - center[0])
  idx = np.argsort(angles)
  return vertices[idx]

# Calculates area of ordered vertices of polygon
def polygon_area(vertices):
  total_area = 0
  for idx in range(0,vertices.shape[0]-1):
    total_area += vertices[idx,0]*vertices[idx+1,1] - vertices[idx+1,0]*vertices[idx,1]
  total_area += vertices[-1,0]*vertices[0,1] - vertices[0,0]*vertices[-1,1]
  return 0.5 * abs(total_area)

def isnan(x):
    return x != x

def unique_name():
	return uuid.uuid4().hex[:6]

def make_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def sum_i_neq_j(x):
	"""Sum over all elements except i-th for all i (used in VIMCO calculation)

	Input:
		x: Variable of size `iw_size` x `batch_size`
	
	Output:
		result: Of size, `iw_size` x `batch_size` (i,j)th element is equal to sum_{k neq i} x_{k,j}
	"""
	iw_size = x.size(0)
	batch_size = x.size(1)

	# TODO: Would torch.expand instead of torch.repeat make this faster?
	inv_mask = (1. - torch.eye(iw_size)
				).unsqueeze(dim=2).repeat(1, 1, batch_size)
	x_masked = torch.mul(x.view(1, iw_size, batch_size), inv_mask)
	return torch.sum(x_masked, dim=1)

def ln_sum_i_neq_j(x):
	"""Sum over all elements except i-th for all i in log-space (used in VIMCO calculation)

	Input:
		x: Variable of size `iw_size` x `batch_size`
	
	Output:
		result: Of size, `iw_size` x `batch_size` (i,j)th element is equal to sum_{k neq i} x_{k,j} in log-space
	"""
	iw_size = x.size(0)
	batch_size = x.size(1)

	# TODO: Would torch.expand instead of torch.repeat make this faster?
	inv_mask = torch.eye(iw_size).unsqueeze(dim=2).repeat(1, 1, batch_size)
	x_masked = x.view(1, iw_size, batch_size) - inv_mask*1000000.0
	return logsumexp(x_masked, dim=1)

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def numify(x):
	return np.round(x.cpu().detach().numpy(), decimals=3)

def numify2(x):
	return x.cpu().detach().numpy()

def stats(v):
  print('min', torch.min(v).cpu().detach().numpy(), 'max', torch.max(v).cpu().detach().numpy(), 'mean', torch.mean(v).cpu().detach().numpy(), 'NaNs', torch.sum(isnan(v)).cpu().detach().numpy(), '-Inf', torch.sum(v==float("-Inf")).cpu().detach().numpy(), '+Inf', torch.sum(v==float("Inf")).cpu().detach().numpy() )
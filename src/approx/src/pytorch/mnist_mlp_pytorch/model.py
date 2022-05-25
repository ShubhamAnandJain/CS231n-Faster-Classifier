'''
MLP model for MNIST (adapted from https://github.com/lancopku/meProp/tree/master/src/pytorch)
'''

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from approx_mul_pytorch import approx_Linear
sample_ratio=0.9
minimal_k = 10
sample_ratio_bwd= sample_ratio#None
minimal_k_bwd = 10
sample_ratio_wu= sample_ratio#None
minimal_k_wu = 10



class MLP(nn.Module):
    '''
    A complete network(MLP) for MNSIT classification.
    
    Input feature is 28*28=784
    Output feature is 10
    Hidden features are of hidden size
    
    Activation is ReLU
    '''

    def __init__(self, hidden, layer, dropout=None):
        super(MLP, self).__init__()
        self.layer = layer
        self.dropout = dropout
        self.model = nn.Sequential(self._create(hidden, layer, dropout))

    def _create(self, hidden, layer, dropout=None):
        if layer == 1:
            d = OrderedDict()
            d['linear0'] = approx_Linear(784, 10, sample_ratio=sample_ratio, minimal_k=minimal_k, sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd = minimal_k_bwd, sample_ratio_wu = sample_ratio_wu, minimal_k_wu = minimal_k_wu)
            return d
        d = OrderedDict()
        for i in range(layer):
            if i == 0:
                d['linear' + str(i)] = approx_Linear(784, hidden, sample_ratio=sample_ratio, minimal_k=minimal_k, sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd = minimal_k_bwd, sample_ratio_wu = sample_ratio_wu, minimal_k_wu = minimal_k_wu)
                d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
            elif i == layer - 1:
                d['linear' + str(i)] = approx_Linear(hidden, 10, sample_ratio=sample_ratio, minimal_k=minimal_k, sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd = minimal_k_bwd, sample_ratio_wu = sample_ratio_wu, minimal_k_wu = minimal_k_wu)
            else:
                d['linear' + str(i)] = approx_Linear(hidden, hidden, sample_ratio=sample_ratio, minimal_k=minimal_k, sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd = minimal_k_bwd, sample_ratio_wu = sample_ratio_wu, minimal_k_wu = minimal_k_wu)
                d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
        return d

    def forward(self, x):
        return F.log_softmax(self.model(x.view(-1, 784)), dim=0)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, type(approx_Linear)):
                m.reset_parameters()

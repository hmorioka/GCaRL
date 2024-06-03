"""GCaRL"""


import torch
import torch.nn as nn
from itertools import combinations
from torchvision.models import resnet18
from subfunc.showdata import *


# =============================================================
# =============================================================
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        m, _ = torch.max(x.reshape([*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size]), dim=-1)
        return m


# =============================================================
# =============================================================
class Net(nn.Module):
    def __init__(self, num_group, num_hdim, num_hidden=100, phi_type='gauss-maxout', hz_sizes=None, hp_sizes=None, phi_share=False, pool_size=2):
        """ Network model
         Args:
             num_group: number of groups
             num_hdim: number of latent variable
             num_hidden: number of nodes at the top of resnet
             phi_type: model type of phi ('maxout')
             hz_sizes: number of nodes for each layer (hz)
             hp_sizes: number of nodes for each layer (hp)
             phi_share: share phi across group-pairs or not
             pool_size: pool size of max-out
        """
        super(Net, self).__init__()

        self.num_group = num_group
        self.num_hdim = num_hdim
        self.num_hidden = num_hidden
        self.group_combs = list(combinations(np.arange(num_group), 2))
        self.num_comb = len(self.group_combs)
        self.maxout = Maxout(pool_size)
        self.phi_type = phi_type
        self.phi_share = phi_share

        # h
        h = torch.nn.Sequential(
            resnet18(num_classes=self.num_hidden),
            torch.nn.BatchNorm1d(num_features=self.num_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.num_hidden, self.num_hdim),
            torch.nn.BatchNorm1d(num_features=self.num_hdim)
        )
        self.h = h
        # hz
        if len(hz_sizes) > 0:
            hz = []
            bnz = []
            for m in range(num_group):
                num_xdim_hz = self.num_hdim * 2 if self.phi_type in {'lap-mlp', 'lap-tanh'} else self.num_hdim
                if len(hz_sizes) > 1:
                    hm = [nn.Linear(num_xdim_hz, hz_sizes[0] * pool_size)]
                    hm = hm + [nn.Linear(hz_sizes[k - 1], hz_sizes[k] * pool_size) for k in range(1, len(hz_sizes) - 1)]
                    hm.append(nn.Linear(hz_sizes[-2], hz_sizes[-1], bias=False))
                else:
                    hm = [nn.Linear(num_xdim_hz, hz_sizes[0], bias=False)]
                hz.append(nn.ModuleList(hm))
                bnz.append(nn.BatchNorm1d(num_features=self.num_hdim))
            #
            self.hz = nn.ModuleList(hz)
            self.bnz = nn.ModuleList(bnz)
        else:
            self.hz = []
            self.bnz = nn.BatchNorm1d(num_features=self.num_hdim)
        # phi
        if self.phi_type == 'gauss-maxout':
            self.w = nn.Parameter(torch.zeros([self.num_hdim, self.num_hdim, 2, self.num_comb]))
            self.zw = nn.Parameter(torch.zeros([num_group, self.num_hdim, 2]))
            if phi_share:
                self.pw = nn.Parameter(torch.ones([1, 2]))
                self.pb = nn.Parameter(torch.zeros([1]))
            else:
                self.pw = nn.Parameter(torch.ones([self.num_comb, 2]))
                self.pb = nn.Parameter(torch.zeros([self.num_comb]))
        else:
            raise ValueError
        self.b = nn.Parameter(torch.zeros([1]))

        # initialize
        for m in range(num_group):
            for k in range(len(self.hz[m])):
                torch.nn.init.xavier_uniform_(self.hz[m][k].weight)

    def forward(self, x, calc_logit=True):
        """ forward
         Args:
             x: input [batch, group, dim]
             calc_logit: obtain logits additionally to h, or not
         """
        batch_size, num_group, num_ch, num_x, num_y = x.size()

        # h
        h_bn = self.h(x.reshape([-1, num_ch, num_x, num_y])).reshape([batch_size, num_group, -1])

        if calc_logit:
            if self.phi_type == 'gauss-maxout':
                logits = torch.zeros(batch_size, device=x.device)
                h_nonlin, _ = torch.max(self.pw[None, None, None, :, :] * (h_bn[:, :, :, None, None] - self.pb[None, None, None, :, None]), dim=-1)
                for c in range(len(self.group_combs)):
                    a = self.group_combs[c][0]
                    b = self.group_combs[c][1]
                    if self.phi_share:
                        phi_ab = h_bn[:, b, None, :] * h_nonlin[:, a, :, 0, None]
                        phi_ba = h_bn[:, a, None, :] * h_nonlin[:, b, :, 0, None]
                    else:
                        phi_ab = h_bn[:, b, None, :] * h_nonlin[:, a, :, c, None]
                        phi_ba = h_bn[:, a, None, :] * h_nonlin[:, b, :, c, None]
                    logits = logits + torch.sum(self.w[None, :, :, 0, c] * phi_ab + self.w[None, :, :, 1, c] * phi_ba, dim=[1, 2])
                # hz
                hz_bn = torch.zeros_like(h_bn)
                for m in range(num_group):
                    hzm = h_bn[:, m, :]
                    for k in range(len(self.hz[m])):
                        hzm = self.hz[m][k](hzm)
                        if k != len(self.hz[m]) - 1:
                            hzm = self.maxout(hzm)
                    hz_bn[:, m, :] = self.bnz[m](hzm)
                logits_z = torch.sum(self.zw[None, :, :, 0] * hz_bn ** 2 + self.zw[None, :, :, 1] * hz_bn, dim=[1, 2])

            logits = logits + logits_z + self.b

        else:
            logits = None

        return logits, h_bn, h_nonlin


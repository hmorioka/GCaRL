"""GCaRL"""


import torch
import torch.nn as nn
from itertools import combinations
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
    def __init__(self, h_sizes, num_group, num_xdim, phi_type='gauss-maxout', hz_sizes=None, hp_sizes=None, phi_share=True, pool_size=2):
        """ Network model
        Args:
             h_sizes: number of nodes for each layer (excluding the input layer)
             num_group: number of groups
             num_xdim: dimension of input
             phi_type: model type of phi ('maxout')
             hz_sizes: number of nodes for each layer (hz)
             hp_sizes: number of nodes for each layer (hp)
             phi_share: share phi across group-pairs or not
             pool_size: pool size of max-out
        """
        super(Net, self).__init__()

        self.num_group = num_group
        self.num_xdim = num_xdim
        self.num_hdim = h_sizes[-1] if len(h_sizes) > 0 else self.num_xdim[0]
        self.group_combs = list(combinations(np.arange(num_group), 2))
        self.num_comb = len(self.group_combs)
        self.maxout = Maxout(pool_size)
        self.phi_type = phi_type
        self.phi_share = phi_share
        if hz_sizes is None:
            hz_sizes = h_sizes.copy()
        # h
        if len(h_sizes) > 0:
            h = []
            bn = []
            for m in range(num_group):
                if len(h_sizes) > 1:
                    hm = [nn.Linear(self.num_xdim[m], h_sizes[0] * pool_size)]
                    hm = hm + [nn.Linear(h_sizes[k - 1], h_sizes[k] * pool_size) for k in range(1, len(h_sizes) - 1)]
                    hm.append(nn.Linear(h_sizes[-2], h_sizes[-1], bias=False))
                else:
                    hm = [nn.Linear(self.num_xdim[m], h_sizes[0], bias=False)]
                h.append(nn.ModuleList(hm))
                bn.append(nn.BatchNorm1d(num_features=self.num_hdim))
            #
            self.h = nn.ModuleList(h)
            self.bn = nn.ModuleList(bn)
        else:
            self.h = [[] for i in np.arange(num_group)]
            self.bn = nn.ModuleList([nn.BatchNorm1d(num_features=self.num_hdim) for i in np.arange(num_group)])
        # hz
        if len(hz_sizes) > 0:
            hz = []
            bnz = []
            for m in range(num_group):
                num_xdim_hz = self.num_xdim[m] * 2 if self.phi_type in {'gauss-mlp', 'lap-mlp', 'lap-tanh'} else self.num_xdim[m]
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
        elif self.phi_type == 'gauss-mlp':
            self.w = nn.Parameter(torch.zeros([self.num_hdim, self.num_hdim, 2, self.num_comb]))
            self.zw = nn.Parameter(torch.zeros([num_group, self.num_hdim, 2]))
            if phi_share:
                hp = [nn.Linear(1, hp_sizes[0])]
                hp = hp + [nn.Linear(hp_sizes[k - 1], hp_sizes[k]) for k in range(1, len(hp_sizes) - 1)]
                hp.append(nn.Linear(hp_sizes[-2], hp_sizes[-1]))
                bnp = nn.BatchNorm1d(num_features=1)
                self.hp = nn.ModuleList(hp)
                self.bnp = bnp
            else:
                hp = []
                bnp = []
                for c in range(self.num_comb):
                    if len(hz_sizes) > 1:
                        hc = [nn.Linear(1, hp_sizes[0])]
                        hc = hc + [nn.Linear(hp_sizes[k - 1], hp_sizes[k]) for k in range(1, len(hp_sizes) - 1)]
                        hc.append(nn.Linear(hp_sizes[-2], hp_sizes[-1]))
                    else:
                        hc = [nn.Linear(1, hp_sizes[0])]
                    hp.append(nn.ModuleList(hc))
                    bnp.append(nn.BatchNorm1d(num_features=1))
                self.hp = nn.ModuleList(hp)
                self.bnp = nn.ModuleList(bnp)
        elif self.phi_type == 'lap-mlp':
            self.w = nn.Parameter(torch.zeros([self.num_hdim, self.num_hdim, 2, self.num_comb]))
            self.w2 = nn.Parameter(torch.zeros([self.num_hdim, self.num_hdim, self.num_comb]))
            self.b2 = nn.Parameter(torch.zeros([self.num_hdim, self.num_hdim, 2, self.num_comb]))
            self.zw = nn.Parameter(torch.zeros([num_group, self.num_hdim, 2]))
            if phi_share:
                hp = [nn.Linear(1, hp_sizes[0])]
                hp = hp + [nn.Linear(hp_sizes[k - 1], hp_sizes[k]) for k in range(1, len(hp_sizes) - 1)]
                hp.append(nn.Linear(hp_sizes[-2], hp_sizes[-1]))
                bnp = nn.BatchNorm1d(num_features=1)
                self.hp = nn.ModuleList(hp)
                self.bnp = bnp
            else:
                hp = []
                bnp = []
                for c in range(self.num_comb):
                    if len(hz_sizes) > 1:
                        hc = [nn.Linear(1, hp_sizes[0])]
                        hc = hc + [nn.Linear(hp_sizes[k - 1], hp_sizes[k]) for k in range(1, len(hp_sizes) - 1)]
                        hc.append(nn.Linear(hp_sizes[-2], hp_sizes[-1]))
                    else:
                        hc = [nn.Linear(1, hp_sizes[0])]
                    hp.append(nn.ModuleList(hc))
                    bnp.append(nn.BatchNorm1d(num_features=1))
                self.hp = nn.ModuleList(hp)
                self.bnp = nn.ModuleList(bnp)
        else:
            raise ValueError
        self.b = nn.Parameter(torch.zeros([1]))

        # initialize
        for m in range(num_group):
            for k in range(len(self.h[m])):
                torch.nn.init.xavier_uniform_(self.h[m][k].weight)
        if self.phi_type == 'lap-mlp':
            if phi_share:
                for k in range(len(self.hp)):
                    torch.nn.init.constant_(self.hp[k].weight, 0)
                    torch.nn.init.uniform_(self.hp[k].bias, -0.01, 0.01)
                torch.nn.init.constant_(self.hp[-1].bias, 0)
            else:
                for g in range(num_group):
                    for k in range(len(self.hp[g])):
                        torch.nn.init.constant_(self.hp[g][k].weight, 0)
                        torch.nn.init.uniform_(self.hp[g][k].bias, -0.01, 0.01)
                    torch.nn.init.constant_(self.hp[g][-1].bias, 0)
            torch.nn.init.uniform_(self.w2, -0.1, 0.1)

    def forward(self, x, calc_logit=True):
        """ forward
         Args:
             x: input [batch, group, dim]
             calc_logit: obtain logits additionally to h, or not
         """
        batch_size, num_group, num_dim = x.size()

        # h
        h_bn = torch.zeros_like(x)
        for m in range(num_group):
            hm = x[:, m, :]
            for k in range(len(self.h[m])):
                hm = self.h[m][k](hm)
                if k != len(self.h[m]) - 1:
                    hm = self.maxout(hm)
            h_bn[:, m, :] = self.bn[m](hm)

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

            elif self.phi_type == 'gauss-mlp':
                logits = torch.zeros(batch_size, device=x.device)
                h_nonlin = torch.zeros([batch_size, 2, num_dim, self.num_comb], device=x[0].device)
                # pre-calculate phi if phi is shared
                if self.phi_share:
                    hp = h_bn.reshape([-1, 1])
                    for k in range(len(self.hp)):
                        hp = self.hp[k](hp)
                        if k != len(self.hp) - 1:
                            hp = torch.tanh(hp)
                    hp = self.bnp(hp).reshape(h_bn.shape)
                #
                for c in range(len(self.group_combs)):
                    a = self.group_combs[c][0]
                    b = self.group_combs[c][1]
                    if self.phi_share:
                        h_nonlin_c = hp[:, [a, b], :]
                    else:
                        hp = h_bn[:, [a, b], :].reshape([-1, 1])
                        for k in range(len(self.hp[c])):
                            hp = self.hp[c][k](hp)
                            if k != len(self.hp[c]) - 1:
                                hp = torch.tanh(hp)
                        hp = self.bnp[c](hp)
                        h_nonlin_c = hp.reshape([h_bn.shape[0], 2, h_bn.shape[-1]])
                    phi_ab = h_bn[:, b, None, :] * h_nonlin_c[:, 0, :, None]
                    phi_ba = h_bn[:, a, None, :] * h_nonlin_c[:, 1, :, None]
                    logits = logits + torch.mean(self.w[None, :, :, 0, c] * phi_ab + self.w[None, :, :, 1, c] * phi_ba, dim=[1, 2]) * 1e2
                    h_nonlin[:, :, :, c] = h_nonlin_c
                # hz
                hz_bn = torch.zeros_like(x)
                for m in range(num_group):
                    if self.phi_share:
                        hzm = torch.cat([h_bn[:, m, :], hp[:, m, :]], dim=1)
                    else:
                        hzm = torch.cat([h_bn[:, m, :], torch.tanh(h_bn[:, m, :])], dim=1)
                    for k in range(len(self.hz[m])):
                        hzm = self.hz[m][k](hzm)
                        if k != len(self.hz[m]) - 1:
                            hzm = self.maxout(hzm)
                    hz_bn[:, m, :] = self.bnz[m](hzm)
                logits_z = torch.sum(self.zw[None, :, :, 0] * hz_bn ** 2 + self.zw[None, :, :, 1] * hz_bn, dim=[1, 2])

            elif self.phi_type == 'lap-mlp':
                logits = torch.zeros(batch_size, device=x.device)
                h_nonlin = torch.zeros([batch_size, 2, num_dim, self.num_comb], device=x[0].device)
                # pre-calculate phi if phi is shared
                if self.phi_share:
                    hp = h_bn.reshape([-1, 1])
                    for k in range(len(self.hp)):
                        hp = self.hp[k](hp)
                        if k != len(self.hp) - 1:
                            hp = torch.tanh(hp)
                    hp = self.bnp(hp).reshape(h_bn.shape)
                #
                for c in range(len(self.group_combs)):
                    a = self.group_combs[c][0]
                    b = self.group_combs[c][1]
                    if self.phi_share:
                        h_nonlin_c = hp[:, [a, b], :]
                    else:
                        hp = h_bn[:, [a, b], :].reshape([-1, 1])
                        for k in range(len(self.hp[c])):
                            hp = self.hp[c][k](hp)
                            if k != len(self.hp[c]) - 1:
                                hp = torch.tanh(hp)
                        hp = self.bnp[c](hp)
                        h_nonlin_c = hp.reshape([h_bn.shape[0], 2, h_bn.shape[-1]])
                    phi_ab = self.w2[None, :, :, c] * h_bn[:, b, None, :] - torch.abs(self.w2[None, :, :, c]) * h_nonlin_c[:, 0, :, None]
                    phi_ba = self.w2[:, :, c].T[None, :, :] * h_bn[:, a, None, :] - torch.abs(self.w2[:, :, c].T[None, :, :]) * h_nonlin_c[:, 1, :, None]
                    logits = logits + torch.mean(self.w[None, :, :, 0, c] * torch.abs(phi_ab) + self.w[None, :, :, 1, c] * torch.abs(phi_ba), dim=[1, 2]) * 1e2
                    h_nonlin[:, :, :, c] = h_nonlin_c
                # hz
                hz_bn = torch.zeros_like(x)
                for m in range(num_group):
                    if self.phi_share:
                        hzm = torch.cat([h_bn[:, m, :], hp[:, m, :]], dim=1)
                    else:
                        hzm = torch.cat([h_bn[:, m, :], torch.tanh(h_bn[:, m, :])], dim=1)
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


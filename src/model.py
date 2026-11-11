#!/usr/bin/env python
# ----------------------------------------------------------
# CBDD - Torribera, Universitat de Barcelona
#
# File: model.py
# Author: Carlos Cruz-Marín
# ----------------------------------------------------------
# Definition of AtomicGNN, the graph neural network that predicts
# atomic logP contributions.
# ----------------------------------------------------------

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import NNConv


class RBFDist(nn.Module):
    def __init__(self, K=32, r_min=0.0, r_max=5.0):
        super().__init__()
        centers = torch.linspace(r_min, r_max, K)
        self.register_buffer('centers', centers)
        self.log_gamma = nn.Parameter(torch.tensor(2.0))
    def forward(self, d):
        gamma = torch.exp(self.log_gamma)
        diff = d.unsqueeze(-1) - self.centers.unsqueeze(0)
        return torch.exp(-gamma * diff * diff)

class FiLM(nn.Module):
    def __init__(self, emb_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 2*hidden),
            nn.SiLU(),
            nn.Linear(2*hidden, 2*hidden)
        )
        self.hidden = hidden
    def forward(self, e_emb):
        gb = self.net(e_emb)
        gamma, beta = gb.split(self.hidden, dim=-1)
        return gamma, beta

class MPNN(nn.Module):
    def __init__(self, hidden, edge_in, p_drop=0.1, aggr='mean', film=None, norm='layer'):
        super().__init__()
        self.film = film
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, 256),
            nn.SiLU(),
            nn.Linear(256, hidden * hidden)
        )
        self.conv = NNConv(hidden, hidden, self.edge_mlp, aggr=aggr)
        self.norm = nn.LayerNorm(hidden) if norm=='layer' else nn.Identity()
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, h, edge_index, edge_attr, elem_emb):
        x = self.conv(h, edge_index, edge_attr)
        x = self.norm(x)
        if self.film is not None:
            gamma, beta = self.film(elem_emb)
            x = gamma * x + beta
        x = self.act(x)
        x = self.drop(x)
        return h + x

class AtomicGNN(nn.Module):
    def __init__( self, x_num_dim, edge_dim, max_Z=100, elem_emb_dim=64, hidden=120, n_blocks=10, p_drop=0.1, n_tasks=1):
        # Default values are the hyperparameters used in the publication
        super().__init__()

        self.n_tasks = n_tasks
        self.hidden = hidden

        self.elem_emb = nn.Embedding(max_Z + 1, elem_emb_dim)

        self.inp = nn.Sequential(
            nn.Linear(x_num_dim + elem_emb_dim, hidden),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden))

        self.rbfd = RBFDist(K=32, r_min=0.0, r_max=5.0)

        self.edge_proj = nn.Sequential( nn.Linear(edge_dim - 1 + 32, edge_dim - 1 + 32), nn.SiLU(), )
        self.blocks = nn.ModuleList([ MPNN( hidden, edge_in=edge_dim - 1 + 32, p_drop=p_drop, film=FiLM(elem_emb_dim, hidden), ) for _ in range(n_blocks) ])

        self.head = nn.Sequential( nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(p_drop), nn.Linear(hidden, n_tasks))

        self.adapt_vec = nn.Embedding(max_Z + 1, hidden * n_tasks)
        self.adapt_bias = nn.Embedding(max_Z + 1, n_tasks)

    def _edge_feat(self, edge_attr):
        dist = edge_attr[:, -1]
        base = edge_attr[:, :-1]
        rbf = self.rbfd(dist)
        ef = torch.cat([base, rbf], dim=-1)
        return self.edge_proj(ef)

    def forward(self, data: Data):
        z = data.z.long()

        e = self.elem_emb(z)
        x = torch.cat([data.x, e], dim=-1)
        h = self.inp(x)

        edge_attr = self._edge_feat(data.edge_attr)

        for blk in self.blocks:
            h = blk(h, data.edge_index, edge_attr, e)

        y_shared = self.head(h)
        adapt_vec = self.adapt_vec(z).view( -1, h.size(-1), self.n_tasks)
        y_adapt = torch.einsum("nh,nht->nt", h, adapt_vec)
        y_adapt = y_adapt + self.adapt_bias(z)
        y_std = y_shared + y_adapt

        return y_std

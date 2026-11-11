#!/usr/bin/env python
# ----------------------------------------------------------
# CBDD - Torribera, Universitat de Barcelona
#
# File: training.py
# Author: Carlos Cruz-Marín
# ----------------------------------------------------------
# Training loop of AtomicGNN and per-element normalization of the
# atomic logP targets.
# ----------------------------------------------------------

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src import logger


@torch.no_grad()
def perZ_stats(dataset, max_Z=100, n_tasks=1):
    """Per-element mean and standard deviation of the atomic logP targets."""
    sums = torch.zeros(max_Z, n_tasks)
    sqs  = torch.zeros(max_Z, n_tasks)
    cnts = torch.zeros(max_Z, n_tasks)

    for d in dataset:
        z = d.z.long() - 1
        y = d.y.float()

        for t in range(n_tasks):
            sums[:, t].scatter_add_(0, z, y[:, t])
            sqs[:, t].scatter_add_(0, z, y[:, t] * y[:, t])
            cnts[:, t].scatter_add_(0, z, torch.ones_like(z, dtype=torch.float))

    mu = sums / cnts.clamp_min(1.0)
    var = (sqs / cnts.clamp_min(1.0)) - mu**2
    sd = var.clamp_min(1e-8).sqrt()
    sd[sd < 1e-6] = 1.0

    return mu, sd, cnts


def train(model, train_set, val_set, max_Z=100, epochs=60, batch_size=32, lr=2e-4, wd=1e-4, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    mu, sd, cnt = perZ_stats(train_set, max_Z=max_Z)
    mu, sd, cnt = mu.to(device), sd.to(device), cnt.to(device)

    wZ = (cnt.max(dim=0, keepdim=True).values / cnt.clamp_min(1.0)).sqrt()
    wZ = wZ.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    best = float('inf'); best_state=None; patience=15; bad=0

    for ep in range(1, epochs+1):
        model.train()
        tot_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            z = data.z.long() -1
            y = data.y.float()
            y_std = (y - mu[z]) / sd[z]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                yhat_std = model(data)
                diff = (yhat_std - y_std)
                per_node = torch.where(
                    diff.abs() < 1.0,
                    0.5 * diff * diff,
                    diff.abs() - 0.5)
                w = wZ[z]
                loss = (per_node * w).sum() / w.sum().clamp_min(1.0)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot_loss += loss.item()

        train_loss = tot_loss / max(len(train_loader), 1)
        model.eval()
        se = 0.0
        n_atoms = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)

                z = data.z.long() - 1
                y = data.y.float()

                yhat_std = model(data)
                yhat = yhat_std * sd[z] + mu[z]

                diff = yhat - y
                se += (diff * diff).sum().item()
                n_atoms += y.shape[0]

        val_rmse = (se / max(n_atoms, 1)) ** 0.5
        sch.step(val_rmse)

        print(f"Epoch {ep:03d} | TrainLoss={train_loss:.4f} | Val logP RMSE={val_rmse:.4f}")

        if val_rmse + 1e-6 < best:
            best = val_rmse
            best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.msg("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (mu.cpu(), sd.cpu())

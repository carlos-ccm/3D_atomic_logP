#!/usr/bin/env python
# ----------------------------------------------------------
# CBDD - Torribera, Universitat de Barcelona
#
# File: train.py
# Author: Carlos Cruz-Marín
# ----------------------------------------------------------
# Trains AtomicGNN to predict atomic logP contributions.
#
# Usage:
#   python train.py -i training_set.sdf
# ----------------------------------------------------------

import argparse
import time

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from src.model import AtomicGNN
from src.training import train
import torch
from torch_geometric.loader import DataLoader

from src.graph import build_dataset
from src import logger

def main():
    st = time.time()
    parser = argparse.ArgumentParser(description='Train a GNN for predicting atomic logP contributions')
    parser.add_argument('-i', required=True, help='Training SDF file (3D, explicit H, per-atom molFileValue with atomic logP in the 2nd column)')

    args = parser.parse_args()

    logger.check_exist(args.i)
    dataset = build_dataset(args.i)
    if len(dataset) < 2:
        logger.msg_error_quit(f"Only {len(dataset)} valid molecule(s) found in {args.i}. Check that atoms carry the 'molFileValue' property.")

    train_set, val_set = train_test_split(dataset, train_size=0.8, random_state=0)
    for i, data in enumerate(train_set):
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            logger.msg(f"Graph {i} in train_set has invalid features")
    for i, data in enumerate(val_set):
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            logger.msg(f"Graph {i} in val_set has invalid features")

    clean_train_set = [data for data in train_set if not (torch.isnan(data.x).any() or torch.isinf(data.x).any())]
    clean_val_set = [data for data in val_set if not (torch.isnan(data.x).any() or torch.isinf(data.x).any())]

    train_set = clean_train_set
    val_set = clean_val_set

    # Create output folders if they do not exist
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("preds").mkdir(parents=True, exist_ok=True)

    x_num_dim = train_set[0].x.size(1)
    edge_dim  = train_set[0].edge_attr.size(1)

    model = AtomicGNN(x_num_dim=x_num_dim, edge_dim=edge_dim)

    model, (mu, sd) = train(model, train_set, val_set, max_Z=100)

    # Save models
    torch.save(model.state_dict(), "models/atomic_gnn.pth")
    torch.save(mu, "models/mu.pt")
    torch.save(sd, "models/sd.pt")
    logger.msg("Saved: models/atomic_gnn.pth")
    logger.msg("Saved: models/mu.pt")
    logger.msg("Saved: models/sd.pt")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device).eval()
    mu = mu.to(device)  # [max_Z, 1]
    sd = sd.to(device)  # [max_Z, 1]

    def write_predictions_csv(dataset, output_csv):
        rows = []
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for data in loader:
                data = data.to(device)

                z = data.z.long() - 1
                y = data.y.float()

                yhat_std = model(data)
                yhat = yhat_std * sd[z] + mu[z]
                batch_index = data.batch if hasattr(data, "batch") else None

                for i in range(y.size(0)):
                    row = {
                        "atom_index_in_batch": i,
                        "Z": int(data.z[i].item()),
                        "Z_index": int(z[i].item())}

                    if hasattr(data, "name") and batch_index is not None:
                        mol_idx = int(batch_index[i].item())

                        if isinstance(data.name, list):
                            row["mol_name"] = data.name[mol_idx]
                        else:
                            row["mol_name"] = data.name

                        row["mol_index_in_batch"] = mol_idx

                    true_value = float(y[i, 0].item())
                    pred_value = float(yhat[i, 0].item())
                    row["True_logP"] = true_value
                    row["Predicted_logP"] = pred_value
                    row["Error_logP"] = pred_value - true_value

                    rows.append(row)
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(rows).to_csv(output_csv, index=False)
            logger.msg(f"Wrote {output_csv}")

    write_predictions_csv(train_set, "preds/train_predictions.csv")
    write_predictions_csv(val_set, "preds/val_predictions.csv")

    et = time.time()
    logger.msg_time(f"Execution time is: {et-st}")

if __name__ == '__main__':
    main()
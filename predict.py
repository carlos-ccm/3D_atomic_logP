#!/usr/bin/env python
# ----------------------------------------------------------
# CBDD - Torribera, Universitat de Barcelona
#
# File: predict.py
# Author: Carlos Cruz-Marín
# ----------------------------------------------------------
# Predicts atomic logP contributions for the molecules of an input
# SDF using a trained AtomicGNN model.
#
# Usage:
#   python predict.py -i input.sdf -o output.sdf --model_dir models/ --num_workers 8
# ----------------------------------------------------------

import os
import sys
import gzip
import argparse

os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool

from src.graph import mol_to_graph
from src.model import AtomicGNN
from src import logger

# Layout of the per-atom 'molFileValue' output: 5 columns, the atomic logP
# goes in the 2nd one (same position it is read from for training) and the
# remaining columns are filled with zeros.
N_OUTPUT_COLUMNS = 5
LOGP_COLUMN = 1


def read_molecules(input_file):
    """
    Reads an SDF (.sdf) or gzipped SDF (.sdf.gz) keeping the hydrogens.
    Entries that RDKit cannot parse are skipped with a warning.
    """
    logger.check_exist(input_file)
    if input_file.endswith(".gz"):
        with gzip.open(input_file) as fh:
            mols = list(Chem.ForwardSDMolSupplier(fh, removeHs=False))
    elif input_file.endswith(".sdf"):
        mols = list(Chem.SDMolSupplier(input_file, removeHs=False))
    else:
        logger.msg_error_quit(f"Unsupported input format (use .sdf or .sdf.gz): {input_file}")

    n_bad = sum(m is None for m in mols)
    if n_bad:
        logger.msg_warning(f"{n_bad} molecule(s) could not be parsed by RDKit and will be skipped")
    return [m for m in mols if m is not None]


class StopProcessing(Exception):
    pass


def has_explicit_hydrogens(mol):
    """True if the molecule carries explicit H atoms, or has no H to add at all."""
    if any(at.GetAtomicNum() == 1 for at in mol.GetAtoms()):
        return True
    return Chem.AddHs(mol).GetNumAtoms() == mol.GetNumAtoms()


def process_mol(args):
    """
    Converts one molecule to its graph. Returns (idx, graph) or None on failure.
    Only the graph is sent back from the worker; the original molecule (with all
    its SDF properties) stays in the main process.
    """
    mol, idx = args
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx}"

    if not has_explicit_hydrogens(mol):
        logger.msg_warning(f"Molecule '{name}' (index {idx}) has no explicit hydrogens. "
                                 "Input structures must be 3D with explicit H.")
        raise StopProcessing()

    try:
        return (idx, mol_to_graph(mol, with_targets=False))
    except Exception as e:
        logger.msg_warning(f"Could not process molecule '{name}' (index {idx}): {e}")
        return None


def generate_graphs_parallel(mols, n_cores=1):
    """
    Builds the graphs of all molecules, optionally in parallel.
    Returns {idx: graph}. Keys are integer indices, so duplicated molecules
    (e.g. several conformers of the same compound) are all kept.
    """
    items = [(mol, idx) for idx, mol in enumerate(mols)]
    try:
        if n_cores == 1:
            results = [process_mol(item) for item in tqdm(items, desc="Generating graphs")]
        else:
            with Pool(processes=n_cores) as pool:
                results = list(tqdm(pool.imap(process_mol, items), total=len(items), desc="Generating graphs"))
    except StopProcessing:
        logger.msg_error("Finishing the prediction")
        sys.exit(1)

    return {idx: graph for idx, graph in (r for r in results if r is not None)}


def predict_gnn(input_file, output_file, model_dir, molxplore, num_workers):
    '''
    Main function to predict atomic properties using a GNN model.
        1. Loads the input molecules
        2. Generates graph representations
        3. Loads the pre-trained model
        4. Makes predictions
        5. Writes the results to an output SDF
    '''

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(model_dir, "atomic_gnn.pth")
    mu_path    = os.path.join(model_dir, "mu.pt")
    sd_path    = os.path.join(model_dir, "sd.pt")

    if not all(os.path.exists(path) for path in [model_path, mu_path, sd_path]):
        logger.msg_error_quit(
            "Model files not found. Expected:\n"
            f"  {model_path}\n" f"  {mu_path}\n" f"  {sd_path}")

    mols = read_molecules(input_file)
    logger.msg_g(f"\nLoaded {len(mols)} molecules from input file.\n")

    graph_dict = generate_graphs_parallel(mols, n_cores=num_workers)
    if not graph_dict:
        logger.msg_error_quit("No molecule could be converted to a graph.")

    first_graph = next(iter(graph_dict.values()))
    x_num_dim = first_graph.x.size(1)
    edge_dim = first_graph.edge_attr.size(1)

    mu = torch.load(mu_path, map_location=DEVICE)
    sd = torch.load(sd_path, map_location=DEVICE)

    if mu.ndim != 2 or sd.ndim != 2 or mu.size(1) != 1 or sd.size(1) != 1:
        logger.msg_error_quit("Loaded mu.pt and sd.pt have an unexpected shape. Expected [max_Z, 1].")

    # ---- LOAD MODEL ----
    model = AtomicGNN(x_num_dim=x_num_dim, edge_dim=edge_dim)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    logger.msg_g("\nPredicting atomic logP\n")

    dict_predictions = {}
    with torch.no_grad():
        for idx, graph in tqdm(graph_dict.items(), total=len(graph_dict), desc="Predicting"):
            graph = graph.to(DEVICE)
            z = graph.z.long() - 1           # [N_atoms]
            yhat_std = model(graph)          # [N_atoms, 1]
            yhat = yhat_std * sd[z] + mu[z]  # [N_atoms, 1]
            dict_predictions[idx] = yhat.cpu().numpy()

    # ---- OUTPUT ----
    out = Chem.SDWriter(output_file)
    for idx, predictions in dict_predictions.items():
        mol = mols[idx]
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx}"
        num_atoms = mol.GetNumAtoms()
        if predictions.shape[0] != num_atoms:
            logger.msg_error_quit(f"Atom mismatch for molecule {name}: mol has {num_atoms} atoms but predictions have {predictions.shape[0]} atoms.")

        molecular_logp = 0.0
        for i in range(num_atoms):
            atomic_logp = float(predictions[i, 0])
            molecular_logp += atomic_logp
            values = [0.0] * N_OUTPUT_COLUMNS
            values[LOGP_COLUMN] = atomic_logp
            formatted_values = [f"{v: .4f}" for v in values]
            if molxplore:
                formatted_values.append("0")
            mol.GetAtomWithIdx(i).SetProp("molFileValue", "  ".join(formatted_values))

        mol.SetProp("ML_molecular_logP", f"{molecular_logp:.4f}")
        out.write(mol)

    out.close()
    n_skipped = len(mols) - len(dict_predictions)
    if n_skipped:
        logger.msg_warning(f"{n_skipped} molecule(s) could not be processed and are not in the output.")
    logger.msg(f"\nPredictions written to {output_file}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict atomic logP contributions using a GNN model')
    parser.add_argument('-i', required=True, help='Input SDF (.sdf or .sdf.gz) with 3D molecules and explicit hydrogens')
    parser.add_argument('-o', required=True, help='Output SDF file for predictions')
    parser.add_argument('--model_dir', default="models/", required=False, help='Directory containing atomic_gnn.pth, mu.pt and sd.pt')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of CPU cores for graph generation')
    parser.add_argument('--molxplore', action='store_true', help='Format the output for visualization in MolXplore')

    args = parser.parse_args()
    logger.msg_gb(f"Models are loaded from: {args.model_dir}")
    predict_gnn(args.i, args.o, args.model_dir, args.molxplore, num_workers=args.num_workers)

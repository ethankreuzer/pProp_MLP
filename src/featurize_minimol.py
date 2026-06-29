#!/usr/bin/env python
"""
One-time MiniMol featurization of every unique molecule -> cached embeddings.

This is the ONLY script that imports MiniMol. MiniMol (graphium + torch-geometric
+ an old torchmetrics) conflicts with the main project venv, so it lives in a
separate environment:

    .venv_minimol/bin/python src/featurize_minimol.py

It writes, into data/cache/:
    minimol_embeddings.npy   (M, 512) float32, one row per unique molecule
    minimol_smiles.txt       M canonical SMILES, aligned to the rows above

The sweep/training code (main venv) only ever reads these two files; it never
imports MiniMol. Re-run with --force to recompute.

Embeddings are computed for the full unique-molecule set (same canonicalize +
max-pProp dedup as make_splits), so any split_N indexes straight into them.
"""

import argparse

import numpy as np

from data_utils import CACHE_DIR, load_unique_molecules

EMB_PATH = CACHE_DIR / "minimol_embeddings.npy"
SMI_PATH = CACHE_DIR / "minimol_smiles.txt"


def to_numpy(x):
    """Coerce one MiniMol output element to a 1-D float32 numpy vector."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="SMILES per MiniMol call.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if the cache exists.")
    args = ap.parse_args()

    if EMB_PATH.exists() and SMI_PATH.exists() and not args.force:
        emb = np.load(EMB_PATH)
        print(f"Cache present: {EMB_PATH} {emb.shape}. Use --force to recompute.")
        return

    print("Loading unique molecules ...")
    mols = load_unique_molecules()
    smiles = mols["canon"].tolist()
    print(f"  {len(smiles):,} unique canonical SMILES.")

    from minimol import Minimol

    print("Loading MiniMol ...")
    featurizer = Minimol()

    vecs = []
    n = len(smiles)
    for start in range(0, n, args.batch_size):
        chunk = smiles[start:start + args.batch_size]
        out = featurizer(chunk)
        vecs.extend(to_numpy(v) for v in out)
        done = min(start + args.batch_size, n)
        print(f"  featurized {done:,}/{n:,}", flush=True)

    emb = np.stack(vecs).astype(np.float32)
    assert emb.shape[0] == n, f"got {emb.shape[0]} embeddings for {n} molecules"
    print(f"Embeddings: {emb.shape} (expected (*, 512))")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)
    SMI_PATH.write_text("\n".join(smiles) + "\n")
    print(f"Wrote {EMB_PATH}\nWrote {SMI_PATH}")


if __name__ == "__main__":
    main()

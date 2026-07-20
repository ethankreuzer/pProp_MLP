#!/usr/bin/env python
"""
Shared data/label utilities for the pProp MLP sweep.

Single source of truth for turning the raw CSV + a split directory's
train.smi / val.smi into (MiniMol embedding, pProp-class label) pairs.

The pProp class binning is imported directly from make_splits.py so the
classifier targets are *identical* to the bins the splits were built on
(CLASS_EDGES / CLASS_NAMES). The CSV -> unique-molecule reduction reproduces
make_splits' dedup exactly: canonicalize SMILES, keep the max-pProp copy of any
duplicate. After building labels for a split we assert the per-class val counts
match that split's split_meta.json.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from make_splits import (
    CACHE_DIR,
    CLASS_EDGES,
    CLASS_NAMES,
    DATA_CSV,
    pprop_class,
)

N_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {nm: i for i, nm in enumerate(CLASS_NAMES)}

# ECFP (Morgan) fingerprint block reused from make_splits' cache. make_splits
# writes mol_fps_r{radius}_b{n_bits}.npy as bit-PACKED uint8 (n_bits//8 cols),
# aligned to raw-CSV row order; we unpack + dedup to the unique-molecule set.
FP_RADIUS, FP_NBITS = 2, 2048


def _canonicalize(smiles):
    """Canonical SMILES for a list of raw SMILES (None where unparseable)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        out.append(Chem.MolToSmiles(mol) if mol is not None else None)
    return out


def load_unique_molecules(csv_path=DATA_CSV):
    """
    Reduce the raw CSV to unique molecules, exactly as make_splits does.

    Returns a DataFrame with columns:
        canon       canonical SMILES (one row per unique molecule)
        pprop       pProp of the max-pProp copy
        class_name  pProp class name (e.g. "5-7.5")
        class_idx   integer class label 0..N_CLASSES-1
    """
    df = pd.read_csv(csv_path)
    raw_smiles = df["SMILES"].astype(str).tolist()
    pprop_all = df["pprop"].to_numpy()

    # Reuse make_splits' cached canonical SMILES if available (computed once,
    # aligned to CSV row order); otherwise canonicalize here.
    scaff_path = CACHE_DIR / "scaffolds.pkl"
    canon = None
    if scaff_path.exists():
        cached = pd.read_pickle(scaff_path)
        if "canon" in cached.columns and len(cached) == len(raw_smiles):
            canon = cached["canon"].to_numpy()
    if canon is None:
        canon = np.array(_canonicalize(raw_smiles), dtype=object)

    valid = np.array([c is not None for c in canon])
    vidx = np.where(valid)[0]

    dedup = (
        pd.DataFrame({"canon": canon[vidx], "pprop": pprop_all[vidx], "row": vidx})
        .sort_values("pprop", ascending=False, kind="stable")
        .drop_duplicates("canon", keep="first")
        .sort_values("row")
    )
    keep = dedup["row"].to_numpy()
    canon_u = np.array([canon[i] for i in keep], dtype=object)
    pprop_u = pprop_all[keep]
    class_name = pprop_class(pprop_u)
    class_idx = np.array([CLASS_TO_IDX[nm] for nm in class_name], dtype=np.int64)

    return pd.DataFrame(
        {
            "canon": canon_u,
            "pprop": pprop_u,
            "class_name": class_name,
            "class_idx": class_idx,
        }
    )


def load_ecfp_features(radius=FP_RADIUS, n_bits=FP_NBITS):
    """
    ECFP feature block aligned to unique molecules, reusing make_splits' cache.

    make_splits already computed and cached the Morgan fingerprints (bit-packed,
    aligned to raw-CSV row order) in mol_fps_r{radius}_b{n_bits}.npy, alongside
    the canonical SMILES (scaffolds.pkl "canon" column) and a validity mask
    (valid.npy). We reuse those directly instead of recomputing: dedup to unique
    canonical SMILES (first occurrence -- duplicate rows share an identical FP),
    then unpack to a dense (M, n_bits) float32 matrix.

    Returns (features, smiles_index): features[i] is the ECFP of smiles_index[i],
    the same (matrix, aligned-canonical-SMILES) shape build_split_arrays expects
    for MiniMol, so it can be passed straight through as an extra feature block.
    """
    fps_path = CACHE_DIR / f"mol_fps_r{radius}_b{n_bits}.npy"
    scaff_path = CACHE_DIR / "scaffolds.pkl"
    valid_path = CACHE_DIR / "valid.npy"
    for p in (fps_path, scaff_path, valid_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found; ECFP features reuse make_splits' cache. Run "
                "`.venv/bin/python src/make_splits.py ...` (or --figures-only) "
                "at least once to populate data/cache/."
            )

    packed = np.load(fps_path)                 # (R, n_bits//8) uint8, CSV-row order
    valid = np.load(valid_path)                # (R,) bool
    canon = pd.read_pickle(scaff_path)["canon"].to_numpy()  # (R,) canonical SMILES
    if not (len(packed) == len(valid) == len(canon)):
        raise ValueError(
            f"ECFP cache row mismatch: fps={len(packed)}, valid={len(valid)}, "
            f"scaffolds={len(canon)}. Recompute with make_splits --force."
        )

    dedup = (
        pd.DataFrame({"canon": canon, "row": np.arange(len(canon)), "valid": valid})
        .query("valid and canon == canon")     # drop invalid + NaN canon
        .drop_duplicates("canon", keep="first")
    )
    rows = dedup["row"].to_numpy()
    smiles_index = dedup["canon"].tolist()
    features = np.unpackbits(packed[rows], axis=1)[:, :n_bits].astype(np.float32)
    return features, smiles_index


def load_ecfp_precomputed(radius, n_bits=FP_NBITS):
    """
    ECFP feature block from a per-radius cache written by src/featurize_ecfp.py.

    Unlike load_ecfp_features (which reuses make_splits' single r2/b2048 cache in
    CSV-row order), this reads a self-contained, unique-molecule-order cache so the
    two-tower branch can SWEEP the Morgan radius: ecfp_r{radius}_b{n_bits}.npy
    (bit-packed uint8) + ecfp_r{radius}_b{n_bits}_smiles.txt (aligned canonical
    SMILES). Both are produced together, so no scaffolds.pkl / valid.npy coupling.

    Returns (features, smiles_index): features[i] is the ECFP of smiles_index[i] --
    the same (matrix, aligned-canonical-SMILES) shape build_split_arrays expects, so
    it passes straight through as an extra feature block alongside MiniMol.
    """
    fps_path = CACHE_DIR / f"ecfp_r{radius}_b{n_bits}.npy"
    smi_path = CACHE_DIR / f"ecfp_r{radius}_b{n_bits}_smiles.txt"
    for p in (fps_path, smi_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found; precompute ECFP for radius {radius} first with "
                f"`.venv/bin/python src/featurize_ecfp.py --radii {radius}`."
            )

    packed = np.load(fps_path)                       # (M, n_bits//8) uint8
    smiles_index = smi_path.read_text().splitlines()
    if len(packed) != len(smiles_index):
        raise ValueError(
            f"ECFP cache row mismatch for radius {radius}: fps={len(packed)}, "
            f"smiles={len(smiles_index)}. Recompute with "
            f"`src/featurize_ecfp.py --radii {radius} --force`."
        )
    features = np.unpackbits(packed, axis=1)[:, :n_bits].astype(np.float32)
    return features, smiles_index


def read_smi(path):
    """Read a .smi file (one SMILES per line) -> list[str]."""
    lines = Path(path).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def build_split_arrays(split_dir, embeddings, smiles_index, verify_meta=True,
                       return_pprop=False, extra_features=None):
    """
    Assemble (X, y) embedding/label arrays for a split's train and val sets.

    Parameters
    ----------
    split_dir : path to data/split_N (contains train.smi, val.smi, split_meta.json)
    embeddings : (M, D) float32 array of MiniMol embeddings
    smiles_index : list[str] of length M; smiles_index[i] is the canonical SMILES
                   of embeddings[i] (the order featurize_minimol.py wrote)
    verify_meta : if True, assert per-class val counts match split_meta.json
    return_pprop : if True, also return pprop_train / pprop_val (continuous pProp
                   values needed as MSE regression targets)
    extra_features : optional list of (features, smiles_index) blocks (e.g. ECFP
                   from load_ecfp_features) column-concatenated onto MiniMol,
                   aligned per molecule by canonical SMILES. A molecule missing
                   from any block is dropped (counted as missing), so X columns
                   are [MiniMol | block_0 | block_1 | ...] with rows all present.

    Returns dict with X_train, y_train, X_val, y_val (numpy) and the class names.
    If return_pprop=True, also includes pprop_train and pprop_val (float64).
    """
    split_dir = Path(split_dir)
    labels = load_unique_molecules()
    label_of = dict(zip(labels["canon"], labels["class_idx"]))
    pprop_of = dict(zip(labels["canon"], labels["pprop"]))
    row_of = {smi: i for i, smi in enumerate(smiles_index)}

    extra = list(extra_features or [])
    extra_row_of = [{smi: i for i, smi in enumerate(si)} for (_, si) in extra]

    def gather(smi_file):
        smis = read_smi(split_dir / smi_file)
        rows, ys, pprops, missing = [], [], [], 0
        extra_rows = [[] for _ in extra]
        for smi in smis:
            if (smi in row_of and smi in label_of
                    and all(smi in er for er in extra_row_of)):
                rows.append(row_of[smi])
                ys.append(label_of[smi])
                pprops.append(pprop_of[smi])
                for k, er in enumerate(extra_row_of):
                    extra_rows[k].append(er[smi])
            else:
                missing += 1
        if missing:
            raise KeyError(
                f"{missing} SMILES in {smi_file} lack an embedding, label, or "
                "extra feature; re-run featurize_minimol.py (and, for ECFP, "
                "make_splits) on the current molecule set."
            )
        idx = np.asarray(rows)
        X = embeddings[idx]
        if extra:
            blocks = [X] + [feats[np.asarray(extra_rows[k])]
                            for k, (feats, _) in enumerate(extra)]
            X = np.concatenate(blocks, axis=1)
        return X, np.asarray(ys, dtype=np.int64), np.asarray(pprops, dtype=np.float64)

    X_train, y_train, pprop_train = gather("train.smi")
    X_val, y_val, pprop_val = gather("val.smi")

    if verify_meta:
        _verify_against_meta(split_dir, y_val)

    out = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "class_names": CLASS_NAMES,
    }
    if return_pprop:
        out["pprop_train"] = pprop_train
        out["pprop_val"] = pprop_val
    return out


def _verify_against_meta(split_dir, y_val):
    """Cross-check per-class val counts against split_meta.json."""
    import json

    meta_path = Path(split_dir) / "split_meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    per_class = meta.get("per_class")
    if not per_class:
        return
    counts = np.bincount(y_val, minlength=N_CLASSES)
    for name, entry in per_class.items():
        n_val = entry.get("val")
        if name in CLASS_TO_IDX and n_val is not None:
            got = int(counts[CLASS_TO_IDX[name]])
            if got != int(n_val):
                raise AssertionError(
                    f"val count mismatch for class {name}: labels give {got}, "
                    f"split_meta.json says {n_val}. Label derivation diverged "
                    "from the split."
                )

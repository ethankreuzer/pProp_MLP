#!/usr/bin/env python
"""
One-time ECFP (Morgan) featurization of every unique molecule -> cached
fingerprints, one cache per radius.

This is the two-tower branch's ECFP counterpart to featurize_minimol.py. Unlike
the main branch (which reuses make_splits' single r2/b2048 cache via
data_utils.load_ecfp_features), here we want to SWEEP the Morgan radius, so we
precompute a self-contained cache per radius up front. Runs in the main .venv
(RDKit is there; no MiniMol needed):

    .venv/bin/python src/featurize_ecfp.py --radii 2 3 4

It writes, into data/cache/, for each radius r:
    ecfp_r{r}_b{nbits}.npy          (M, nbits//8) uint8, bit-packed, one row / mol
    ecfp_r{r}_b{nbits}_smiles.txt   M canonical SMILES, aligned to the rows above

Fingerprints are computed for the full unique-molecule set in the SAME order as
load_unique_molecules() (canonicalize + max-pProp dedup), identical to MiniMol's
cache, so any split_N indexes straight into them and build_split_arrays can align
the two blocks by canonical SMILES. Re-run with --force to recompute.

Faithfulness: a Morgan fingerprint depends only on the molecular graph, so
FP(canonical SMILES) == FP(raw SMILES) for the same molecule. The radius-2 cache
here therefore reproduces make_splits' mol_fps_r2_b2048.npy bit-for-bit on the
overlapping molecules (verified separately).
"""

import argparse

import numpy as np

from data_utils import CACHE_DIR, FP_NBITS, load_unique_molecules


def cache_paths(radius, n_bits):
    """(fps_path, smiles_path) for a given radius / n_bits."""
    fps = CACHE_DIR / f"ecfp_r{radius}_b{n_bits}.npy"
    smi = CACHE_DIR / f"ecfp_r{radius}_b{n_bits}_smiles.txt"
    return fps, smi


def _fp_chunk(smiles_chunk, radius, n_bits):
    """Bit-packed Morgan FPs for a chunk of canonical SMILES -> (M, n_bits//8) uint8."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import DataStructs, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    packed = np.zeros((len(smiles_chunk), n_bits // 8), dtype=np.uint8)
    for i, smi in enumerate(smiles_chunk):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # load_unique_molecules only yields parseable canonical SMILES, so
            # this should never fire; guard so one bad row can't shift alignment.
            raise ValueError(f"unparseable canonical SMILES at chunk index {i}: {smi!r}")
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        packed[i] = np.packbits(arr)
    return packed


def featurize_radius(smiles, radius, n_bits, n_jobs, force):
    """Compute + cache the packed ECFP matrix for one radius. Returns the path."""
    fps_path, smi_path = cache_paths(radius, n_bits)
    if fps_path.exists() and smi_path.exists() and not force:
        packed = np.load(fps_path)
        print(f"  r={radius}: cache present {fps_path.name} {packed.shape}. "
              "Use --force to recompute.")
        return fps_path

    from joblib import Parallel, delayed

    n = len(smiles)
    n_chunks = max((n_jobs if n_jobs > 0 else 64) * 4, 1)
    bounds = np.linspace(0, n, n_chunks + 1).astype(int)
    chunks = [smiles[bounds[k]:bounds[k + 1]]
              for k in range(n_chunks) if bounds[k + 1] > bounds[k]]
    print(f"  r={radius}: computing Morgan FPs for {n:,} molecules "
          f"({len(chunks)} chunks, n_bits={n_bits})...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fp_chunk)(ch, radius, n_bits) for ch in chunks
    )

    packed = np.zeros((n, n_bits // 8), dtype=np.uint8)
    pos = 0
    for pk in results:
        packed[pos:pos + len(pk)] = pk
        pos += len(pk)
    assert pos == n, f"assembled {pos} rows for {n} molecules"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(fps_path, packed)
    smi_path.write_text("\n".join(smiles) + "\n")
    print(f"  r={radius}: wrote {fps_path} {packed.shape}\n"
          f"  r={radius}: wrote {smi_path}")
    return fps_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=int, nargs="+", default=[2, 3, 4],
                    help="Morgan radii to precompute (one cache each).")
    ap.add_argument("--n-bits", type=int, default=FP_NBITS,
                    help="Fingerprint bit length (fpSize).")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="joblib worker count (-1 = all cores).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if a radius cache exists.")
    args = ap.parse_args()

    print("Loading unique molecules ...")
    mols = load_unique_molecules()
    smiles = mols["canon"].tolist()
    print(f"  {len(smiles):,} unique canonical SMILES.")

    for radius in args.radii:
        featurize_radius(smiles, radius, args.n_bits, args.n_jobs, args.force)


if __name__ == "__main__":
    main()

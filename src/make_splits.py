#!/usr/bin/env python
"""
pProp-class-stratified train/val split generator for data/ampc_subset.csv.

Produces `n` independent train/val splits, each in its own directory
`data/split_{i}/` (existing splits are never overwritten).

Goal
----
A validation set that (a) SPANS EVERY pProp class and (b) keeps every val
molecule's max ECFP Tanimoto to any train molecule <= --ceiling (default 0.70)
wherever that is feasible, relaxing the ceiling ONLY for the high-pProp classes
that are too tightly clustered to hold out cleanly.

pProp = log10(N / docking_rank) (rank 1 -> pProp ~9.17), high pProp = rare elite
binder. Classes: [0,4.5) [4.5,5.5) [5.5,6.5) [6.5,7) [7,7.5) [7.5,inf).

How the ceiling is guaranteed
-----------------------------
We build single-linkage connected components of the ">ceiling" Tanimoto graph
over ALL molecules. A molecule and all its >ceiling neighbours share one
component, so assigning whole components to val (or train) atomically can never
separate a val molecule from a close neighbour -> every val molecule in a
held-out component is guaranteed <= ceiling to train.

Assembly (per split, with a per-split RNG seed)
-----------------------------------------------
  * Per-class target = --val-frac of each class (so val mirrors the class mix).
  * Greedily hold out whole small components (size <= --max-unit-frac * N),
    scarce-class first, until each class hits its target. Components larger than
    the cap (the percolating giant) stay in train. This is the clean path.
  * BEST-EFFORT fallback: if some class still can't reach its target from clean
    components, move that class's least-train-similar leftover molecules into
    val until the target is met. These MAY exceed the ceiling; they are marked
    and reported per class. (Not triggered at the default 12.5% target.)
  * Every split is brute-force verified on GPU: max val->train Tanimoto per
    class, and the count of molecules over the ceiling.

Pipeline / caching
------------------
  1. Per-molecule scaffold + Morgan FP (r=2, 2048b) + canonical SMILES; cached
     once (param-independent).
  2. Deduplicate to unique molecules by canonical SMILES.
  3. >ceiling components over all molecules; cached, keyed by ceiling.
  4. Assemble + verify each split.

Outputs per split: train.smi, val.smi (one canonical SMILES/line, deduped),
clusters.csv (smiles, scaffold, pprop, pprop_class, component, split,
max_tani_to_train, over_ceiling), split_meta.json. Figures deferred.
"""

import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

PROJ = Path(__file__).resolve().parent.parent
DATA_CSV = PROJ / "data" / "ampc_subset.csv"
CACHE_DIR = PROJ / "data" / "cache"
SPLITS_PARENT = PROJ / "data"

CLASS_EDGES = [(-np.inf, 4.5), (4.5, 5.5), (5.5, 6.5), (6.5, 7.0), (7.0, 7.5), (7.5, np.inf)]
CLASS_NAMES = ["0-4.5", "4.5-5.5", "5.5-6.5", "6.5-7", "7-7.5", "7.5+"]


def pprop_class(pprop):
    out = np.empty(len(pprop), dtype=object)
    for (lo, hi), nm in zip(CLASS_EDGES, CLASS_NAMES):
        out[(pprop >= lo) & (pprop < hi)] = nm
    return out


def unpack_bits(packed):
    return np.unpackbits(packed, axis=1)[:, :packed.shape[1] * 8].astype(np.float32)


# ---------------------------------------------------------------------------
# Step 1: per-molecule scaffold + Morgan fingerprint + canonical SMILES (cache)
# ---------------------------------------------------------------------------
def _process_chunk(smiles_chunk, radius, n_bits):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    scaffolds, canon = [], []
    packed = np.zeros((len(smiles_chunk), n_bits // 8), dtype=np.uint8)
    valid = np.zeros(len(smiles_chunk), dtype=bool)
    for i, smi in enumerate(smiles_chunk):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds.append(None)
            canon.append(None)
            continue
        valid[i] = True
        canon.append(Chem.MolToSmiles(mol))
        try:
            scaffolds.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)))
        except Exception:
            scaffolds.append("")
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        packed[i] = np.packbits(arr)
    return scaffolds, canon, packed, valid


def build_scaffolds_and_fps(smiles, radius, n_bits, n_jobs, force):
    scaff_path = CACHE_DIR / "scaffolds.pkl"
    fps_path = CACHE_DIR / f"mol_fps_r{radius}_b{n_bits}.npy"
    valid_path = CACHE_DIR / "valid.npy"

    if not force and scaff_path.exists() and fps_path.exists() and valid_path.exists():
        cached = pd.read_pickle(scaff_path)
        if "canon" in cached.columns and len(cached) == len(smiles):
            print(f"  Loading cached scaffolds + FPs from {CACHE_DIR}")
            packed = np.load(fps_path)
            valid = np.load(valid_path)
            if len(packed) == len(smiles):
                return cached["scaffold"].to_numpy(), cached["canon"].to_numpy(), packed, valid
        print("  WARNING: cache miss/length mismatch; recomputing.")

    from joblib import Parallel, delayed

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n = len(smiles)
    n_chunks = max((n_jobs if n_jobs > 0 else 64) * 4, 1)
    bounds = np.linspace(0, n, n_chunks + 1).astype(int)
    chunks = [smiles[bounds[k]:bounds[k + 1]] for k in range(n_chunks) if bounds[k + 1] > bounds[k]]
    print(f"  Computing scaffolds + Morgan FPs for {n:,} molecules ({len(chunks)} chunks)...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_chunk)(ch, radius, n_bits) for ch in chunks
    )

    scaffolds = np.empty(n, dtype=object)
    canon = np.empty(n, dtype=object)
    packed = np.zeros((n, n_bits // 8), dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)
    pos = 0
    for sc, cn, pk, vl in results:
        m = len(sc)
        scaffolds[pos:pos + m] = sc
        canon[pos:pos + m] = cn
        packed[pos:pos + m] = pk
        valid[pos:pos + m] = vl
        pos += m
    assert pos == n

    pd.DataFrame({"smiles": smiles, "canon": canon, "scaffold": scaffolds}).to_pickle(scaff_path)
    np.save(fps_path, packed)
    np.save(valid_path, valid)
    print(f"  Cached scaffolds + FPs ({(~valid).sum()} unparseable molecules).")
    return scaffolds, canon, packed, valid


# ---------------------------------------------------------------------------
# Step 3: >ceiling single-linkage components over ALL molecules (cache)
# ---------------------------------------------------------------------------
def build_components(packed_v, ceiling, device, row_block, force):
    """Connected components of the graph {(i,j): Tanimoto(i,j) > ceiling}.
    Holding out a whole component guarantees its members are <= ceiling to
    every molecule outside it. Cached, keyed by ceiling."""
    cpath = CACHE_DIR / f"simcomp_all_c{ceiling:g}.npy"
    if not force and cpath.exists():
        labels = np.load(cpath)
        if len(labels) == len(packed_v):
            print(f"  Loading cached components from {cpath}")
            return labels
        print("  WARNING: cached components length mismatch; recomputing.")

    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(packed_v)
    print(f"  Building >{ceiling:g} Tanimoto graph over {n:,} molecules on {device}...")
    F = torch.from_numpy(unpack_bits(packed_v)).to(device)
    row_sum = F.sum(dim=1)
    rows, cols = [], []
    for b in range(0, n, row_block):
        bt = torch.arange(b, min(b + row_block, n), device=device)
        inter = F[bt] @ F.T
        union = row_sum[bt][:, None] + row_sum[None, :] - inter
        T = inter / union.clamp_min(1.0)
        mask = T > ceiling
        li = torch.arange(len(bt), device=device)
        mask[li, bt] = False
        nz = mask.nonzero(as_tuple=False)
        gi, gj = bt[nz[:, 0]], nz[:, 1]
        upper = gj > gi                                   # keep each undirected edge once
        rows.append(gi[upper].cpu().numpy().astype(np.int32))
        cols.append(gj[upper].cpu().numpy().astype(np.int32))
        del inter, union, T, mask, nz
    del F
    if device.type == "cuda":
        torch.cuda.empty_cache()

    r = np.concatenate(rows) if rows else np.zeros(0, np.int32)
    c = np.concatenate(cols) if cols else np.zeros(0, np.int32)
    print(f"  {len(r):,} edges; running connected_components...")
    A = coo_matrix((np.ones(len(r), bool), (r, c)), shape=(n, n))
    _, labels = connected_components(A, directed=False, connection="weak")
    labels = labels.astype(np.int32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cpath, labels)
    sizes = np.bincount(labels)
    print(f"  {labels.max() + 1:,} components; largest = {sizes.max():,} "
          f"({100 * sizes.max() / n:.2f}% of N). Cached -> {cpath}")
    return labels


# ---------------------------------------------------------------------------
# Step 4a: stratified clean holdout of whole components
# ---------------------------------------------------------------------------
def assemble_clean(labels, classes, val_frac, max_unit_frac, rng):
    """Hold out whole small components, scarce-class first, to reach a per-class
    target of val_frac * |class|. Returns (split array, val_count, targets)."""
    n = len(labels)
    n_comp = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=n_comp)
    cap = max(int(max_unit_frac * n), 1)

    targets = {nm: int(round(val_frac * int((classes == nm).sum()))) for nm in CLASS_NAMES}

    # members per component (small ones only)
    order = np.argsort(labels, kind="stable")
    sorted_lab = labels[order]
    cuts = np.flatnonzero(np.diff(sorted_lab)) + 1
    groups = np.split(order, cuts)
    comp_ids = sorted_lab[np.concatenate(([0], cuts))]
    members = {int(cid): g for cid, g in zip(comp_ids, groups) if sizes[cid] <= cap}

    comp_class = {cid: classes[g] for cid, g in members.items()}
    by_class = defaultdict(list)
    small_ids = list(members.keys())
    rng.shuffle(small_ids)
    for cid in small_ids:
        for nm in set(comp_class[cid]):
            by_class[nm].append(cid)

    val_count = {nm: 0 for nm in CLASS_NAMES}
    chosen = set()
    for nm in reversed(CLASS_NAMES):                      # scarce (7.5+) -> abundant (0-4.5)
        for cid in by_class[nm]:
            if val_count[nm] >= targets[nm]:
                break
            if cid in chosen:
                continue
            chosen.add(cid)
            cc = comp_class[cid]
            for nm2 in CLASS_NAMES:
                val_count[nm2] += int((cc == nm2).sum())

    split = np.full(n, "train", dtype=object)
    if chosen:
        sel = np.zeros(n_comp, dtype=bool)
        sel[list(chosen)] = True
        split[sel[labels]] = "val"
    return split, val_count, targets


# ---------------------------------------------------------------------------
# Step 4b: best-effort top-up (may exceed the ceiling) for under-filled classes
# ---------------------------------------------------------------------------
def best_effort_topup(split, packed_v, classes, val_count, targets, device, row_block, rng):
    """For classes still below target, move their least-train-similar leftover
    molecules into val. Returns a boolean 'relaxed' mask (these may be >ceiling)."""
    import torch

    relaxed = np.zeros(len(split), dtype=bool)
    under = [nm for nm in CLASS_NAMES if val_count[nm] < targets[nm]]
    if not under:
        return relaxed

    bits = unpack_bits(packed_v)
    train_idx = np.where(split == "train")[0]
    Tr = torch.from_numpy(bits[train_idx]).to(device)
    tr_sum = Tr.sum(dim=1)
    for nm in under:
        cand = np.where((classes == nm) & (split == "train"))[0]
        if len(cand) == 0:
            continue
        mx = np.empty(len(cand), dtype=np.float32)       # max sim of each candidate to train
        for b in range(0, len(cand), row_block):
            cb = cand[b:b + row_block]
            V = torch.from_numpy(bits[cb]).to(device)
            inter = V @ Tr.T
            union = V.sum(1)[:, None] + tr_sum[None, :] - inter
            T = inter / union.clamp_min(1.0)
            T[T >= 0.999] = -1.0                          # ignore self / exact dup vs itself in train
            mx[b:b + len(cb)] = T.max(1).values.cpu().numpy()
            del V, inter, union, T
        need = targets[nm] - val_count[nm]
        pick = cand[np.argsort(mx)[:need]]               # least similar first
        split[pick] = "val"
        relaxed[pick] = True
        val_count[nm] += len(pick)
    del Tr
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return relaxed


# ---------------------------------------------------------------------------
# Brute-force verification: max val->train Tanimoto for EVERY val molecule
# ---------------------------------------------------------------------------
def verify_split(packed_v, split, device, row_block):
    import torch

    val_idx = np.where(split == "val")[0]
    train_idx = np.where(split == "train")[0]
    bits = unpack_bits(packed_v)
    Tr = torch.from_numpy(bits[train_idx]).to(device)
    tr_sum = Tr.sum(dim=1)
    maxes = np.empty(len(val_idx), dtype=np.float32)
    for b in range(0, len(val_idx), row_block):
        vb = val_idx[b:b + row_block]
        V = torch.from_numpy(bits[vb]).to(device)
        inter = V @ Tr.T
        union = V.sum(1)[:, None] + tr_sum[None, :] - inter
        T = inter / union.clamp_min(1.0)
        maxes[b:b + len(vb)] = T.max(1).values.cpu().numpy()
        del V, inter, union, T
    del Tr
    if device.type == "cuda":
        torch.cuda.empty_cache()
    full = np.zeros(len(split), dtype=np.float32)
    full[val_idx] = maxes
    return full                                          # max-to-train per molecule (0 for train rows)


# ---------------------------------------------------------------------------
# Figures: 8 PNGs per split (val->train similarity for all + 6 classes; counts)
# ---------------------------------------------------------------------------
def _sim_hist(ax, arr, title, ceiling):
    """Histogram of per-val-molecule max Tanimoto to train on `ax`."""
    ax.set_xlabel("Max ECFP Tanimoto similarity to train (per val molecule)")
    ax.set_ylabel("Number of validation molecules")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.3)
    if len(arr) == 0:
        ax.text(0.5, 0.5, "no validation molecules in this class",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1.0)
        return
    mx, md, mn = float(np.max(arr)), float(np.median(arr)), float(np.mean(arr))
    bins = np.linspace(0.0, max(1.0, mx), 50)
    ax.hist(arr, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.axvline(mn, color="green", ls="--", lw=1.3, label=f"mean = {mn:.3f}")
    ax.axvline(md, color="red", ls="--", lw=1.3, label=f"median = {md:.3f}")
    ax.axvline(mx, color="purple", ls=":", lw=1.5, label=f"max = {mx:.3f}")
    if ceiling is not None:
        ax.axvline(ceiling, color="black", lw=1.0, alpha=0.6, label=f"ceiling = {ceiling:.2f}")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="upper left")
    ax.text(0.98, 0.97,
            f"n = {len(arr):,}\nmean = {mn:.3f}\nmedian = {md:.3f}\nmax = {mx:.3f}",
            transform=ax.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))


def generate_figures(out_dir, ceiling=None):
    """Read out_dir/clusters.csv and write the 8 PNGs for that split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "clusters.csv")
    is_val = df["split"].to_numpy() == "val"
    cls = df["pprop_class"].to_numpy()
    sim = df["max_tani_to_train"].to_numpy()
    split_name = out_dir.name

    # 1 (all) + 6 (per class) similarity distributions
    panels = [("all", is_val)] + [(nm, is_val & (cls == nm)) for nm in CLASS_NAMES]
    for label, mask in panels:
        arr = sim[mask]
        arr = arr[~np.isnan(arr)]
        fig, ax = plt.subplots(figsize=(9, 5))
        title = (f"{split_name}: validation → train max similarity"
                 f"{'' if label == 'all' else f'  [pProp {label}]'}")
        _sim_hist(ax, arr, title, ceiling)
        fig.tight_layout()
        fig.savefig(out_dir / f"val_sim_to_train__{label}.png", dpi=150)
        plt.close(fig)

    # 8th: validation class counts + proportion-of-class
    counts, totals, props = [], [], []
    for nm in CLASS_NAMES:
        m = cls == nm
        tot = int(m.sum())
        vc = int((m & is_val).sum())
        counts.append(vc)
        totals.append(tot)
        props.append(vc / tot if tot else 0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CLASS_NAMES))
    ax.bar(x, counts, color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_ylim(top=max(max(counts), 1) * 5)
    for xi, vc, pr, tot in zip(x, counts, props, totals):
        ax.text(xi, max(vc, 1), f"{vc:,}\nprop={pr:.2f}\n/{tot:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("pProp class")
    ax.set_ylabel("Validation molecule count (log scale)")
    ax.set_title(f"{split_name}: validation class counts "
                 f"(count, proportion of class held out, /class total)")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "val_class_counts.png", dpi=150)
    plt.close(fig)
    print(f"    wrote 8 figures -> {out_dir}")


def next_split_index(parent, prefix="split_"):
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    existing = [
        int(m.group(1)) for p in parent.iterdir()
        if p.is_dir() and (m := pattern.match(p.name))
    ] if parent.exists() else []
    return max(existing, default=0) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="number of new splits to make")
    ap.add_argument("--figures-only", action="store_true",
                    help="(re)generate figures for all existing split_* dirs and exit; no splitting")
    ap.add_argument("--no-figures", action="store_true", help="skip figure generation")
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--out-parent", type=str, default=None)
    ap.add_argument("--ceiling", type=float, default=0.70,
                    help="target max Tanimoto of a val molecule to train (guaranteed where feasible)")
    ap.add_argument("--val-frac", type=float, default=0.125,
                    help="fraction of EACH class held out to val")
    ap.add_argument("--max-unit-frac", type=float, default=0.005,
                    help="components larger than this fraction of N stay in train")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--row-block", type=int, default=2048)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global DATA_CSV, CACHE_DIR, SPLITS_PARENT
    if args.csv:
        DATA_CSV = Path(args.csv)
    if args.cache_dir:
        CACHE_DIR = Path(args.cache_dir)
    if args.out_parent:
        SPLITS_PARENT = Path(args.out_parent)

    # Figures-only: rebuild PNGs for existing splits from their clusters.csv, then exit.
    if args.figures_only:
        dirs = sorted(p for p in SPLITS_PARENT.glob("split_*")
                      if p.is_dir() and (p / "clusters.csv").exists())
        if not dirs:
            print(f"No split_*/clusters.csv found under {SPLITS_PARENT}")
            return
        for d in dirs:
            ceiling = args.ceiling
            meta_path = d / "split_meta.json"
            if meta_path.exists():
                ceiling = json.loads(meta_path.read_text()).get("ceiling", ceiling)
            print(f"  {d.name}")
            generate_figures(d, ceiling)
        print("\nDone.")
        return

    if args.n < 1:
        ap.error("--n must be >= 1 (or use --figures-only)")

    import torch
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Loading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)
    smiles_all = df["SMILES"].astype(str).tolist()
    pprop_all = df["pprop"].to_numpy()
    print(f"  {len(smiles_all):,} rows.")

    print("\n[1/3] Scaffolds + Morgan fingerprints")
    scaffolds, canon, packed_fps, valid = build_scaffolds_and_fps(
        smiles_all, args.radius, args.n_bits, args.n_jobs, args.force,
    )
    vidx = np.where(valid)[0]
    if len(vidx) < len(smiles_all):
        print(f"  Dropping {len(smiles_all) - len(vidx):,} unparseable SMILES.")
    dedup = (pd.DataFrame({"canon": canon[vidx], "pprop": pprop_all[vidx], "row": vidx})
             .sort_values("pprop", ascending=False, kind="stable")
             .drop_duplicates("canon", keep="first").sort_values("row"))
    keep = dedup["row"].to_numpy()
    if len(keep) < len(vidx):
        print(f"  Collapsing {len(vidx) - len(keep):,} duplicate molecules "
              f"-> {len(keep):,} unique (by canonical SMILES).")
    smiles_v = [canon[i] for i in keep]
    scaffold_v = np.array(["" if scaffolds[i] is None else scaffolds[i] for i in keep], dtype=object)
    pprop_v = pprop_all[keep]
    packed_v = packed_fps[keep]
    classes_v = pprop_class(pprop_v)
    n_total = len(smiles_v)

    print(f"\n[2/3] Similarity components (ceiling={args.ceiling:g})")
    labels = build_components(packed_v, args.ceiling, device, args.row_block, args.force)

    print(f"\n[3/3] Generating {args.n} split(s)  (val {args.val_frac:.1%} per class)")
    for _ in range(args.n):
        i = next_split_index(SPLITS_PARENT)
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        print(f"\n  split_{i}  (seed={seed})")

        split, val_count, targets = assemble_clean(
            labels, classes_v, args.val_frac, args.max_unit_frac, rng,
        )
        relaxed = best_effort_topup(
            split, packed_v, classes_v, val_count, targets, device, args.row_block, rng,
        )
        max_to_train = verify_split(packed_v, split, device, args.row_block)

        out_dir = SPLITS_PARENT / f"split_{i}"
        out_dir.mkdir(parents=True, exist_ok=False)
        for name in ("train", "val"):
            smis = pd.unique(np.array(smiles_v, dtype=object)[split == name]).tolist()
            (out_dir / f"{name}.smi").write_text("\n".join(map(str, smis)) + "\n")
        over = (max_to_train > args.ceiling + 1e-6) & (split == "val")
        pd.DataFrame({
            "smiles": smiles_v, "scaffold": scaffold_v, "pprop": pprop_v,
            "pprop_class": classes_v, "component": labels, "split": split,
            "max_tani_to_train": np.where(split == "val", max_to_train, np.nan),
            "over_ceiling": over,
        }).to_csv(out_dir / "clusters.csv", index=False)

        # per-class report
        n_val = int((split == "val").sum())
        per_class = {}
        print(f"    train={int((split == 'train').sum()):,}  val={n_val:,} "
              f"({n_val / n_total:.2%})")
        print("    class       val/total  (val%)   max->train   over{:.2f}  relaxed".format(args.ceiling))
        worst = 0.0
        for nm in CLASS_NAMES:
            m = classes_v == nm
            mv = m & (split == "val")
            nv, nt = int(mv.sum()), int(m.sum())
            cmax = float(max_to_train[mv].max()) if nv else 0.0
            nover = int(((max_to_train > args.ceiling + 1e-6) & mv).sum())
            nrel = int((relaxed & mv).sum())
            worst = max(worst, cmax)
            per_class[nm] = {"val": nv, "total": nt, "max_to_train": cmax,
                             "n_over_ceiling": nover, "n_relaxed": nrel}
            print(f"    {nm:8s} {nv:>7,}/{nt:>7,} ({100 * nv / max(nt, 1):4.1f}%)   "
                  f"{cmax:6.4f}      {nover:>5}     {nrel:>5}")
        total_over = int(over.sum())
        verdict = "all <= ceiling" if total_over == 0 else f"{total_over} over ceiling (high-class relaxation)"
        print(f"    overall max val->train = {worst:.4f}   [{verdict}]")

        meta = {
            "split_index": i, "seed": seed,
            "n_train": int((split == "train").sum()), "n_val": n_val,
            "val_frac_realized": n_val / n_total,
            "ceiling": args.ceiling, "val_frac_target": args.val_frac,
            "max_unit_frac": args.max_unit_frac,
            "overall_max_val_to_train": worst,
            "n_val_over_ceiling": total_over,
            "n_val_relaxed": int(relaxed.sum()),
            "per_class": per_class,
        }
        (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"    wrote {out_dir}")
        if not args.no_figures:
            generate_figures(out_dir, args.ceiling)

    print("\nDone.")


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`pProp_MLP` trains a neural net to predict **pProp** from a molecule's SMILES.
`pProp = log10(N / docking_rank)` with `N = 1.468e9`, so rank-1 → pProp ≈ 9.17 and
**higher pProp = rarer, more potent (elite) binder**. The model is a *classifier*
over pProp bins: `[0,4.5) [4.5,5.5) [5.5,6.5) [6.5,7) [7,7.5) [7.5,∞)`.

Data: `data/ampc_subset.csv` — columns `SMILES, score, pprop`; 206,296 rows,
**203,012 unique molecules** after canonical-SMILES dedup.

Two pieces of code: (1) **`src/make_splits.py`** builds the dissimilarity-aware
train/val splits (below); (2) the **MLP classifier + wandb sweep** (MiniMol
embeddings → MLP → pProp class), see *The MLP classifier + wandb sweep* near the
end.

## Environment / commands

- Main project venv: `.venv/bin/python ...` (rdkit, scanpy, torch+CUDA, umap,
  leidenalg, scikit-learn, wandb). 3× RTX A6000, 128 cores. Used for splits AND
  for training/sweeps. `uv`-managed (no `pip`; use `uv pip ...`).
- Separate **`.venv_minimol`** venv for MiniMol featurization ONLY (graphium
  conflicts with the main torch stack). Pinned torch 2.6.0+cu124 / scipy 1.10 /
  setuptools<81 — see *MiniMol environment* below. Only `src/featurize_minimol.py`
  uses it.
- Generate `k` train/val splits: `.venv/bin/python src/make_splits.py --n <k>`
  (override paths with `--csv --cache-dir --out-parent`; `--force` recomputes
  caches; `--gpu` picks the GPU).
- Featurize all molecules once: `.venv_minimol/bin/python src/featurize_minimol.py`
- Train one model: `.venv/bin/python src/sweep_train.py --split_dir data/split_3 ...`
- Launch the wandb sweep on SLURM: `sbatch launch_sweep.sh` (after `wandb sweep
  sweeps/sweep.yaml`).

## The train/val split (`src/make_splits.py`)

Writes `data/split_{i}/` (next free index,
**never overwrites**): `train.smi`, `val.smi` (one canonical SMILES per line,
deduped), `clusters.csv`, `split_meta.json`, and 8 figures (see Figures below).

### Goal
A validation set that **spans every pProp class** AND keeps every val molecule's
max ECFP Tanimoto to any train molecule **≤ `--ceiling` (default 0.70)** wherever
feasible, relaxing the ceiling only for high-pProp classes that are too tightly
clustered to hold out cleanly (at the default 12.5% target this fallback is not
triggered — all classes come out clean).

### How the ≤0.70 guarantee works
Build **single-linkage connected components of the ">ceiling" Tanimoto graph over
ALL molecules**. A molecule and all its >ceiling neighbors share one component, so
assigning whole components to val/train atomically can never separate a val
molecule from a close neighbor → any held-out component is guaranteed ≤ ceiling to
train.

### Assembly (per split, per-split RNG seed → distinct splits)
1. **Per-class target** = `--val-frac` × |class| (val mirrors the class mix).
2. **Stratified clean holdout:** greedily hold out whole *small* components
   (size ≤ `--max-unit-frac · N`), **scarce-class first** (7.5+ → 0-4.5), until
   each class hits its target. The percolating giant component exceeds the cap and
   stays in train. → guaranteed-clean val members spanning all classes.
3. **Best-effort fallback** (`best_effort_topup`): if a class still can't reach its
   target from clean components, move that class's *least-train-similar* leftover
   molecules into val. These MAY exceed the ceiling; they're flagged
   (`over_ceiling`, `n_relaxed`) and reported per class. Not hit at 12.5%.
4. **Brute-force GPU verification** (`verify_split`): max val→train Tanimoto for
   every val molecule, independent of the component construction.

### CLI defaults
`--ceiling 0.70 --val-frac 0.125 --max-unit-frac 0.005 --seed 42`.

### Caching (`data/cache/`)
- Param-independent, computed once: `scaffolds.pkl` (incl. canonical SMILES),
  `mol_fps_r2_b2048.npy` (bit-packed FPs), `valid.npy`.
- `simcomp_all_c{ceiling}.npy`: the >ceiling components, keyed by ceiling.
Each step loads its cache if present, else computes and saves; `--force` ignores.

## Why it's built this way — history, decisions, trade-offs

The design evolved through two rejected approaches. A future instance must not
relitigate these:

1. **Scaffold-cluster holdout was the original spec and is infeasible.** The first
   implementation clustered scaffolds (Morgan→kNN→UMAP→Leiden) and held out whole
   clusters under a Tanimoto ceiling (adapted from
   `/home/ethan2/GrowthNet/scripts/make_splits.py`). It **cannot guarantee a
   per-molecule ceiling**: clustering is on *scaffolds* but the constraint is on
   *full-molecule* similarity, and in this analog-dense docking library many
   molecules have a >0.60–0.70 twin under a different Bemis–Murcko scaffold. A
   resolution sweep (15→5→2) showed the ineligible fraction **plateaus ~25%** —
   resolution is NOT a usable knob — and the Monte-Carlo hit 40k/40k rejects at
   every resolution. → **Decision:** make the holdout unit the **similarity graph
   itself** (connected components), turning the ceiling into a guarantee.
   **Trade-off:** we drop "novel scaffold family" semantics — a scaffold may be
   split across train/val as long as the ≤0.70 bar holds.

2. **A class-conditional strict/random split (interim) was replaced.** A middle
   design kept strict ≤0.70 only for pProp∈[4.5,7.0) and randomly held out the
   percolating ≥7.0 tail. It worked but left val **class-skewed** (~74% the
   4.5–5.5 class; ~3.5% of the 0–4.5 class) and gave the rare classes a small,
   partly-random val. → **Decision (current):** **stratified per-class holdout** —
   target `--val-frac` of *each* class so val spans all classes proportionally.

3. **Ceiling = 0.70, chosen by the user.** Stricter (0.60) leaves ~67% of
   high-pProp molecules unable to enter any clean val; 0.70 (the reference's own
   value) frees ~70% and is still a strong dissimilarity bar. **Trade-off:** weaker
   gap than 0.60, but it makes a clean, class-spanning val achievable.

4. **The high classes percolate, but it no longer forces relaxation.** The >0.70
   graph has one giant component (~7.3% of N) that traps 30–50% of the high-pProp
   mass; it goes to train. Crucially, with **all-molecule seeding** each class
   still has *more* cleanly-holdoutable molecules than a 12.5% target needs (even
   7.5+: all 46 sit in small components). So at the default target every class is
   held out cleanly at ≤0.70 (verified: 0 over ceiling). The `best_effort_topup`
   path (allowing >0.70 for a class that can't be met cleanly) exists for higher
   `--val-frac` or tighter `--ceiling`; when used it is reported per class.
   **Trade-off:** the giant blob in train means val under-samples the *specific*
   tightly-clustered elite analog series, not the classes themselves.

5. **Duplicate molecules leak across train/val.** Individually-placed molecules
   meant two copies of one molecule could split (3,284 exact dup rows; ~3% of
   SMILES non-canonical). → **Decision:** canonicalize + dedup to unique molecules
   up front (keep max-pProp copy of conflicts). After any change, check
   `comm -12 <(sort -u train.smi) <(sort -u val.smi)` returns 0.

## Scale notes
A full N×N Tanimoto matrix (~170 GB at this N) is infeasible; all similarity work
runs in GPU row-blocks — graph edges in `build_components`, candidate scoring in
`best_effort_topup`, verification in `verify_split`.

## The MLP classifier + wandb sweep

Predicts the pProp class from SMILES: **SMILES → MiniMol 512-d embedding → MLP →
6-class softmax**. Featurization is decoupled from training (embed once, reuse
across all sweep runs); the training/sweep code never imports MiniMol.

### Files (`src/`, `sweeps/`, repo root)
- **`featurize_minimol.py`** (runs in `.venv_minimol`): embeds every unique
  canonical molecule once → `data/cache/minimol_embeddings.npy` (203012×512 f32)
  + `minimol_smiles.txt` (aligned). `--force` recomputes.
- **`data_utils.py`**: CSV → canonical SMILES → max-pProp dedup → class labels
  (reuses `make_splits.CLASS_EDGES`/`pprop_class`); reads a split's
  `train.smi`/`val.smi`; gathers `(embedding, label)` arrays and **asserts val
  per-class counts match `split_meta.json`**.
- **`model.py`**: `MLP` with **LayerNorm** + `load_checkpoint()`.
- **`losses.py`**: `inverse_frequency_weights(y_train, n_classes)` only.
- **`metrics.py`**: one-vs-rest per-class + support-weighted + macro AUC/AP.
- **`sweep_train.py`**: one training run / the sweep program. AdamW +
  CosineAnnealingLR (`T_max=epochs`, `eta_min=1e-8`), weighted-CE loss, full-set
  metrics each epoch, early stopping + best checkpoint.
- **`sweeps/sweep.yaml`**: wandb Bayesian sweep config.
- **`launch_sweep.sh`**: SLURM array + MPS launcher (`python -m wandb agent ...`).

### Key design decisions
- **LayerNorm, not BatchNorm.** `batch_size` is a swept hyperparameter (can be
  small) and the classes are extreme-imbalanced, so BatchNorm's per-batch stats
  would be noisy and its train/eval behavior would interact with the sweep.
  LayerNorm is batch-size-independent and identical train/eval.
- **Loss = weighted cross-entropy, inverse class frequency** (`w_c = N/(C·n_c)`,
  normalized to mean 1, computed from the train split). User chose to keep it
  simple — no focal / alternative weighting schemes. Weights are logged to the run
  config.
- **Objective = best val MACRO AP.** The sweep maximizes `best_val_macro_ap` (the
  running max over epochs, i.e. the best epoch not the last). Macro = unweighted
  mean AP across all 6 classes, so the rare high-pProp classes count equally
  (support-weighted AP is majority-dominated and barely moves on rare-class gains).
  Early stopping and the saved checkpoint also track val macro AP. **Caveat:** the
  rarest val classes are tiny (7-7.5: 13 mols, 7.5+: 12), so macro AP is
  high-variance epoch-to-epoch and run-to-run.
- **Swept hyperparameters** (7): `n_layers, hidden_dim, init_lr, weight_decay,
  dropout, batch_size, epochs`. Constants: `split_dir` (default `data/split_3`,
  the 0.65-ceiling split), `patience`, `eta_min`.

### Metrics logged to wandb (per epoch, train + val, on the whole set)
`{train,val}/loss`, `.../weighted_auc`, `.../weighted_ap`, `.../macro_auc`,
`.../macro_ap`, and per class `.../auc/pprop_<range>`, `.../ap/pprop_<range>`
(range = the pProp bin, e.g. `pprop_5.5-6.5`). Run summary (sortable columns):
`best_val_macro_ap`, `best_train_macro_ap`, `best_val_weighted_ap`, `best_epoch`.
Each run also writes `runs/<run_id>/best_model.pt` (+ `best_meta.json`).

### Running the sweep (SLURM + MPS)
1. `.venv/bin/wandb sweep sweeps/sweep.yaml` → prints `<entity>/pprop-mlp-minimol/<id>`.
2. Put `<id>` in `launch_sweep.sh` (or `SWEEP_ID=<id> sbatch launch_sweep.sh`).
3. `sbatch launch_sweep.sh` — array of agents, each `--gres=mps:20` (20% of a GPU).

wandb project `pprop-mlp-minimol` auto-creates; the user (`ethan_personal`) is
already logged in (`~/.netrc`, shared on the cluster).

### MiniMol environment (`.venv_minimol`)
MiniMol → `graphium==2.4.7` conflicts with the main torch 2.6 stack, so it lives
in its own venv. Non-obvious pins (each fixes a real break): `torch==2.6.0+cu124`
(a plain install upgrades torch to a cu13 build the driver can't run),
`torch-scatter/sparse/cluster` from the pyg `torch-2.6.0+cu124` wheel index,
`scipy==1.10.1` + `numpy==1.26.4` (newer scipy rejects graphium's float16 sparse
adjacency), `setuptools<81` (graphium imports the removed `pkg_resources`). MiniMol
API: `Minimol()(list_of_smiles)` → list of (512,) float32 tensors.

## Figures (`generate_figures`)
Each split gets **8 PNGs**, written automatically and rebuildable from
`clusters.csv` without re-splitting via `--figures-only` (applies to all existing
`data/split_*/`):
- `val_sim_to_train__all.png` and one per class
  (`val_sim_to_train__{0-4.5,4.5-5.5,5.5-6.5,6.5-7,7-7.5,7.5+}.png`): histogram of
  per-val-molecule max Tanimoto to train, annotated with n / mean / median / max
  and the ceiling line.
- `val_class_counts.png`: per-class val count (log scale) annotated with the
  count, the proportion of the class held out, and the class total.

`clusters.csv` carries `max_tani_to_train` and `over_ceiling` per val molecule;
`split_meta.json` carries per-class stats.

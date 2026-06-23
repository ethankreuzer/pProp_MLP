# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`pProp_MLP` trains a neural net to predict **pProp** from a molecule's SMILES.
`pProp = log10(N / docking_rank)` with `N = 1.468e9`, so rank-1 → pProp ≈ 9.17 and
**higher pProp = rarer, more potent (elite) binder**. The planned model is a
*classifier* over pProp bins: `[0,4.5) [4.5,5.5) [5.5,6.5) [6.5,7) [7,7.5) [7.5,∞)`.

Data: `data/ampc_subset.csv` — columns `SMILES, score, pprop`; 206,296 rows,
**203,012 unique molecules** after canonical-SMILES dedup.

## Environment / commands

- Use the project venv: `.venv/bin/python ...` (rdkit, scanpy, torch+CUDA, umap,
  leidenalg, scikit-learn). 3× RTX A6000, 128 cores.
- Generate `k` train/val splits: `.venv/bin/python src/make_splits.py --n <k>`
- Override paths for experiments: `--csv --cache-dir --out-parent`. Recompute
  caches with `--force`. Pick GPU with `--gpu`.

## The train/val split (`src/make_splits.py`)

The only substantial code so far. Writes `data/split_{i}/` (next free index,
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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`pProp_MLP` trains a neural net to predict **pProp** from a molecule's SMILES.
`pProp = log10(N / docking_rank)` with `N = 1.468e9`, so rank-1 → pProp ≈ 9.17 and
**higher pProp = rarer, more potent (elite) binder**. The model is a **dual-head
MLP** (`src/model.py`) over MiniMol embeddings: a 4-class classifier over pProp bins
`[0,3.5) [3.5,5) [5,7.5) [7.5,∞)` **plus** a scalar regression
head predicting continuous pProp (raw, or a train-set normalization chosen by
`--pprop_norm`). Both heads are trained jointly (see Model & training below).

Data: `data/ampc_subset.csv` — columns `SMILES, score, pprop`; 631,296 rows,
**623,835 unique molecules** after canonical-SMILES dedup. Class sizes:
454,450 / 155,561 / 13,778 / 46.

## Environment / commands

- Use the project venv: `.venv/bin/python ...` (rdkit, scanpy, torch+CUDA, umap,
  leidenalg, scikit-learn). 3× RTX A6000, 128 cores.
- Generate `k` train/val splits: `.venv/bin/python src/make_splits.py --n <k>`
- Override paths for experiments: `--csv --cache-dir --out-parent`. Recompute
  caches with `--force`. Pick GPU with `--gpu`.

## Model & training (`src/model.py`, `src/sweep_train.py`, `src/losses.py`)

**Input.** 512-d **MiniMol** embeddings, precomputed once by
`src/featurize_minimol.py` in a *separate* venv (`.venv_minimol/bin/python`, isolated
because graphium/MiniMol conflicts with the main env) → `data/cache/minimol_embeddings.npy`
+ `minimol_smiles.txt`. `src/data_utils.py` maps a split's `train.smi`/`val.smi` to
`(embedding, class_idx, continuous pProp)` — bins imported from `make_splits.py` so
targets match the split exactly.

**Optional ECFP input.** `--use_ecfp 1` column-concatenates the 2048-d Morgan
fingerprint onto MiniMol (→ 2560-d input). The FP is *not* recomputed: it is reused
from make_splits' packed cache (`mol_fps_r2_b2048.npy`), unpacked and aligned to the
unique-molecule set by canonical SMILES in `data_utils.load_ecfp_features`, then
concatenated by `build_split_arrays(extra_features=…)`. `in_dim` adapts
automatically; `use_ecfp` and a `feature_set` tag are stored in the run config +
checkpoint. **Concat is plain (unnormalized)** and the trunk's LayerNorm is *after*
the first Linear — but the consequence is that **ECFP is under-weighted, not
dominant**: measured, MiniMol carries ~90% of the first Linear's input energy
(dense, per-sample L2 ≈ 20.4) vs ECFP's ~10% (2.2% density, ≈45 bits on, L2 ≈ 6.7),
while ECFP occupies 80% of the input width and of that layer's parameters. **A single
input LayerNorm over the concat is the wrong fix** (it shares per-sample stats across
two non-commensurable blocks; per-raw-block LN overshoots the other way — LN puts any
d-dim block at L2 √d, so ECFP would dominate ~4:1 purely on dims, 2048 vs 512).
AdamW's per-parameter scaling makes the init-scale imbalance largely self-correct;
the live mechanism is **weight-decay fairness** (ECFP weights must run 3–5× larger,
and decoupled WD penalizes that uniformly) — which is severe at `weight_decay=0.1`
and absent at `1e-6`. **`use_ecfp` is therefore PINNED, not swept:** compare the
feature sets with two separate equal-budget sweeps (`[0]` and `[1]`) so the ECFP arm
can select its own weight decay. Full reasoning + the proposed two-tower fix (if ever
needed): `docs/ecfp_concat.md`.

> **Branch `minimol-ecfp-twotower`.** That two-tower fix is implemented on this
> branch: `model.TwoTowerDualHeadMLP` projects MiniMol and ECFP *separately*
> (`Linear→LN→ReLU`) then concatenates before the trunk, with **independent swept
> projection widths** `proj_dim_minimol` / `proj_dim_ecfp`. It also makes the ECFP
> **Morgan radius sweepable** (`ecfp_radius ∈ {2,3,4}`): ECFP is precomputed
> per-radius by `src/featurize_ecfp.py` → `data/cache/ecfp_r{r}_b{nbits}.npy`, loaded
> via `data_utils.load_ecfp_precomputed` (not the main-branch `load_ecfp_features`).
> `use_ecfp` is pinned `[1]` in `sweeps/sweep.yaml`, project
> `pprop-mlp-minimol-ecfp-twotower`. Checkpoints carry `arch="two_tower"` +
> `minimol_dim`/`ecfp_dim`/`proj_dim_*`, and `load_checkpoint` dispatches on that tag.
> **Precompute every swept radius before launching** (`src/featurize_ecfp.py --radii
> 2 3 4`), or runs sampling a missing radius crash at load.
>
> **Memory design (so 5 agents still fit per A6000 at `--gres=mps:20`).** The 2048-d
> ECFP block ~5×'d the per-run GPU footprint vs MiniMol-only, causing OOMs. Two fixes:
> (1) **`forward()` takes MiniMol and ECFP as *separate* tensors** — `model(x_minimol,
> x_ecfp)` — and the ECFP block lives resident on the GPU as **uint8** (binary → the
> per-batch `.float()` upcast in the ECFP tower is lossless), 4× smaller than fp32.
> `load_data` returns `X_train`/`X_val` (MiniMol f32) plus `E_train`/`E_val` (ECFP
> uint8, or `None` when `use_ecfp=0`). (2) **`evaluate()`/`get_preds()` chunk the
> full-set forward** (`forward_full`, `EVAL_CHUNK`) so activation memory is per-chunk,
> not O(N·width) — metrics stay exact, not subsampled. Net: resident ~6.4→2.6 GB; the
> full-set eval spike is gone (~3 GB), so the binding per-run peak is now the training
> step (~7 GB at the widest/deepest config with `batch_size=10000`, measured). 5 agents
> fit per 47.5 GB A6000 at `--gres=mps:20` (~42 GB even if all 5 max out at once).
> **Note:** the two-tensor `forward` signature is a breaking
> change for any external caller (e.g. `sweeps/eval_best_model.ipynb`) that still does
> `model(X)` on a concatenated tensor.

**Optional target normalization.** `--pprop_norm {none,zscore,minmax}` (default
`none`) makes the **regression head predict a normalized pProp** instead of raw
pProp (`src/normalization.py`). Stats come from the **train split only**: `zscore` =
`(x−mean)/std`, `minmax` = `(x−min)/max` (≈ standard min-max since `min(pprop)≈0`).
The three regression loss terms train against the normalized target, but
`evaluate()` **denormalizes predictions before the metrics** (`compute_class_mae_metrics`,
`compute_class_pearson_metrics`) so MAE/Pearson/`goal_metric` stay on the raw pProp scale
and remain comparable across strategies — critical because MAE enters `goal_metric`
directly and minmax would otherwise shrink it ~9×. `norm_stats` + `pprop_norm` are stored in
the run config + checkpoint. Swept via `pprop_norm: [none, zscore, minmax]`. Only the
regression target is affected — the classifier, `CLASS_EDGES`, and bins are untouched.
**Caveat:** `sweeps/eval_best_model.ipynb` does *not* yet denormalize — a normalized
checkpoint renders every regression figure there in normalized units (read
`ckpt["norm_stats"]` + `denormalize_pprop` to fix). Swept `huber_delta`/`w_pair`/`w_std`
ranges are in target units and assume the raw scale.

**Model.** `DualHeadMLP` — shared trunk `(Linear→LayerNorm→ReLU→Dropout) × n_layers`,
then a **4-class classification head** (→ `n_classes` logits, passed explicitly —
never hardcode it) and a **scalar regression head**
(→ 1). LayerNorm not BatchNorm (batch size is swept and small; class imbalance
454450 vs 46 makes per-batch BN stats unstable).

**Loss (current).** The combined loss is
```
loss = w_cls · cls_loss  +  huber  +  w_pair · pair  +  w_std · std
```
- `cls_loss` — **grouped-inverse-frequency-weighted** cross-entropy
  (`grouped_frequency_weights`, groups = `WEIGHT_GROUPS`).
- `huber` — per-sample **grouped-inverse-frequency-weighted Huber** on continuous pProp
  (`weighted_huber_loss`, delta = `huber_delta`). Robust replacement for the old MSE;
  **grounded at weight 1** — the only term anchoring the *absolute* pProp level.
- `pair` — **unweighted** pairwise-distance loss (`pairwise_distance_loss`):
  `mean_{i,j} |(pred_i−pred_j) − (target_i−target_j)|` over all in-batch pairs.
  Shift-invariant; matches relative pProp *gaps*.
- `std` — **unweighted** std-matching (`std_match_loss`): `(std(pred) − std(target))²`.
  Counters regression-to-the-mean.

`pair` + `std` are relative "shape" terms added to fight the variance-shrinkage that
plain MSE/Huber induce. Class weighting is kept only on the two per-sample
value-fitting terms (`cls`, `huber`); `pair` and `std` are left unweighted.

**Why grouped, not per-class, inverse frequency (`WEIGHT_GROUPS = [0,1,2,0]`).** Plain
inverse *class* frequency conflates "rare" with "important" and **inverts on `7.5+`**:
those 46 molecules are **artifacts** (high pProp, null hit-rate) — the class we care
about *least* — yet as the rarest class they'd get the largest weight. So `7.5+`
shares the **null-hit-rate group** with `0-3.5`, and weights come from inverse *group*
frequency, then broadcast to member classes (`grouped_frequency_weights`, normalized so
present-class weights average 1 — same scale as before, so `w_cls`/`huber_delta` sweep
ranges are unchanged). Effect (normalized, mean 1): `[0.0004, 0.0012, 0.013, 3.99]` →
`[0.11, 0.31, 3.48, 0.11]` — `7.5+` drops to the null floor and `5-7.5` (the class of
interest) becomes the **top-weighted** class (was ~300× *below* `7.5+`). The 7.5-boundary
weight discontinuity only touches the ~10-11 elite molecules in train — negligible.
**Watch:** down-weighting `7.5+` in CE lets the classifier fold artifacts into `5-7.5` —
inspect the 5-7.5↔7.5+ confusion / per-class AP.

**Swept loss hyperparameters** (`sweeps/sweep.yaml`, log-uniform): `w_cls`, `w_pair`,
`w_std`, `huber_delta`. The Huber weight itself is fixed at 1 (see above); only its
`delta` is swept.

**Eval / OOM guard.** `evaluate()` scores the *full* train (~200K) and val sets each
epoch. `cls`/`huber`/`std` and all metrics are O(N) and exact; the pairwise term is
O(N²) so its reported value is estimated on a **fixed seeded 4096-point subsample**
(`PAIR_EVAL_K`) — the loss scalar is an estimate, the metrics are not.

**Metrics** (`src/metrics.py`). Only objective-relevant metrics are **tracked** (logged
to wandb). Classification: the two average-precision aggregates over `OBJECTIVE_CLASSES`
— `macro_ap_obj` (unweighted class mean, rare-balanced) and `weighted_ap_obj` (support-
weighted, overall) — plus per-class AP (`ap/pprop_*`, keeping the excluded `7.5+` class
visible). Regression MAE and **Pearson**, each over the whole set two ways — plain
(`mae`/`pearson`, every point weighted 1) and **grouped-inverse-frequency-weighted**
(`weighted_mae`/`pearson_weighted`, weight_i = 1/n_group(class(i)), `groups=WEIGHT_GROUPS`)
— plus per-class MAE. MAE also becomes a bounded **skill score**
(`mae_skill`/`weighted_mae_skill` = `1 − MAE/MAE_ref`, `MAE_ref` = same-weighting MAE of
the constant **median** predictor), which is what enters the objective so MAE is
commensurable with AP/Pearson; raw `mae`/`weighted_mae` are kept only for pProp-unit
interpretability. The grouped weighting matches training, so the objective metrics
de-weight the `7.5+` artifacts (they used to *dominate* the weighted ones at ~9879×; now
they sit at the null floor). Per-epoch sub-losses `huber/pair_loss/std_loss`, the objective
blends `ap_star/pearson_star/mae_skill_star`, and the two-task decomposition
`goal_term/{cls,reg}` (which sum to `goal_metric`) are all logged for both splits so the
objective's balance is visible directly. **ROC-AUC and the 4-class `macro_ap` are computed
by `compute_metrics` but no longer tracked** — AUC is misleading on the 454k-vs-46
imbalance (the objective is AP-only) and `macro_ap` is superseded by `macro_ap_obj`.

**Sweep objective** (`sweeps/sweep.yaml`, maximize) at the **final** epoch (no early
stopping — the final-epoch model is saved to `runs/<run_id>/final_model.pt` +
`final_meta.json`):
```
AP*        = ½(macro_ap_obj + weighted_ap_obj)     # rare-balanced + overall precision
Pearson*   = ½(pearson + pearson_weighted)         # unweighted + group-weighted correlation
MAE_skill* = ½(mae_skill + weighted_mae_skill)     # median-baseline skill, both flavors
val/goal_metric = AP*  +  ½(Pearson* + MAE_skill*)
```
**Equal-weighted classification and regression** (each task contributes one unit — no `3×`);
within each, the weighted and unweighted flavors are averaged so the model must do well on
both. MAE enters only as the bounded, unitless skill score (the raw `weighted_mae` is *not*
subtracted — it isn't commensurable). All objective classes are `OBJECTIVE_CLASSES =
{0-3.5, 3.5-5, 5-7.5}`: the `7.5+` artifact class is still scored and its per-class AP
reported (`ap/pprop_7.5+`), but **excluded from selection**. **`goal_metric`
values are NOT comparable across the objective's revisions** (old `3·macro_ap+…`, the
interim `3·macro_ap_obj+…`, and this one) — compare only runs whose config logs the same
objective (e.g. presence of `weighted_ap_obj`/`mae_skill` + `weight_groups`).

**Run it.** Single run:
`.venv/bin/python src/sweep_train.py --split_dir data/split_1 --w_cls … --w_pair … --w_std … --huber_delta … [--use_ecfp 0|1] [--pprop_norm none|zscore|minmax] --gpu 0`.
Sweep: `.venv/bin/wandb sweep sweeps/sweep.yaml` → `sbatch launch_sweep.sh SWEEP_ID=<id>`
(SLURM array packs several wandb agents per GPU via NVIDIA MPS).

**Stale `runs/` — the mismatch is SILENT, not a crash.** Everything under `runs/`
predates the 4-class rebin + the 3× dataset growth: those checkpoints are 6-class
(`['0-4.5','4.5-5.5','5.5-6.5','6.5-7','7-7.5','7.5+']`). Critically, the splits they
trained on were **regenerated in place at the same paths** — `data/split_1` … `split_8`
all still exist, but now hold 4-class splits. So an old checkpoint's
`config["split_dir"]` still resolves, and **nothing errors**: `_verify_against_meta`
passes (it compares today's 4-class labels against a now-4-class `split_meta.json`),
then the 6-class head scores 4-class bins and prints a plausible-looking table of wrong
numbers. Do not read any result produced this way.

`sweeps/eval_best_model.ipynb` and `sweeps/data_scaling.ipynb` pin a `RUN_ID` from that
era (`eval_best_model.ipynb` cell 3: `yqtzz3a4`) and resolve `split_dir` from its
config, so both need a **new RUN_ID from a post-rebin sweep** before use — they will not
warn you. `load_checkpoint` still loads old checkpoints (it reads `n_classes` from the
checkpoint, defaulting to 6), but their metrics are not comparable to new ones: 6 bins
vs 4, on ⅓ the data, on different splits. Old runs also share the wandb project with new
ones and log the same aggregate names (`val/goal_metric`, `val/macro_ap`) with different
meanings — filter on `n_classes == 4` when comparing.

## The train/val split (`src/make_splits.py`)

Writes `data/split_{i}/` (next free index,
**never overwrites**): `train.smi`, `val.smi` (one canonical SMILES per line,
deduped), `clusters.csv`, `split_meta.json`, and 6 figures (see Figures below).

### Goal
A validation set that **spans every pProp class** AND keeps every val molecule's
max ECFP Tanimoto to any train molecule **≤ `--ceiling` (default 0.65)** wherever
feasible, relaxing the ceiling only for high-pProp classes that are too tightly
clustered to hold out cleanly (at the `VAL_TARGETS` defaults this fallback is not
triggered — all classes come out clean, verified 0 over ceiling / 0 relaxed).

### How the ≤0.65 guarantee works
Build **single-linkage connected components of the ">ceiling" Tanimoto graph over
ALL molecules**. A molecule and all its >ceiling neighbors share one component, so
assigning whole components to val/train atomically can never separate a val
molecule from a close neighbor → any held-out component is guaranteed ≤ ceiling to
train.

### Assembly (per split, per-split RNG seed → distinct splits)
1. **Per-class target** = the `VAL_TARGETS` dict in `make_splits.py` (there is no
   `--val-frac` flag; a single scalar can't express the spec). A value `< 1` is a
   **fraction** of that class, a value `≥ 1` is an **absolute count**;
   `resolve_val_targets()` turns the spec into counts:
   `{"0-3.5": 0.15, "3.5-5": 0.15, "5-7.5": 0.12, "7.5+": 35}`
   → 68,168 / 23,334 / 1,653 / 35.
2. **Stratified clean holdout:** greedily hold out whole *small* components
   (size ≤ `--max-unit-frac · N`), **scarce-class first** (7.5+ → 0-3.5), until
   each class hits its target. The percolating giant component exceeds the cap and
   stays in train. → guaranteed-clean val members spanning all classes.
   Realized counts may slightly **overshoot** a target, since components are atomic
   and a component held out for one class can carry members of another (e.g. 7.5+
   lands on 35–37, and 5-7.5 on 12.0–12.4%). Harmless and still clean.
3. **Best-effort fallback** (`best_effort_topup`): if a class still can't reach its
   target from clean components, move that class's *least-train-similar* leftover
   molecules into val. These MAY exceed the ceiling; they're flagged
   (`over_ceiling`, `n_relaxed`) and reported per class. Not hit at the current targets.
4. **Brute-force GPU verification** (`verify_split`): max val→train Tanimoto for
   every val molecule, independent of the component construction.

### CLI defaults
`--ceiling 0.65 --max-unit-frac 0.005 --seed 42`. Val targets are **not** CLI
flags — edit `VAL_TARGETS` in `src/make_splits.py`.

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
   split across train/val as long as the ≤ceiling bar holds.

2. **A class-conditional strict/random split (interim) was replaced.** A middle
   design kept a strict ceiling only for the mid-pProp range and randomly held out
   the percolating high tail. It worked but left val **class-skewed** and gave the
   rare classes a small, partly-random val. → **Decision (current):** **stratified
   per-class holdout** — a per-class target (`VAL_TARGETS`) so val spans all classes.

3. **Ceiling = 0.65, chosen by the user** (was 0.70). A stricter generalization bar.
   See item 5 for the cost this imposes on the 5-7.5 class.

4. **Bins are 4 classes, chosen by the user** (was 6: `0-4.5 / 4.5-5.5 / 5.5-6.5 /
   6.5-7 / 7-7.5 / 7.5+`). The old scheme put 93% of molecules in one class and left
   three unusably-small middle classes (3,932 / 306 / 94). The 4-class scheme splits
   the bulk 73%/25% and merges the middle into one workable class of 13,778.
   **Note:** overall imbalance barely improves (12,617× → 9,879×) because the 46-molecule
   7.5+ class sets that ratio either way.

5. **5-7.5 gets 12%, not 15% — this is a hard constraint, do not "fix" it.**
   Measured on the current data: the >0.65 graph has a giant component of **140,904
   (22.6% of N)** — far worse percolation than 0.70, whose giant is only 22,761 (3.65%).
   **11,563 of the 13,778 molecules in 5-7.5 (83.9%) sit inside that blob** and must
   stay in train to preserve the guarantee, leaving only **1,842 (13.37%)** cleanly
   holdoutable. A 15% target (2,067) is therefore *physically impossible at 0.65*
   without relaxing the ceiling; 12% (1,653) keeps headroom under that cap. Raising
   `--max-unit-frac` does nothing until 0.05, which would swallow a ~31k-molecule
   component atomically. → **Decision:** keep the hard 0.65 guarantee, accept 12%.
   (At 0.70 this class is 41.9% holdoutable and 15% would be fine — that is the
   trade being made.)

6. **7.5+ is an absolute 35 of 46, chosen by the user.** All 46 sit in small
   components at 0.65 (29 are singletons; the rest in components of size 2/3/4/7),
   so 35 is cleanly holdoutable — verified. **Consequence:** only **~10-11 elite
   molecules remain in train**, at very high inverse-frequency weight. Deliberate:
   it buys a meaningful val signal for the rarest class. Watch for overfitting /
   instability on this class.

7. **The blob in train means** val under-samples the *specific* tightly-clustered
   elite analog series, not the classes themselves.

8. **Duplicate molecules leak across train/val.** Individually-placed molecules
   meant two copies of one molecule could split (7,461 duplicate molecules collapse
   at the current dataset size). → **Decision:** canonicalize + dedup to unique molecules
   up front (keep max-pProp copy of conflicts). After any change, check
   `comm -12 <(sort -u train.smi) <(sort -u val.smi)` returns 0.

## Scale notes
A full N×N Tanimoto matrix (~170 GB at this N) is infeasible; all similarity work
runs in GPU row-blocks — graph edges in `build_components`, candidate scoring in
`best_effort_topup`, verification in `verify_split`.

## Figures (`generate_figures`)
Each split gets **6 PNGs** (`len(CLASS_NAMES) + 2`, so this count tracks the bins),
written automatically and rebuildable from
`clusters.csv` without re-splitting via `--figures-only` (applies to all existing
`data/split_*/`):
- `val_sim_to_train__all.png` and one per class
  (`val_sim_to_train__{0-3.5,3.5-5,5-7.5,7.5+}.png`): histogram of
  per-val-molecule max Tanimoto to train, annotated with n / mean / median / max
  and the ceiling line.
- `val_class_counts.png`: per-class val count (log scale) annotated with the
  count, the proportion of the class held out, and the class total.

`clusters.csv` carries `max_tani_to_train` and `over_ceiling` per val molecule;
`split_meta.json` carries per-class stats.

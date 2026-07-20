# Feeding MiniMol + ECFP: what the concatenation actually does

`--use_ecfp 1` column-concatenates the 2048-d Morgan fingerprint onto the 512-d
MiniMol embedding, giving the trunk a plain 2560-d input. This note records what
that does to the first layer, what the fix would be if we need one, why the fix may
not be needed at all, and why MiniMol-vs-ECFP is compared with two separate sweeps
rather than a swept `use_ecfp`.

## The issue

The two blocks live on completely different scales, and nothing equalizes them
before they meet. The trunk's `LayerNorm` sits *after* the first `Linear`
(`src/model.py:45`), so the raw concatenation is what that `Linear` sees.

Measured on the real caches in `data/cache/`:

| block   | dims | density                     | per-sample L2 | ‖x‖² into first Linear |
|---------|------|-----------------------------|---------------|------------------------|
| MiniMol | 512  | dense (mean 0.76, std 0.49) | 20.4          | ≈ 416 (**90%**)        |
| ECFP    | 2048 | 2.2% (≈45 of 2048 bits on)  | 6.7           | ≈ 45 (**10%**)         |

**ECFP is under-weighted, not dominant.** This is the opposite of the intuition that
a sparse binary block will overwhelm an un-normalized `Linear`. The fingerprint
contributes about 10% of the first layer's input energy while occupying 80% of the
input width and 80% of that layer's parameters. At initialization the network is
close to ignoring it.

## Why the obvious fix is wrong

A single input `LayerNorm` over the 2560-d concatenation is the natural-looking
move, and it is a bad one. LayerNorm computes one per-sample mean and variance
across *all* 2560 features, i.e. across both blocks at once — so it injects
MiniMol's per-sample scale into the fingerprint encoding. The two blocks are not
commensurable, and sharing normalization statistics between them is meaningless: how
a molecule's fingerprint gets encoded would depend on the spread of its MiniMol
embedding.

Normalizing each *raw* block separately doesn't equalize them either. LayerNorm gives
each dimension unit variance, so any d-dimensional block emerges at L2 √d regardless
of its distribution — measured: MiniMol lands at 22.63 (√512 = 22.63) and ECFP at
45.24 (√2048 = 45.25). ECFP would then carry **4× the energy** of MiniMol purely from
having 4× the dimensions. Note what this is *not* — the overshoot has nothing to do
with sparsity or binarity; it is the dimension count, full stop. Trading a 9:1
imbalance for a 1:4 imbalance is not progress.

## The fix, if we need one

Project each block first, normalize *after*:

```
minimol (512) --> Linear(512 -> h)  --> LN --> ReLU --\
                                                       concat --> existing trunk
ecfp    (2048) -> Linear(2048 -> h) --> LN --> ReLU --/
```

This is precisely the thing per-block LayerNorm fails at. Projecting both blocks to a
**common width `h`** before normalizing means each emerges at an equal L2 √h — the
4:1 dimension advantage is gone by construction, and each block also gets a
fan-in-appropriate initialization. `use_ecfp=0` degenerates to just the MiniMol
tower, which keeps that path comparable to earlier runs.

The cost worth knowing up front: this breaks the `in_dim`-only checkpoint contract.
`load_checkpoint` (`src/model.py:59`) rebuilds the model from `ckpt["in_dim"]` alone,
so the per-tower dimensions would have to be stored in the checkpoint too.

## Why the payoff is uncertain: AdamW

The scale imbalance matters less than it looks, because **AdamW is per-parameter
scale-adaptive**. If ECFP's input scale is ~3× smaller, the gradients on its weights
are ~3× smaller — but Adam's update is `g/√v`, which is invariant to the scale of
`g`. It takes the *same-sized step* on those weights regardless. The
"under-weighted at initialization" problem therefore largely self-corrects during
training; the weights just grow, at the same rate anything else would.

What does **not** self-correct is **weight-decay fairness**. To express the same
function through a 3–5× smaller input scale, the ECFP weights must sit 3–5× larger
than the MiniMol weights. Decoupled AdamW weight decay penalizes that uniformly and
quadratically, so there is a systematic bias against the fingerprint block that no
amount of training removes.

That bias is real, but it is *hyperparameter-dependent*: severe at the sweep's
`weight_decay` ceiling of 0.1, and essentially absent at its floor of 1e-6. Which is
what makes the experiment design, not the architecture, the thing to fix first.

## Why two separate sweeps

`use_ecfp` was previously a swept parameter (`values: [0, 1]`). It is now pinned, and
the two feature sets are compared by running two sweeps of equal budget. Two reasons,
the second being the important one:

**Bayes cannot serve both arms.** A single bayes sweep shares one hyperparameter
posterior across both feature sets, which assumes hyperparameters transfer across the
switch. They don't — ECFP wants a different `weight_decay` and a wider `hidden_dim`.
Bayes exploits, so a few bad draws in the ECFP arm are enough for it to stop sampling
there, long before it reaches the region where ECFP is actually good.

**A pinned ECFP sweep fixes the problem the rewrite was for.** Given its own sweep,
the ECFP arm selects its own low weight decay — which neutralizes most of the
weight-decay-fairness bias described above. The cheap experiment may therefore make
the two-tower rewrite unnecessary. And if it doesn't, it is the baseline the rewrite
has to beat, so it is worth running either way.

**A confound that has now been removed.** `sweeps/sweep.yaml` contained a typo,
`values: [0, I1]`. wandb passed `--use_ecfp=I1`, argparse's `type=int` raised, and
the run died before training. **Every ECFP arm of that sweep crashed at startup** —
so any earlier MiniMol-vs-ECFP result from it reflects a crash, not a trained model,
and should be discarded. Confirm the arm trains before spending budget on it:

```
.venv/bin/python src/sweep_train.py --split_dir data/split_1 --use_ecfp 1 --epochs 1 --gpu 0
```

Expect `in_dim=2560` in the wandb config and a `final_model.pt` tagged
`feature_set="minimol+ecfp"`.

## Status

No model change has been made. The plan is to run the MiniMol-only sweep, then a
separate sweep with MiniMol+ECFP pinned on. The architecture question above is
reopened only if ECFP underperforms *with its own hyperparameters selected*.

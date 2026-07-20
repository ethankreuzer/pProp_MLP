#!/usr/bin/env python
"""
Train one dual-head MLP (one sweep run) on MiniMol embeddings.

The model has a shared trunk, a 4-class classification head, and a scalar
regression head. The combined loss is:

    loss = w_cls * cls_loss + huber + w_pair * pair + w_std * std

where
  cls_loss : inverse-frequency-weighted cross-entropy (classification),
  huber    : inverse-frequency-weighted Huber on continuous pProp (grounded at
             weight 1 -- the only term anchoring the absolute pProp level),
  pair     : unweighted pairwise-distance loss (predicted vs true pProp gaps),
  std      : unweighted std-matching loss (prediction spread vs target spread).

`w_cls`, `w_pair`, `w_std`, and the Huber `huber_delta` are swept hyperparameters;
the Huber weight is fixed at 1. `pair` and `std` fight the variance-shrinkage that
plain MSE/Huber induce.

Classification (average-precision) and regression (`pearson`/`mae` and their
group-weighted forms, plus median-baseline MAE skill scores) metrics are tracked for
both splits. The sweep objective is `val/goal_metric` = AP* + 0.5*(Pearson* +
MAE_skill*) at the final epoch: equal-weighted classification and regression, each
metric blended across its weighted + unweighted flavor (AP* over OBJECTIVE_CLASSES,
7.5+ excluded). See CLAUDE.md "Sweep objective".

Input features are 512-d MiniMol embeddings, optionally concatenated with the
2048-d ECFP fingerprint (`--use_ecfp 1`, reused from make_splits' cache) for a
2560-d input; `in_dim` adapts automatically and is recorded in the checkpoint.

The regression target can be normalized with `--pprop_norm {none,zscore,minmax}`
(stats from the train split only; `none` = raw pProp, the default). The head then
predicts the normalized value, but eval denormalizes predictions before metrics so
MAE/Pearson/goal_metric stay on the raw scale; `norm_stats` is saved in the checkpoint.

Run directly for a single run, or as the program a `wandb agent` launches:

    .venv/bin/python src/sweep_train.py --split_dir data/split_1 --epochs 50 ...
    .venv/bin/python src/sweep_train.py --split_dir data/split_1 --use_ecfp 1 ...
    .venv/bin/python src/sweep_train.py --split_dir data/split_1 --pprop_norm zscore ...
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import wandb
from data_utils import CACHE_DIR
from make_splits import OBJECTIVE_CLASSES, WEIGHT_GROUPS
from losses import (
    grouped_frequency_weights,
    pairwise_distance_loss,
    sample_weights_from_classes,
    std_match_loss,
    weighted_huber_loss,
)
from metrics import (
    compute_class_mae_metrics,
    compute_class_pearson_metrics,
    compute_metrics,
)
from model import DualHeadMLP, TwoTowerDualHeadMLP
from normalization import compute_norm_stats, denormalize_pprop, normalize_pprop

EMB_PATH = CACHE_DIR / "minimol_embeddings.npy"
SMI_PATH = CACHE_DIR / "minimol_smiles.txt"
PROJECT_DEFAULT = "pprop-mlp-minimol-multitask"


def parse_args():
    ap = argparse.ArgumentParser()
    # data / infra (not swept)
    ap.add_argument("--split_dir", default="data/split_1")
    ap.add_argument("--project", default=PROJECT_DEFAULT)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # input features: MiniMol (always on) + optional ECFP concatenation.
    # int-valued (0/1) rather than store_true so it drops into sweep.yaml as
    # `values: [0, 1]` for a single MiniMol-vs-MiniMol+ECFP comparison sweep.
    ap.add_argument("--use_ecfp", type=int, default=0, choices=(0, 1),
                    help="1 = add the ECFP tower alongside MiniMol; 0 = MiniMol only.")
    # two-tower ECFP input: each block is projected separately (Linear->LN->ReLU)
    # then concatenated (see model.TwoTowerDualHeadMLP / docs/ecfp_concat.md).
    ap.add_argument("--proj_dim_minimol", type=int, default=256,
                    help="Projection width of the MiniMol tower.")
    ap.add_argument("--proj_dim_ecfp", type=int, default=256,
                    help="Projection width of the ECFP tower (ignored if use_ecfp=0).")
    ap.add_argument("--ecfp_radius", type=int, default=2,
                    help="Morgan radius of the precomputed ECFP cache to load "
                         "(must exist: src/featurize_ecfp.py --radii ...).")
    ap.add_argument("--ecfp_nbits", type=int, default=2048,
                    help="Morgan fingerprint bit length of the ECFP cache to load.")
    # regression target normalization (train-set stats). "none" reproduces the
    # pre-normalization behavior byte-for-byte; swept like use_ecfp in sweep.yaml.
    ap.add_argument("--pprop_norm", default="none",
                    choices=("none", "zscore", "minmax"),
                    help="Normalize the regression target: none | zscore "
                         "(x-mean)/std | minmax (x-min)/max. Stats from train.")
    # trunk hyperparameters
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--hidden_dim", type=int, default=256)
    # classification head hyperparameters
    ap.add_argument("--cls_n_layers", type=int, default=1)
    ap.add_argument("--cls_hidden_dim", type=int, default=64)
    # regression head hyperparameters
    ap.add_argument("--reg_n_layers", type=int, default=0)
    ap.add_argument("--reg_hidden_dim", type=int, default=64)
    # loss balance: total = w_cls*cls + huber + w_pair*pair + w_std*std
    ap.add_argument("--w_cls", type=float, default=1.0,
                    help="Scale factor on the classification (weighted CE) term.")
    ap.add_argument("--w_pair", type=float, default=1.0,
                    help="Scale factor on the pairwise-distance term.")
    ap.add_argument("--w_std", type=float, default=1.0,
                    help="Scale factor on the std-matching term.")
    ap.add_argument("--huber_delta", type=float, default=1.0,
                    help="Huber transition point (L2->L1) for the regression term.")
    ap.add_argument("--init_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    # training control (constants in the sweep)
    ap.add_argument("--eta_min", type=float, default=1e-8)
    return ap.parse_args()


def load_data(split_dir, device, use_ecfp=False, ecfp_radius=2, ecfp_nbits=2048):
    """Load cached embeddings (+ optional per-radius ECFP) + a split into GPU tensors.

    Returns minimol_dim / ecfp_dim (the split point in the concatenated X) so the
    two-tower model can slice the two blocks apart. ecfp_dim is 0 when use_ecfp=0.
    """
    from data_utils import build_split_arrays, load_ecfp_precomputed

    if not EMB_PATH.exists():
        raise FileNotFoundError(
            f"{EMB_PATH} not found. Run `.venv_minimol/bin/python "
            "src/featurize_minimol.py` first."
        )
    embeddings = np.load(EMB_PATH)
    smiles_index = SMI_PATH.read_text().splitlines()
    minimol_dim = embeddings.shape[1]
    extra_features = (
        [load_ecfp_precomputed(ecfp_radius, ecfp_nbits)] if use_ecfp else None
    )
    data = build_split_arrays(split_dir, embeddings, smiles_index,
                              return_pprop=True, extra_features=extra_features)

    # build_split_arrays returns X = [MiniMol | ECFP] (float32). Split the two
    # blocks so the ECFP block can live on the GPU as uint8: it is binary, so the
    # cast is lossless, and it shrinks the resident copy 4x (2048 f32 -> 2048 u8).
    # The ECFP tower upcasts each batch back to float32 (model.TwoTowerDualHeadMLP).
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    Xm_train = data["X_train"][:, :minimol_dim]
    Xm_val = data["X_val"][:, :minimol_dim]
    if use_ecfp:
        E_train = t(data["X_train"][:, minimol_dim:].astype(np.uint8), torch.uint8)
        E_val = t(data["X_val"][:, minimol_dim:].astype(np.uint8), torch.uint8)
        ecfp_dim = data["X_train"].shape[1] - minimol_dim
    else:
        E_train = E_val = None
        ecfp_dim = 0

    return {
        "X_train": t(Xm_train, torch.float32),
        "E_train": E_train,
        "y_train": t(data["y_train"], torch.long),
        "pprop_train": t(data["pprop_train"], torch.float32),
        "X_val": t(Xm_val, torch.float32),
        "E_val": E_val,
        "y_val": t(data["y_val"], torch.long),
        "pprop_val": t(data["pprop_val"], torch.float32),
        "class_names": data["class_names"],
        "in_dim": minimol_dim + ecfp_dim,
        "minimol_dim": minimol_dim,
        "ecfp_dim": ecfp_dim,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """Build a matplotlib confusion matrix figure (not saved to disk)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    return fig


# Full-set eval forwards the whole train/val set through the model. Doing it in
# one call materializes O(N x width) activations (~10 GB at N=530k); chunking it
# caps activations at O(chunk x width) and frees them between chunks, so the peak
# is per-chunk. Only the small per-row OUTPUTS (logits N x C, reg N x 1) are kept
# and concatenated. Metrics stay exact (this is not a subsample).
EVAL_CHUNK = 16384


@torch.no_grad()
def forward_full(model, X, E, chunk=EVAL_CHUNK):
    """Run model over the full set in chunks; return (cls_logits, reg_out) concatenated."""
    model.eval()
    n = X.shape[0]
    cls_parts, reg_parts = [], []
    for s in range(0, n, chunk):
        xe = E[s:s + chunk] if E is not None else None
        cl, ro = model(X[s:s + chunk], xe)
        cls_parts.append(cl)
        reg_parts.append(ro)
    return torch.cat(cls_parts, dim=0), torch.cat(reg_parts, dim=0)


@torch.no_grad()
def get_preds(model, X, E):
    """Return argmax class predictions from the classification head."""
    cls_logits, _ = forward_full(model, X, E)
    return cls_logits.argmax(dim=1).cpu().numpy()


# The pairwise term is O(N^2); at the full train set (~200K) that is infeasible,
# so the reported eval `pair` loss is estimated on a fixed random subsample of
# points (deterministic seed -> comparable across epochs). Metrics stay exact.
PAIR_EVAL_K = 4096


@torch.no_grad()
def evaluate(model, X, E, y, pprop, class_weights, w_cls, w_pair, w_std,
             huber_delta, class_names, device, norm_stats):
    """Full-set loss + classification metrics + per-class regression metrics.

    The regression head predicts a *normalized* target, so the loss terms
    (huber/pair/std) are computed against the normalized `pprop` — matching what
    training optimizes — while the metrics (MAE, Pearson) are computed on the raw
    pProp scale by denormalizing the predictions. `pprop` is passed in raw.

    X is the MiniMol block (float32); E is the ECFP block (uint8) or None. The
    forward is chunked (forward_full) so activation memory stays per-chunk.
    """
    model.eval()
    cls_logits, reg_out = forward_full(model, X, E)
    pred = reg_out.squeeze(-1)
    pprop_norm = normalize_pprop(pprop, norm_stats)
    cw = class_weights.to(device)
    cls_loss = F.cross_entropy(cls_logits, y, weight=cw).item()
    sample_w = sample_weights_from_classes(y, cw)
    huber = weighted_huber_loss(reg_out, pprop_norm, sample_w, delta=huber_delta).item()
    std_loss = std_match_loss(pred, pprop_norm).item()

    # Pairwise on a fixed random subsample (O(N^2) is infeasible on the full set).
    n = X.shape[0]
    if n > PAIR_EVAL_K:
        g = torch.Generator(device=device).manual_seed(0)
        sub = torch.randperm(n, generator=g, device=device)[:PAIR_EVAL_K]
        pair = pairwise_distance_loss(pred[sub], pprop_norm[sub]).item()
    else:
        pair = pairwise_distance_loss(pred, pprop_norm).item()

    reg_loss = huber + w_pair * pair + w_std * std_loss
    total_loss = w_cls * cls_loss + reg_loss

    y_np = y.cpu().numpy()
    probs = F.softmax(cls_logits, dim=1).cpu().numpy()
    # Metrics use raw-pProp predictions (denormalized) vs raw targets so MAE /
    # Pearson / goal_metric stay on one comparable scale across normalizations.
    pred_pprop = denormalize_pprop(pred, norm_stats).cpu().numpy()
    pprop_np = pprop.cpu().numpy()
    m = compute_metrics(y_np, probs, class_names, objective_classes=OBJECTIVE_CLASSES)
    m.update(compute_class_mae_metrics(pprop_np, pred_pprop, y_np, class_names,
                                       groups=WEIGHT_GROUPS))
    m.update(compute_class_pearson_metrics(pprop_np, pred_pprop, y_np, class_names,
                                           groups=WEIGHT_GROUPS))
    m["loss"] = total_loss
    m["cls_loss"] = cls_loss
    m["reg_loss"] = reg_loss
    m["huber"] = huber
    m["pair_loss"] = pair
    m["std_loss"] = std_loss
    return m


def log_dict(split, m, class_names):
    """Flatten a metrics dict into legible wandb keys for one split."""
    # Objective (maximize): equal-weighted classification and regression (no 3x).
    # Each metric is blended across its weighted + unweighted flavors, and MAE is a
    # bounded median-baseline skill score so it is commensurable with AP / Pearson:
    #   AP*        = mean(macro_ap_obj, weighted_ap_obj)    # rare-balanced + overall precision
    #   Pearson*   = mean(pearson, pearson_weighted)        # unweighted + group-weighted corr
    #   MAE_skill* = mean(mae_skill, weighted_mae_skill)    # median-baseline skill, both flavors
    #   goal = AP* + 0.5*(Pearson* + MAE_skill*)
    # The two tasks (classification = AP*, regression = 0.5*(Pearson*+MAE_skill*))
    # each contribute one unit; goal_term/{cls,reg} sum to goal_metric.
    ap_star = 0.5 * (m["macro_ap_obj"] + m["weighted_ap_obj"])
    pearson_star = 0.5 * (m["pearson"] + m["pearson_weighted"])
    mae_skill_star = 0.5 * (m["mae_skill"] + m["weighted_mae_skill"])
    goal_cls = ap_star
    goal_reg = 0.5 * (pearson_star + mae_skill_star)
    goal_metric = goal_cls + goal_reg
    out = {
        f"{split}/loss": m["loss"],
        f"{split}/cls_loss": m["cls_loss"],
        f"{split}/reg_loss": m["reg_loss"],
        f"{split}/huber": m["huber"],
        f"{split}/pair_loss": m["pair_loss"],
        f"{split}/std_loss": m["std_loss"],
        f"{split}/goal_metric": goal_metric,
        # --- goal_metric decomposition: the two tasks sum to goal_metric ---
        f"{split}/goal_term/cls": goal_cls,   # AP*
        f"{split}/goal_term/reg": goal_reg,   # 0.5*(Pearson* + MAE_skill*)
        # --- task blends (each = mean of a weighted + unweighted flavor) ---
        f"{split}/ap_star": ap_star,
        f"{split}/pearson_star": pearson_star,
        f"{split}/mae_skill_star": mae_skill_star,
        # --- classification: the two AP flavors that make up AP* ---
        f"{split}/macro_ap_obj": m["macro_ap_obj"],
        f"{split}/weighted_ap_obj": m["weighted_ap_obj"],
        # --- regression: objective components + raw pProp-unit MAE for interpretability ---
        f"{split}/pearson": m["pearson"],
        f"{split}/pearson_weighted": m["pearson_weighted"],
        f"{split}/mae_skill": m["mae_skill"],
        f"{split}/weighted_mae_skill": m["weighted_mae_skill"],
        f"{split}/mae": m["mae"],
        f"{split}/weighted_mae": m["weighted_mae"],
    }
    # Per-class AP (keeps the excluded 7.5+ artifact class visible) and per-class MAE.
    for name in class_names:
        out[f"{split}/ap/pprop_{name}"] = m["ap"][name]
        out[f"{split}/mae/pprop_{name}"] = m["mae_per_class"][name]
    return out


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    defaults = dict(
        split_dir=args.split_dir, seed=args.seed, use_ecfp=args.use_ecfp,
        pprop_norm=args.pprop_norm,
        proj_dim_minimol=args.proj_dim_minimol, proj_dim_ecfp=args.proj_dim_ecfp,
        ecfp_radius=args.ecfp_radius, ecfp_nbits=args.ecfp_nbits,
        n_layers=args.n_layers, hidden_dim=args.hidden_dim,
        cls_n_layers=args.cls_n_layers, cls_hidden_dim=args.cls_hidden_dim,
        reg_n_layers=args.reg_n_layers, reg_hidden_dim=args.reg_hidden_dim,
        w_cls=args.w_cls, w_pair=args.w_pair, w_std=args.w_std,
        huber_delta=args.huber_delta,
        init_lr=args.init_lr, weight_decay=args.weight_decay,
        dropout=args.dropout, batch_size=args.batch_size, epochs=args.epochs,
        eta_min=args.eta_min,
    )
    wandb.init(project=args.project, config=defaults)
    cfg = wandb.config

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    data = load_data(cfg.split_dir, device, use_ecfp=bool(cfg.use_ecfp),
                     ecfp_radius=cfg.ecfp_radius, ecfp_nbits=cfg.ecfp_nbits)
    class_names = data["class_names"]
    X_train, y_train, pprop_train = data["X_train"], data["y_train"], data["pprop_train"]
    X_val, y_val, pprop_val = data["X_val"], data["y_val"], data["pprop_val"]
    E_train, E_val = data["E_train"], data["E_val"]   # ECFP blocks (uint8) or None
    n_classes = len(class_names)

    # Regression-target normalization: stats from TRAIN only. The head learns the
    # normalized target; eval denormalizes predictions before metrics so MAE /
    # Pearson / goal_metric stay on the raw pProp scale (comparable across
    # strategies). "none" -> identity, i.e. unchanged behavior.
    norm_stats = compute_norm_stats(pprop_train.cpu().numpy(), cfg.pprop_norm)
    pprop_train_norm = normalize_pprop(pprop_train, norm_stats)
    pprop_val_norm = normalize_pprop(pprop_val, norm_stats)

    model = TwoTowerDualHeadMLP(
        minimol_dim=data["minimol_dim"],
        ecfp_dim=data["ecfp_dim"],
        proj_dim_minimol=cfg.proj_dim_minimol,
        proj_dim_ecfp=cfg.proj_dim_ecfp,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        cls_hidden_dim=cfg.cls_hidden_dim,
        cls_n_layers=cfg.cls_n_layers,
        reg_hidden_dim=cfg.reg_hidden_dim,
        reg_n_layers=cfg.reg_n_layers,
        n_classes=n_classes,
    ).to(device)
    wandb.config.update(
        {"in_dim": data["in_dim"], "minimol_dim": data["minimol_dim"],
         "ecfp_dim": data["ecfp_dim"], "n_classes": n_classes,
         "task": "dual", "arch": "two_tower"},
        allow_val_change=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.init_lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min,
    )
    class_weights = grouped_frequency_weights(y_train.cpu().numpy(), n_classes,
                                              WEIGHT_GROUPS)
    class_weights_dev = class_weights.to(device)
    wandb.config.update(
        {
            "class_weights": [round(float(x), 4) for x in class_weights],
            "weight_groups": list(WEIGHT_GROUPS),
            "objective_classes": list(OBJECTIVE_CLASSES),
        },
        allow_val_change=True,
    )
    wandb.config.update({"norm_stats": norm_stats}, allow_val_change=True)
    cfg_dict = {k: v for k, v in cfg.items()}

    run_dir = Path(args.out_dir) / wandb.run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "final_model.pt"

    n_train = X_train.shape[0]
    train_m = val_m = None

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad()
            e_idx = E_train[idx] if E_train is not None else None
            cls_logits, reg_out = model(X_train[idx], e_idx)
            pred = reg_out.squeeze(-1)
            cls_loss = F.cross_entropy(cls_logits, y_train[idx], weight=class_weights_dev)
            sample_w = sample_weights_from_classes(y_train[idx], class_weights_dev)
            huber = weighted_huber_loss(reg_out, pprop_train_norm[idx], sample_w,
                                        delta=cfg.huber_delta)
            pair = pairwise_distance_loss(pred, pprop_train_norm[idx])
            std_loss = std_match_loss(pred, pprop_train_norm[idx])
            loss = (cfg.w_cls * cls_loss + huber
                    + cfg.w_pair * pair + cfg.w_std * std_loss)
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_m = evaluate(model, X_train, E_train, y_train, pprop_train, class_weights,
                           cfg.w_cls, cfg.w_pair, cfg.w_std, cfg.huber_delta,
                           class_names, device, norm_stats)
        val_m = evaluate(model, X_val, E_val, y_val, pprop_val, class_weights,
                         cfg.w_cls, cfg.w_pair, cfg.w_std, cfg.huber_delta,
                         class_names, device, norm_stats)

        log = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        log.update(log_dict("train", train_m, class_names))
        log.update(log_dict("val", val_m, class_names))
        wandb.log(log)

    # Save final model (after all training steps complete).
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg_dict,
            "class_names": class_names,
            "in_dim": data["in_dim"],
            "minimol_dim": data["minimol_dim"],
            "ecfp_dim": data["ecfp_dim"],
            "n_classes": n_classes,
            "arch": "two_tower",
            "feature_set": (
                f"minimol+ecfp_r{cfg.ecfp_radius}" if cfg.use_ecfp else "minimol"
            ),
            "norm_stats": norm_stats,
            "task": "dual",
            "epoch": cfg.epochs - 1,
            "val_goal_metric": log["val/goal_metric"],
            "val_ap_star": log["val/ap_star"],
            "val_pearson_star": log["val/pearson_star"],
            "val_mae_skill_star": log["val/mae_skill_star"],
        },
        ckpt_path,
    )
    wandb.save(str(ckpt_path))

    # Upload confusion matrices as static images (true = y-axis, predicted = x-axis).
    y_train_np = y_train.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    train_preds = get_preds(model, X_train, E_train)
    val_preds = get_preds(model, X_val, E_val)
    train_fig = plot_confusion_matrix(y_train_np, train_preds, class_names, "Train")
    val_fig = plot_confusion_matrix(y_val_np, val_preds, class_names, "Val")
    wandb.log({
        "train/confusion_matrix": wandb.Image(train_fig),
        "val/confusion_matrix": wandb.Image(val_fig),
    })
    plt.close("all")

    (run_dir / "final_meta.json").write_text(json.dumps(
        {
            "val_goal_metric": log["val/goal_metric"],
            "val_ap_star": log["val/ap_star"],
            "val_pearson_star": log["val/pearson_star"],
            "val_mae_skill_star": log["val/mae_skill_star"],
            "final_epoch": cfg.epochs - 1,
            "run_id": wandb.run.id,
            "config": cfg_dict,
        },
        indent=2,
    ))
    print(f"Done. val goal metric {log['val/goal_metric']:.4f} @ epoch {cfg.epochs - 1}. "
          f"Checkpoint: {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()

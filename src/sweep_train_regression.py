#!/usr/bin/env python
"""
Train one MLP regression model (one sweep run) on MiniMol embeddings.

Predicts pProp as a continuous scalar (MSE loss) instead of a class index
(cross-entropy). Everything else mirrors sweep_train.py exactly: same
architecture (MLP with LayerNorm), same inverse-class-frequency sample
weighting, same AdamW + CosineAnnealingLR schedule, same early-stopping
criterion (best val macro AP), same wandb logging keys.

Per-class AUC/AP for the regression output are computed via soft scores:
for class c with bin [lo_c, hi_c), the score for a predicted pProp p is
  score_c(p) = -max(0, lo_c - p) - max(0, p - hi_c)
which is 0 inside the bin and negative-proportional-to-distance outside.
This plugs directly into the same one-vs-rest AUC/AP pipeline as the
classifier, making `best_val_macro_ap` directly comparable across runs.

    .venv/bin/python src/sweep_train_regression.py --split_dir data/split_3 --epochs 50
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import wandb
from data_utils import CACHE_DIR, CLASS_EDGES, CLASS_NAMES
from losses import inverse_frequency_weights, sample_weights_from_classes, weighted_mse_loss
from metrics import (
    compute_correlation_metrics,
    compute_error_metrics,
    compute_regression_metrics,
    enrichment_factor,
)
from model import MLP

EMB_PATH = CACHE_DIR / "minimol_embeddings.npy"
SMI_PATH = CACHE_DIR / "minimol_smiles.txt"
PROJECT_DEFAULT = "pprop-mlp-minimol"
N_CLASSES = len(CLASS_NAMES)

# Enrichment Factor is tracked at these "active = pProp >= threshold" cutoffs,
# each at the top 1% of the model-ranked list (predicted pProp = ranking score).
EF_THRESHOLDS = [4.0, 4.5, 5.0, 5.5, 6.0]
EF_FRACTIONS = [0.01]

# Pearson/Spearman correlations split molecules into two groups by TRUE pProp:
# [pProp < edge) and [pProp >= edge). Wide groups avoid per-bin range restriction.
CORR_GROUP_EDGE = 5.0
CORR_GROUP_LT = f"0-{CORR_GROUP_EDGE:g}"   # low group label, e.g. "0-5"
CORR_GROUP_GE = f"{CORR_GROUP_EDGE:g}+"    # high group label, e.g. "5+"


def _fmt_thr(thr):
    return f"{thr:g}"


def _fmt_frac(frac):
    return f"{frac * 100:g}pct"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default="data/split_3")
    ap.add_argument("--project", default=PROJECT_DEFAULT)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--init_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--eta_min", type=float, default=1e-8)
    return ap.parse_args()


def load_data(split_dir, device):
    """Load cached embeddings + split into GPU tensors. Returns tensors + metadata."""
    from data_utils import build_split_arrays

    if not EMB_PATH.exists():
        raise FileNotFoundError(
            f"{EMB_PATH} not found. Run `.venv_minimol/bin/python "
            "src/featurize_minimol.py` first."
        )
    embeddings = np.load(EMB_PATH)
    smiles_index = SMI_PATH.read_text().splitlines()
    data = build_split_arrays(split_dir, embeddings, smiles_index, return_pprop=True)

    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return {
        "X_train": t(data["X_train"], torch.float32),
        "y_train": t(data["y_train"], torch.long),          # class indices, for weighting
        "pprop_train": t(data["pprop_train"], torch.float32),  # MSE targets
        "X_val": t(data["X_val"], torch.float32),
        "y_val": t(data["y_val"], torch.long),
        "pprop_val": t(data["pprop_val"], torch.float32),
        "class_names": data["class_names"],
        "in_dim": data["X_train"].shape[1],
    }


@torch.no_grad()
def evaluate(model, X, pprop_true, class_names):
    """Full-set unweighted MSE + regression metrics in eval mode (no dropout)."""
    model.eval()
    pred = model(X).squeeze(-1)
    loss = F.mse_loss(pred, pprop_true).item()
    pred_np = pred.cpu().numpy()
    pprop_np = pprop_true.cpu().numpy()
    m = compute_regression_metrics(pprop_np, pred_np, CLASS_EDGES, class_names)
    # Rank by predicted pProp (higher = predicted more elite) for enrichment.
    m["enrichment"] = enrichment_factor(pprop_np, pred_np, EF_THRESHOLDS, EF_FRACTIONS)
    # Pearson/Spearman; weighting + two-group split derive from true pProp only.
    m["correlation"] = compute_correlation_metrics(pprop_np, pred_np, CORR_GROUP_EDGE)
    # MAE (robust complement to the MSE loss), same two-group layout.
    m["error"] = compute_error_metrics(pprop_np, pred_np, CORR_GROUP_EDGE)
    m["loss"] = loss
    return m


def log_dict(split, m, class_names):
    """Flatten a metrics dict into legible wandb keys for one split."""
    out = {
        f"{split}/loss": m["loss"],
        f"{split}/weighted_auc": m["weighted_auc"],
        f"{split}/weighted_ap": m["weighted_ap"],
        f"{split}/macro_auc": m["macro_auc"],
        f"{split}/macro_ap": m["macro_ap"],
        f"{split}/bin_accuracy": m["bin_accuracy"],
    }
    for name in class_names:
        out[f"{split}/auc/pprop_{name}"] = m["auc"][name]
        out[f"{split}/ap/pprop_{name}"] = m["ap"][name]
    for (thr, frac), ef in m["enrichment"].items():
        out[f"{split}/enrichment/pprop{_fmt_thr(thr)}_top{_fmt_frac(frac)}"] = ef
    c = m["correlation"]
    out[f"{split}/pearson_unweighted"] = c["pearson"]
    out[f"{split}/pearson_weighted"] = c["pearson_weighted"]
    out[f"{split}/spearman_unweighted"] = c["spearman"]
    out[f"{split}/spearman_weighted"] = c["spearman_weighted"]
    out[f"{split}/pearson/pprop{CORR_GROUP_LT}"] = c["pearson_group_lt"]
    out[f"{split}/pearson/pprop{CORR_GROUP_GE}"] = c["pearson_group_ge"]
    out[f"{split}/spearman/pprop{CORR_GROUP_LT}"] = c["spearman_group_lt"]
    out[f"{split}/spearman/pprop{CORR_GROUP_GE}"] = c["spearman_group_ge"]
    e = m["error"]
    out[f"{split}/mae_unweighted"] = e["mae"]
    out[f"{split}/mae_weighted"] = e["mae_weighted"]
    out[f"{split}/mae/pprop{CORR_GROUP_LT}"] = e["mae_group_lt"]
    out[f"{split}/mae/pprop{CORR_GROUP_GE}"] = e["mae_group_ge"]
    return out


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    defaults = dict(
        task="regression",
        split_dir=args.split_dir, seed=args.seed,
        n_layers=args.n_layers, hidden_dim=args.hidden_dim,
        init_lr=args.init_lr, weight_decay=args.weight_decay,
        dropout=args.dropout, batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, eta_min=args.eta_min,
    )
    wandb.init(project=args.project, config=defaults)
    cfg = wandb.config

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    data = load_data(cfg.split_dir, device)
    class_names = data["class_names"]
    X_train, y_train = data["X_train"], data["y_train"]
    pprop_train = data["pprop_train"]
    X_val, pprop_val = data["X_val"], data["pprop_val"]

    # Single scalar output — no softmax; raw pProp prediction.
    model = MLP(
        in_dim=data["in_dim"], hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
        n_classes=1, dropout=cfg.dropout,
    ).to(device)
    wandb.config.update({"in_dim": data["in_dim"], "n_classes": 1},
                        allow_val_change=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.init_lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min,
    )

    # Same inverse-class-frequency weights as the classification model.
    class_weights = inverse_frequency_weights(y_train.cpu().numpy(), N_CLASSES).to(device)
    wandb.config.update(
        {"class_weights": [round(float(x), 4) for x in class_weights]},
        allow_val_change=True,
    )
    cfg_dict = {k: v for k, v in cfg.items()}

    run_dir = Path(args.out_dir) / wandb.run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_model.pt"

    best_val_macro_ap = -1.0
    best_train_macro_ap = -1.0
    best_val_weighted_ap = -1.0
    best_val_enrichment = {}  # val EF dict captured at the best epoch
    best_val_correlation = {}  # val correlation dict captured at the best epoch
    best_val_error = {}  # val MAE dict captured at the best epoch
    best_epoch = -1
    epochs_no_improve = 0
    n_train = X_train.shape[0]

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            sample_w = sample_weights_from_classes(y_train[idx], class_weights)
            optimizer.zero_grad()
            loss = weighted_mse_loss(model(X_train[idx]), pprop_train[idx], sample_w)
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_m = evaluate(model, X_train, pprop_train, class_names)
        val_m = evaluate(model, X_val, pprop_val, class_names)

        best_train_macro_ap = max(best_train_macro_ap, train_m["macro_ap"])
        best_val_weighted_ap = max(best_val_weighted_ap, val_m["weighted_ap"])

        val_macro_ap = val_m["macro_ap"]
        improved = val_macro_ap > best_val_macro_ap
        if improved:
            best_val_macro_ap = val_macro_ap
            best_val_enrichment = val_m["enrichment"]
            best_val_correlation = val_m["correlation"]
            best_val_error = val_m["error"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg_dict,
                    "class_names": class_names,
                    "in_dim": data["in_dim"],
                    "n_classes": 1,
                    "task": "regression",
                    "epoch": epoch,
                    "val_macro_ap": val_macro_ap,
                    "val_weighted_ap": val_m["weighted_ap"],
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        log = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
               "best_val_macro_ap": best_val_macro_ap,
               "best_train_macro_ap": best_train_macro_ap,
               "best_val_weighted_ap": best_val_weighted_ap}
        log.update(log_dict("train", train_m, class_names))
        log.update(log_dict("val", val_m, class_names))
        wandb.log(log)

        if epochs_no_improve >= cfg.patience:
            print(f"Early stop at epoch {epoch} "
                  f"(best val macro AP {best_val_macro_ap:.4f} @ epoch {best_epoch})")
            break

    wandb.summary["best_val_macro_ap"] = best_val_macro_ap
    wandb.summary["best_train_macro_ap"] = best_train_macro_ap
    wandb.summary["best_val_weighted_ap"] = best_val_weighted_ap
    wandb.summary["best_epoch"] = best_epoch
    # Val enrichment at the selected (best) epoch -> sortable summary columns.
    best_ef_flat = {
        f"best_val_ef/pprop{_fmt_thr(thr)}_top{_fmt_frac(frac)}": ef
        for (thr, frac), ef in best_val_enrichment.items()
    }
    for k, v in best_ef_flat.items():
        wandb.summary[k] = v
    # Val correlations at the selected (best) epoch -> summary columns.
    best_corr_flat = {
        "best_val_pearson_unweighted": best_val_correlation.get("pearson"),
        "best_val_pearson_weighted": best_val_correlation.get("pearson_weighted"),
        "best_val_spearman_unweighted": best_val_correlation.get("spearman"),
        "best_val_spearman_weighted": best_val_correlation.get("spearman_weighted"),
        f"best_val_pearson_pprop{CORR_GROUP_LT}": best_val_correlation.get("pearson_group_lt"),
        f"best_val_pearson_pprop{CORR_GROUP_GE}": best_val_correlation.get("pearson_group_ge"),
        f"best_val_spearman_pprop{CORR_GROUP_LT}": best_val_correlation.get("spearman_group_lt"),
        f"best_val_spearman_pprop{CORR_GROUP_GE}": best_val_correlation.get("spearman_group_ge"),
    }
    for k, v in best_corr_flat.items():
        if v is not None:
            wandb.summary[k] = v
    # Val MAE at the selected (best) epoch -> summary columns.
    best_mae_flat = {
        "best_val_mae_unweighted": best_val_error.get("mae"),
        "best_val_mae_weighted": best_val_error.get("mae_weighted"),
        f"best_val_mae_pprop{CORR_GROUP_LT}": best_val_error.get("mae_group_lt"),
        f"best_val_mae_pprop{CORR_GROUP_GE}": best_val_error.get("mae_group_ge"),
    }
    for k, v in best_mae_flat.items():
        if v is not None:
            wandb.summary[k] = v
    wandb.save(str(ckpt_path))
    (run_dir / "best_meta.json").write_text(json.dumps(
        {"best_val_macro_ap": best_val_macro_ap,
         "best_train_macro_ap": best_train_macro_ap,
         "best_val_weighted_ap": best_val_weighted_ap,
         "best_val_enrichment": best_ef_flat,
         "best_val_correlation": best_val_correlation,
         "best_val_error": best_val_error,
         "best_epoch": best_epoch,
         "run_id": wandb.run.id, "config": cfg_dict}, indent=2))
    print(f"Done. Best val macro AP {best_val_macro_ap:.4f} @ epoch {best_epoch}. "
          f"Checkpoint: {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()

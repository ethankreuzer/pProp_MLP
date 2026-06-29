#!/usr/bin/env python
"""
Train one MLP classifier (one sweep run) on MiniMol embeddings.

Run directly for a single run, or as the program a `wandb agent` launches for a
Bayesian sweep (sweeps/sweep.yaml). Either way every hyperparameter lands in
wandb.config and is recorded with the run.

Per epoch, on the WHOLE train set and the WHOLE val set, we log: cross-entropy
loss, support-weighted ROC-AUC, support-weighted average precision, and per-class
ROC-AUC / average precision (keyed by pProp range so train/val and the class are
both legible on wandb). Early stopping monitors val weighted AP and checkpoints
the best model. The sweep objective `best_val_weighted_ap` is the running max of
val weighted AP (the best epoch, not the last).

    .venv/bin/python src/sweep_train.py --split-dir data/split_3 --epochs 50 ...
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import wandb
from data_utils import CACHE_DIR
from losses import inverse_frequency_weights
from metrics import compute_metrics
from model import MLP

EMB_PATH = CACHE_DIR / "minimol_embeddings.npy"
SMI_PATH = CACHE_DIR / "minimol_smiles.txt"
PROJECT_DEFAULT = "pprop-mlp-minimol"


def parse_args():
    # Flags use underscores so they match the `--name=value` args a wandb sweep
    # agent forwards (parameter keys in sweep.yaml are underscored).
    ap = argparse.ArgumentParser()
    # data / infra (not swept; keep argparse defaults under the agent)
    ap.add_argument("--split_dir", default="data/split_3")
    ap.add_argument("--project", default=PROJECT_DEFAULT)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # swept hyperparameters (defaults used for non-sweep direct runs)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--init_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    # training control (constants in the sweep)
    ap.add_argument("--patience", type=int, default=25,
                    help="Early-stop after this many epochs without val "
                         "weighted-AP improvement.")
    ap.add_argument("--eta_min", type=float, default=1e-8,
                    help="Final (floor) LR of the cosine schedule.")
    return ap.parse_args()


def load_data(split_dir, device):
    """Load cached embeddings + a split into GPU tensors. Returns tensors+names."""
    from data_utils import build_split_arrays

    if not EMB_PATH.exists():
        raise FileNotFoundError(
            f"{EMB_PATH} not found. Run `.venv_minimol/bin/python "
            "src/featurize_minimol.py` first."
        )
    embeddings = np.load(EMB_PATH)
    smiles_index = SMI_PATH.read_text().splitlines()
    data = build_split_arrays(split_dir, embeddings, smiles_index)

    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return {
        "X_train": t(data["X_train"], torch.float32),
        "y_train": t(data["y_train"], torch.long),
        "X_val": t(data["X_val"], torch.float32),
        "y_val": t(data["y_val"], torch.long),
        "class_names": data["class_names"],
        "in_dim": data["X_train"].shape[1],
    }


@torch.no_grad()
def evaluate(model, X, y, class_names):
    """Full-set loss + metrics in eval mode (no dropout)."""
    model.eval()
    logits = model(X)
    loss = F.cross_entropy(logits, y).item()
    probs = F.softmax(logits, dim=1).cpu().numpy()
    m = compute_metrics(y.cpu().numpy(), probs, class_names)
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
    }
    for name in class_names:
        out[f"{split}/auc/pprop_{name}"] = m["auc"][name]
        out[f"{split}/ap/pprop_{name}"] = m["ap"][name]
    return out


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Defaults seed wandb.config; a sweep agent overrides the swept ones.
    defaults = dict(
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
    X_val, y_val = data["X_val"], data["y_val"]
    n_classes = len(class_names)

    model = MLP(
        in_dim=data["in_dim"], hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
        n_classes=n_classes, dropout=cfg.dropout,
    ).to(device)
    wandb.config.update({"in_dim": data["in_dim"], "n_classes": n_classes},
                        allow_val_change=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.init_lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min,
    )
    class_weights = inverse_frequency_weights(y_train.cpu().numpy(), n_classes)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    wandb.config.update(
        {"class_weights": [round(float(x), 4) for x in class_weights]},
        allow_val_change=True,
    )
    cfg_dict = {k: v for k, v in cfg.items()}  # plain dict for checkpoints

    run_dir = Path(args.out_dir) / wandb.run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_model.pt"

    # Objective + early stopping + checkpoint track val MACRO AP (the goal metric).
    best_val_macro_ap = -1.0
    best_train_macro_ap = -1.0
    best_val_weighted_ap = -1.0  # tracked for reference, not the objective
    best_epoch = -1
    epochs_no_improve = 0
    n_train = X_train.shape[0]

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_m = evaluate(model, X_train, y_train, class_names)
        val_m = evaluate(model, X_val, y_val, class_names)

        best_train_macro_ap = max(best_train_macro_ap, train_m["macro_ap"])
        best_val_weighted_ap = max(best_val_weighted_ap, val_m["weighted_ap"])

        val_macro_ap = val_m["macro_ap"]
        improved = val_macro_ap > best_val_macro_ap
        if improved:
            best_val_macro_ap = val_macro_ap
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg_dict,
                    "class_names": class_names,
                    "in_dim": data["in_dim"],
                    "n_classes": n_classes,
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
    wandb.save(str(ckpt_path))
    (run_dir / "best_meta.json").write_text(json.dumps(
        {"best_val_macro_ap": best_val_macro_ap,
         "best_train_macro_ap": best_train_macro_ap,
         "best_val_weighted_ap": best_val_weighted_ap,
         "best_epoch": best_epoch,
         "run_id": wandb.run.id, "config": cfg_dict}, indent=2))
    print(f"Done. Best val macro AP {best_val_macro_ap:.4f} @ epoch {best_epoch}. "
          f"Checkpoint: {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()

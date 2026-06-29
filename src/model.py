#!/usr/bin/env python
"""
MLP classifier over MiniMol embeddings.

Normalization: LayerNorm (not BatchNorm). Two reasons specific to this task:
  1. `batch_size` is a swept hyperparameter and can be small; BatchNorm's
     per-batch statistics get noisy at small batch sizes, and its train/eval
     behavior differs (running stats), which would interact with the sweep.
  2. The classes are extremely imbalanced (159k vs 46), so a minibatch's
     feature statistics are dominated by the majority class and vary run to run.
LayerNorm normalizes per sample across features, so it is independent of batch
size and identical in train vs eval — robust under both conditions.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, n_classes, dropout):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d = hidden_dim
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_checkpoint(path, device="cpu"):
    """
    Rebuild a trained MLP from a best_model.pt saved by sweep_train.py.

    Returns (model_in_eval_mode, checkpoint_dict). The checkpoint also carries
    `config`, `class_names`, `val_weighted_ap` and `epoch`.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MLP(
        in_dim=ckpt["in_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        n_classes=ckpt["n_classes"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt

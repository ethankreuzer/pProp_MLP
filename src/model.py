#!/usr/bin/env python
"""
Dual-head MLP for multi-task pProp classification + regression.

Architecture:
  Shared trunk : (Linear -> LayerNorm -> ReLU -> Dropout) x n_layers
  Cls head     : (Linear -> LayerNorm -> ReLU -> Dropout) x cls_n_layers -> Linear(*, n_classes)
  Reg head     : (Linear -> LayerNorm -> ReLU -> Dropout) x reg_n_layers -> Linear(*, 1)

If cls_n_layers or reg_n_layers is 0, that head is a single Linear projection from
the trunk output.

Normalization: LayerNorm (not BatchNorm). batch_size is a swept hyperparameter and
can be very small; class imbalance (454k vs 46) makes per-batch BN stats unstable
and biased toward the majority class. LayerNorm normalizes per sample across
features, so it is independent of batch size and identical in train vs eval.
"""

import torch
import torch.nn as nn


def _make_head(in_dim, hidden_dim, n_layers, out_dim, dropout):
    """Build a head block: n_layers of (Linear->LN->ReLU->Dropout) then a Linear."""
    if n_layers == 0:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim),
                   nn.ReLU(), nn.Dropout(dropout)]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class DualHeadMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, dropout,
                 cls_hidden_dim, cls_n_layers,
                 reg_hidden_dim, reg_n_layers, n_classes):
        super().__init__()
        trunk = []
        d = in_dim
        for _ in range(n_layers):
            trunk += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim),
                      nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.trunk = nn.Sequential(*trunk)
        trunk_out = hidden_dim if n_layers > 0 else in_dim

        self.cls_head = _make_head(trunk_out, cls_hidden_dim, cls_n_layers, n_classes, dropout)
        self.reg_head = _make_head(trunk_out, reg_hidden_dim, reg_n_layers, 1, dropout)

    def forward(self, x):
        h = self.trunk(x)
        return self.cls_head(h), self.reg_head(h)


def load_checkpoint(path, device="cpu"):
    """
    Rebuild a trained DualHeadMLP from a best_model.pt saved by sweep_train.py.

    Returns (model_in_eval_mode, checkpoint_dict).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DualHeadMLP(
        in_dim=ckpt["in_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        cls_hidden_dim=cfg["cls_hidden_dim"],
        cls_n_layers=cfg["cls_n_layers"],
        reg_hidden_dim=cfg["reg_hidden_dim"],
        reg_n_layers=cfg["reg_n_layers"],
        n_classes=ckpt.get("n_classes", 6),   # pre-4-class checkpoints predate the key
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt

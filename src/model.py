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


class TwoTowerDualHeadMLP(nn.Module):
    """
    Two-tower variant: project MiniMol and ECFP SEPARATELY (Linear -> LN -> ReLU),
    concat the projections, then run the same shared trunk + dual heads as
    DualHeadMLP. This is the fix sketched in docs/ecfp_concat.md: normalizing after
    a per-block projection (rather than concatenating the raw 512-d MiniMol with the
    raw 2048-d ECFP) keeps the two non-commensurable blocks on comparable scales
    before they meet the trunk.

    The two projection widths are independent hyperparameters (proj_dim_minimol,
    proj_dim_ecfp) -- deliberately NOT a single common width, so a sweep can explore
    unequal per-tower capacity.

    forward() takes the two blocks as SEPARATE tensors: x_minimol (float32) and
    x_ecfp (may be a compact uint8 tensor, upcast to float32 inside the ECFP tower
    per batch — the block is binary so this is lossless and keeps the resident copy
    4x smaller). If ecfp_dim == 0 / x_ecfp is None (use_ecfp off) the model
    degenerates to a single MiniMol tower.
    """

    def __init__(self, minimol_dim, ecfp_dim, proj_dim_minimol, proj_dim_ecfp,
                 hidden_dim, n_layers, dropout,
                 cls_hidden_dim, cls_n_layers,
                 reg_hidden_dim, reg_n_layers, n_classes):
        super().__init__()
        self.minimol_dim = minimol_dim
        self.ecfp_dim = ecfp_dim

        self.minimol_tower = nn.Sequential(
            nn.Linear(minimol_dim, proj_dim_minimol),
            nn.LayerNorm(proj_dim_minimol),
            nn.ReLU(),
        )
        concat_dim = proj_dim_minimol
        if ecfp_dim > 0:
            self.ecfp_tower = nn.Sequential(
                nn.Linear(ecfp_dim, proj_dim_ecfp),
                nn.LayerNorm(proj_dim_ecfp),
                nn.ReLU(),
            )
            concat_dim += proj_dim_ecfp
        else:
            self.ecfp_tower = None

        trunk = []
        d = concat_dim
        for _ in range(n_layers):
            trunk += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim),
                      nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.trunk = nn.Sequential(*trunk)
        trunk_out = hidden_dim if n_layers > 0 else concat_dim

        self.cls_head = _make_head(trunk_out, cls_hidden_dim, cls_n_layers, n_classes, dropout)
        self.reg_head = _make_head(trunk_out, reg_hidden_dim, reg_n_layers, 1, dropout)

    def forward(self, x_minimol, x_ecfp=None):
        z = self.minimol_tower(x_minimol)
        if self.ecfp_tower is not None:
            if x_ecfp is None:
                raise ValueError("ecfp_tower is present but x_ecfp was not provided")
            # x_ecfp may be uint8 (compact resident storage); upcast per batch.
            z = torch.cat([z, self.ecfp_tower(x_ecfp.float())], dim=1)
        h = self.trunk(z)
        return self.cls_head(h), self.reg_head(h)


def load_checkpoint(path, device="cpu"):
    """
    Rebuild a trained model from a checkpoint saved by sweep_train.py.

    Dispatches on the checkpoint's "arch" tag: "two_tower" ->
    TwoTowerDualHeadMLP (rebuilt from the stored per-tower dims, since the
    in_dim-only contract can't capture a two-tower geometry); otherwise the
    single-input DualHeadMLP (default; also covers pre-tag checkpoints).

    Returns (model_in_eval_mode, checkpoint_dict).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if ckpt.get("arch") == "two_tower":
        model = TwoTowerDualHeadMLP(
            minimol_dim=ckpt["minimol_dim"],
            ecfp_dim=ckpt["ecfp_dim"],
            proj_dim_minimol=cfg["proj_dim_minimol"],
            proj_dim_ecfp=cfg["proj_dim_ecfp"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            cls_hidden_dim=cfg["cls_hidden_dim"],
            cls_n_layers=cfg["cls_n_layers"],
            reg_hidden_dim=cfg["reg_hidden_dim"],
            reg_n_layers=cfg["reg_n_layers"],
            n_classes=ckpt.get("n_classes", 4),
        )
    else:
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

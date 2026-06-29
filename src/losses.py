#!/usr/bin/env python
"""
Inverse-class-frequency weighting for the pProp MLP classifier.

The pProp classes are extreme-imbalanced (train ~ [159547, 39087, 3932, 306, 94,
46]), so plain cross-entropy ignores the rare high-pProp classes. We weight the
loss by inverse class frequency so each class contributes comparably.
"""

import numpy as np
import torch


def inverse_frequency_weights(y_train, n_classes):
    """
    Per-class loss weights ∝ 1 / class frequency, i.e. w_c = N / (C * n_c),
    normalized so the present-class weights average 1 (keeps the loss scale
    stable). Empty classes get weight 0.
    """
    counts = np.bincount(np.asarray(y_train), minlength=n_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros(n_classes, dtype=np.float64)
    w[present] = counts.sum() / (present.sum() * counts[present])
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def sample_weights_from_classes(y_class, class_weights):
    """
    Map a per-class weight tensor to a per-sample weight tensor.

    Parameters
    ----------
    y_class      : (N,) long tensor of class indices
    class_weights: (C,) float tensor from inverse_frequency_weights

    Returns (N,) float tensor where entry i = class_weights[y_class[i]].
    """
    return class_weights[y_class]


def weighted_mse_loss(pred, target_pprop, sample_weights):
    """
    Weighted MSE loss normalized by the sum of sample weights (mirrors how
    nn.CrossEntropyLoss(weight=w) divides by Σ w_{y_i} rather than batch count).

    Parameters
    ----------
    pred          : (N,) or (N,1) float tensor of predicted pProp values
    target_pprop  : (N,) float tensor of true pProp values
    sample_weights: (N,) float tensor of per-sample weights
    """
    sq_err = (pred.squeeze(-1) - target_pprop) ** 2
    return (sample_weights * sq_err).sum() / sample_weights.sum()

#!/usr/bin/env python
"""
Inverse-class-frequency weighting for the pProp MLP.

The pProp classes are extreme-imbalanced (~ [454450, 155561, 13778, 46] over the
full set), so plain cross-entropy ignores the rare high-pProp classes. We weight
the loss by inverse class frequency so each class contributes comparably.
"""

import numpy as np
import torch


def inverse_frequency_weights(y_train, n_classes):
    """
    Per-class loss weights proportional to 1 / class frequency:
    w_c = N / (C * n_c), normalized so the present-class weights average 1
    (keeps the loss scale stable). Empty classes get weight 0.
    """
    counts = np.bincount(np.asarray(y_train), minlength=n_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros(n_classes, dtype=np.float64)
    w[present] = counts.sum() / (present.sum() * counts[present])
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def grouped_frequency_weights(y_train, n_classes, groups):
    """
    Inverse-frequency weights computed over *groups* of classes rather than over
    individual classes, then broadcast back to each member class.

    Plain per-class inverse frequency conflates "rare" with "important": the rarest
    class (7.5+, the artifacts) would get the largest weight even though it matters
    least. Grouping 7.5+ with the other null-hit-rate class (0-3.5) fixes the
    direction — classes in one group share a single weight
    `w_g = N / (G * n_g)` (n_g = total members of group g, G = number of present
    groups), so 7.5+ inherits the null-group floor while 5-7.5 rises to the top.

    Normalized so the present-class weights average 1 (same scale convention as
    `inverse_frequency_weights`, keeping loss scale / sweep ranges stable). Empty
    classes get weight 0.

    Parameters
    ----------
    y_train   : (N,) int array/tensor of class indices
    n_classes : int, number of classes C
    groups    : length-C sequence mapping class index -> group id (0..G-1)
    """
    counts = np.bincount(np.asarray(y_train), minlength=n_classes).astype(np.float64)
    groups = np.asarray(groups)
    n_groups = int(groups.max()) + 1
    group_counts = np.bincount(groups, weights=counts, minlength=n_groups)
    present_groups = group_counts > 0
    gw = np.zeros(n_groups, dtype=np.float64)
    gw[present_groups] = group_counts.sum() / (present_groups.sum() * group_counts[present_groups])
    w = gw[groups]                       # broadcast group weight to member classes
    present = counts > 0
    w[~present] = 0.0                     # a class with no members gets weight 0
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def group_sample_weights(y_class, n_classes, groups):
    """
    Per-sample class-balancing weights based on inverse *group* frequency, the
    numpy analogue of `grouped_frequency_weights` for the reported/objective
    metrics. Sample i in class c gets weight `1 / n_group(group(c))`, so members of
    the same group are balanced together (7.5+ shares 0-3.5's group, so both get the
    null-group weight). Only relative magnitudes matter — the normalization constant
    cancels in every weighted-mean the metrics compute.

    Parameters
    ----------
    y_class   : (N,) int array of class indices
    n_classes : int, number of classes C
    groups    : length-C sequence mapping class index -> group id
    """
    y_class = np.asarray(y_class, dtype=np.int64)
    groups = np.asarray(groups)
    support = np.bincount(y_class, minlength=n_classes).astype(np.float64)
    group_counts = np.bincount(groups, weights=support, minlength=int(groups.max()) + 1)
    return 1.0 / group_counts[groups[y_class]]


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
    nn.CrossEntropyLoss(weight=w) divides by sum(w_{y_i}) rather than batch count).

    Parameters
    ----------
    pred          : (N,) or (N,1) float tensor of predicted pProp values
    target_pprop  : (N,) float tensor of true pProp values
    sample_weights: (N,) float tensor of per-sample weights
    """
    sq_err = (pred.squeeze(-1) - target_pprop) ** 2
    return (sample_weights * sq_err).sum() / sample_weights.sum()


def weighted_huber_loss(pred, target_pprop, sample_weights, delta=1.0):
    """
    Weighted Huber (smooth-L1) loss, the robust analogue of weighted_mse_loss.

    Quadratic for residuals |r| <= delta, linear beyond, so large tail residuals
    contribute a bounded gradient. Normalized by the sum of sample weights
    (same convention as weighted_mse_loss / nn.CrossEntropyLoss(weight=w)).

    Parameters
    ----------
    pred          : (N,) or (N,1) float tensor of predicted pProp values
    target_pprop  : (N,) float tensor of true pProp values
    sample_weights: (N,) float tensor of per-sample weights
    delta         : float, Huber transition point (L2 -> L1)
    """
    hub = torch.nn.functional.huber_loss(
        pred.squeeze(-1), target_pprop, delta=delta, reduction="none"
    )
    return (sample_weights * hub).sum() / sample_weights.sum()


def pairwise_distance_loss(pred, target_pprop):
    """
    Distance-matching loss over all pairs (i, j) in a batch: the predicted pProp
    *gap* between two molecules should equal their true pProp gap. Encourages the
    model to reproduce relative differences (shift-invariant), countering the
    variance-shrinkage that plain MSE/Huber induce.

        L = mean_{i,j} | (pred_i - pred_j) - (target_i - target_j) |

    Unweighted. Vectorized over the full B x B difference matrix (the diagonal
    contributes 0; the pair-count constant is absorbed by the caller's weight).

    Parameters
    ----------
    pred          : (N,) or (N,1) float tensor of predicted pProp values
    target_pprop  : (N,) float tensor of true pProp values
    """
    pred = pred.squeeze(-1)
    d_pred = pred[:, None] - pred[None, :]
    d_true = target_pprop[:, None] - target_pprop[None, :]
    return (d_pred - d_true).abs().mean()


def std_match_loss(pred, target_pprop):
    """
    Squared difference between the standard deviation of predictions and of
    targets over the batch. Pushes prediction spread toward target spread,
    directly countering regression-to-the-mean. Unweighted.

        L = (std(pred) - std(target)) ** 2

    Parameters
    ----------
    pred          : (N,) or (N,1) float tensor of predicted pProp values
    target_pprop  : (N,) float tensor of true pProp values
    """
    return (pred.squeeze(-1).std() - target_pprop.std()) ** 2

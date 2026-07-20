#!/usr/bin/env python
"""
Multiclass AUC / average-precision metrics, one-vs-rest.

`average_precision_score` has no multiclass mode, so we binarize labels to
one-vs-rest, compute AP (and ROC-AUC) per class, then support-weight them into
the "weighted" aggregates. Computing per-class first gives the per-class metrics
the user wants for free, and makes the weighted aggregate identical to sklearn's
average="weighted" (a class with zero support is dropped from the average).
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from losses import group_sample_weights


def _weighted_pearson(x, y, w):
    """Weighted Pearson correlation between x and y with per-point weights w.

    Returns np.nan if there are fewer than 2 points or either weighted variance
    is zero (correlation undefined). Invariant to the scale of w (weights are
    renormalized to sum 1 internally).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    # Correlation is undefined for <2 points, non-positive total weight, or a
    # constant input. The constant check uses peak-to-peak (exact) rather than
    # the weighted variance, whose floating-point dust can be a tiny non-zero
    # that slips past a `denom == 0` guard and yields a spurious ~0.0.
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    wsum = w.sum()
    if wsum <= 0:
        return np.nan
    w = w / wsum
    mx = np.sum(w * x)
    my = np.sum(w * y)
    dx = x - mx
    dy = y - my
    cov = np.sum(w * dx * dy)
    var_x = np.sum(w * dx * dx)
    var_y = np.sum(w * dy * dy)
    denom = np.sqrt(var_x * var_y)
    if denom == 0:
        return np.nan
    r = cov / denom
    return float(r) if np.isfinite(r) else np.nan


def compute_weighted_regression_stats(y_true, y_pred, sample_weights):
    """Inverse-class-frequency-weighted Pearson, Spearman, MAE & R² between true
    and predicted pProp, so each statistic reflects the rare high-pProp classes
    rather than being dominated by the majority class.

    - Weighted Spearman is the weighted Pearson of the rank-transformed values
      (`scipy.stats.rankdata`, average method for ties).
    - Weighted MAE and R² reuse sklearn's `sample_weight` support.

    With uniform weights every statistic reduces to its ordinary counterpart
    (`scipy.stats.pearsonr` / `spearmanr`, `np.mean(|Δ|)`, `sklearn.r2_score`).

    Parameters
    ----------
    y_true        : (N,) float — true continuous pProp values
    y_pred        : (N,) float — regression head's predicted pProp
    sample_weights: (N,) float — per-point weights (e.g. inverse class frequency)

    Returns
    -------
    dict with weighted_pearson, weighted_spearman, weighted_mae, weighted_r2.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(sample_weights, dtype=np.float64)

    weighted_pearson = _weighted_pearson(y_true, y_pred, w)
    weighted_spearman = _weighted_pearson(rankdata(y_true), rankdata(y_pred), w)

    n = len(y_true)
    wsum = w.sum()
    if n == 0 or wsum <= 0:
        weighted_mae = np.nan
    else:
        weighted_mae = float(mean_absolute_error(y_true, y_pred, sample_weight=w))

    # R² is undefined with <2 points or a constant target (mirrors _safe_r2 in
    # the eval notebook), so guard those rather than let sklearn return a 0.0.
    if n < 2 or wsum <= 0 or np.ptp(y_true) == 0:
        weighted_r2 = np.nan
    else:
        weighted_r2 = float(r2_score(y_true, y_pred, sample_weight=w))

    return {
        "weighted_pearson": weighted_pearson,
        "weighted_spearman": weighted_spearman,
        "weighted_mae": weighted_mae,
        "weighted_r2": weighted_r2,
    }


def compute_class_mae_metrics(y_true_pprop, pred_pprop, y_class, class_names,
                              groups=None):
    """
    Whole-set MAE between predicted and true pProp, reported two ways, plus a
    per-class breakdown.

    - `mae`          : simple MAE over the entire set (every point weighted 1).
    - `weighted_mae` : class-balanced MAE over the entire set. With `groups=None`
                       each point is weighted by inverse *class* frequency
                       (weight_i = 1 / n_{class(i)}). With `groups` supplied, points
                       are weighted by inverse *group* frequency instead, so classes
                       sharing a group are balanced together (this keeps the reported
                       objective metric consistent with the grouped training weights,
                       e.g. artifacts in 7.5+ no longer dominate).
    - `mae_skill` / `weighted_mae_skill` : bounded, unitless skill scores of `mae`
                       and `weighted_mae` against the **median-predictor baseline**,
                       `skill = 1 - MAE / MAE_ref` (1 = perfect, 0 = no better than
                       predicting the median, <0 = worse). `MAE_ref` is the same-
                       weighting MAE of the constant median(y) predictor; it depends
                       only on the targets, so it is a fixed per-split scale that puts
                       MAE on the same ~[0,1] footing as AP / Pearson in the objective.
    - `mae_per_class`: per-class MAE over molecules truly in each class
                       (y_class == c); classes with no members get np.nan.

    Parameters
    ----------
    y_true_pprop : (N,) float — true continuous pProp values
    pred_pprop   : (N,) float — regression head's predicted pProp
    y_class      : (N,) int   — class label for each molecule (0..C-1)
    class_names  : list[str] length C
    groups       : optional length-C class->group map; None = per-class weighting.

    Returns
    -------
    dict with mae, weighted_mae, mae_per_class[class_name].
    """
    y_true_pprop = np.asarray(y_true_pprop, dtype=np.float64)
    pred_pprop = np.asarray(pred_pprop, dtype=np.float64)
    y_class = np.asarray(y_class, dtype=np.int64)
    n_classes = len(class_names)

    abs_err = np.abs(pred_pprop - y_true_pprop)
    n = len(abs_err)

    support = np.bincount(y_class, minlength=n_classes)
    per_mae = {}
    for c, name in enumerate(class_names):
        mask = y_class == c
        if mask.sum() == 0:
            per_mae[name] = np.nan
        else:
            per_mae[name] = float(np.mean(abs_err[mask]))

    mae = float(np.mean(abs_err)) if n else np.nan

    # Per-point class-balancing weights (each present point's class has count >= 1,
    # so no division by zero). The normalization constant cancels in the weighted-
    # mean ratio. Per-class 1/n_c by default; inverse *group* frequency if `groups`.
    if n:
        if groups is None:
            w = 1.0 / support[y_class].astype(np.float64)
        else:
            w = group_sample_weights(y_class, n_classes, groups)
        weighted_mae = float(np.sum(w * abs_err) / np.sum(w))

        # Skill vs the median-predictor baseline, under the same weighting. MAE_ref
        # depends only on the targets -> a fixed per-split scale (see docstring).
        base_abs = np.abs(y_true_pprop - np.median(y_true_pprop))
        mae_ref = float(np.mean(base_abs))
        weighted_mae_ref = float(np.sum(w * base_abs) / np.sum(w))
        mae_skill = 1.0 - mae / mae_ref if mae_ref > 0 else np.nan
        weighted_mae_skill = (1.0 - weighted_mae / weighted_mae_ref
                              if weighted_mae_ref > 0 else np.nan)
    else:
        weighted_mae = mae_skill = weighted_mae_skill = np.nan

    return {
        "mae": mae,
        "weighted_mae": weighted_mae,
        "mae_skill": mae_skill,
        "weighted_mae_skill": weighted_mae_skill,
        "mae_per_class": per_mae,
    }


def compute_class_pearson_metrics(y_true_pprop, pred_pprop, y_class, class_names,
                                  groups=None):
    """
    Whole-set Pearson correlation between predicted and true pProp, reported two
    ways (mirrors `compute_class_mae_metrics`).

    - `pearson`          : simple Pearson over the entire set (every point
                           weighted 1).
    - `pearson_weighted` : class-balanced Pearson over the entire set, using the
                           *same* weighting scheme as `weighted_mae` — inverse class
                           frequency by default, inverse *group* frequency when
                           `groups` is supplied.

    Parameters
    ----------
    y_true_pprop : (N,) float — true continuous pProp values
    pred_pprop   : (N,) float — regression head's predicted pProp
    y_class      : (N,) int   — class label for each molecule (0..C-1)
    class_names  : list[str] length C
    groups       : optional length-C class->group map; None = per-class weighting.

    Returns
    -------
    dict with pearson, pearson_weighted.
    """
    y = np.asarray(y_true_pprop, dtype=np.float64)
    p = np.asarray(pred_pprop, dtype=np.float64)
    y_class = np.asarray(y_class, dtype=np.int64)
    n_classes = len(class_names)
    n = len(y)

    if n == 0:
        return {"pearson": np.nan, "pearson_weighted": np.nan}

    # Uniform weights reduce _weighted_pearson to ordinary Pearson.
    pearson = _weighted_pearson(y, p, np.ones(n, dtype=np.float64))

    # Per-point class-balancing weights, identical to weighted_mae's scheme:
    # per-class 1/n_c by default, inverse *group* frequency when `groups` is given.
    if groups is None:
        support = np.bincount(y_class, minlength=n_classes)
        w = 1.0 / support[y_class].astype(np.float64)
    else:
        w = group_sample_weights(y_class, n_classes, groups)
    pearson_weighted = _weighted_pearson(y, p, w)

    return {
        "pearson": pearson,
        "pearson_weighted": pearson_weighted,
    }


def compute_metrics(y_true, probs, class_names, objective_classes=None):
    """
    Parameters
    ----------
    y_true : (N,) int labels in [0, C)
    probs  : (N, C) predicted class probabilities (rows sum to 1)
    class_names : list[str] length C, e.g. ["0-3.5", ...] for legible keys
    objective_classes : optional list of class_names to average for `macro_ap_obj`
        (the macro-AP the sweep objective selects on). None -> macro_ap_obj = nan.
        Used to exclude the 7.5+ artifact class from model selection while still
        scoring and reporting its per-class AP.

    Returns
    -------
    dict with:
        weighted_auc, weighted_ap          support-weighted over present classes
        macro_auc, macro_ap                unweighted mean over present classes
        macro_ap_obj                       unweighted mean over `objective_classes`
        weighted_ap_obj                    support-weighted mean over `objective_classes`
        auc[class_name], ap[class_name]     per-class (np.nan if class absent)
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    n_classes = len(class_names)

    Y = label_binarize(y_true, classes=list(range(n_classes)))  # (N, C)
    support = Y.sum(axis=0)  # per-class positive count
    n = len(y_true)

    per_auc, per_ap = {}, {}
    for c, name in enumerate(class_names):
        # AUC/AP need both positives and negatives present for this class.
        if support[c] == 0 or support[c] == n:
            per_auc[name] = np.nan
            per_ap[name] = np.nan
        else:
            per_auc[name] = float(roc_auc_score(Y[:, c], probs[:, c]))
            per_ap[name] = float(average_precision_score(Y[:, c], probs[:, c]))

    present = (support > 0) & (support < n)
    w = support[present].astype(float)
    auc_vals = np.array([per_auc[class_names[c]] for c in np.where(present)[0]])
    ap_vals = np.array([per_ap[class_names[c]] for c in np.where(present)[0]])
    weighted_auc = float(np.average(auc_vals, weights=w)) if w.sum() else np.nan
    weighted_ap = float(np.average(ap_vals, weights=w)) if w.sum() else np.nan
    # Macro = unweighted mean over present classes (rewards rare-class quality).
    macro_auc = float(np.mean(auc_vals)) if auc_vals.size else np.nan
    macro_ap = float(np.mean(ap_vals)) if ap_vals.size else np.nan

    # Objective AP over the selected objective classes (present ones only), so the
    # 7.5+ artifact class can be excluded from model selection. Both flavors:
    # macro_ap_obj = unweighted mean (rare-balanced); weighted_ap_obj = support-
    # weighted (overall/majority) — averaged into AP* in the sweep objective.
    if objective_classes:
        obj = [(per_ap[nm], support[class_names.index(nm)]) for nm in objective_classes
               if nm in per_ap and not np.isnan(per_ap[nm])]
        obj_ap = [a for a, _ in obj]
        obj_sup = [s for _, s in obj]
        macro_ap_obj = float(np.mean(obj_ap)) if obj_ap else np.nan
        weighted_ap_obj = (float(np.average(obj_ap, weights=obj_sup))
                           if obj_ap and sum(obj_sup) > 0 else np.nan)
    else:
        macro_ap_obj = weighted_ap_obj = np.nan

    return {
        "weighted_auc": weighted_auc,
        "weighted_ap": weighted_ap,
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "macro_ap_obj": macro_ap_obj,
        "weighted_ap_obj": weighted_ap_obj,
        "auc": per_auc,
        "ap": per_ap,
    }

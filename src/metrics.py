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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def _weighted_pearson(x, y, w):
    """
    Weighted Pearson correlation. With w all-ones this is the ordinary Pearson r.
    Returns np.nan if fewer than 2 points, zero total weight, or zero variance in
    either variable (correlation undefined).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if len(x) < 2 or w.sum() <= 0:
        return np.nan
    wsum = w.sum()
    mx = np.sum(w * x) / wsum
    my = np.sum(w * y) / wsum
    cov = np.sum(w * (x - mx) * (y - my)) / wsum
    vx = np.sum(w * (x - mx) ** 2) / wsum
    vy = np.sum(w * (y - my) ** 2) / wsum
    denom = np.sqrt(vx * vy)
    return float(cov / denom) if denom > 0 else np.nan


def _pearson(x, y, w):
    return _weighted_pearson(x, y, w)


def _spearman(x, y, w):
    # Spearman = Pearson on ranks; weighted Spearman = weighted Pearson on the
    # ordinary (average-tie) ranks of each variable.
    if len(x) < 2:
        return np.nan
    return _weighted_pearson(rankdata(x), rankdata(y), w)


def _two_group_weights(y_true_pprop, group_edge):
    """
    Split molecules into a low group [pProp < group_edge) and a high group
    [pProp >= group_edge) by TRUE pProp, and return inverse-group-frequency
    weights (w = N / (2 * n_group), mean 1) so the two groups contribute equally.

    Returns (weights (N,), high_group_mask (N,) bool). Weights are all-ones if
    either group is empty.
    """
    y = np.asarray(y_true_pprop, dtype=np.float64)
    n = len(y)
    ge = y >= group_edge
    n_ge = int(ge.sum())
    n_lt = n - n_ge
    w = np.ones(n, dtype=np.float64)
    if n_lt > 0 and n_ge > 0:
        w[~ge] = n / (2.0 * n_lt)
        w[ge] = n / (2.0 * n_ge)
    return w, ge


def compute_correlation_metrics(y_true_pprop, pred_pprop, group_edge=5.0):
    """
    Pearson / Spearman correlation between predicted and true pProp.

    Molecules are split into two groups by their TRUE pProp: a low group
    [pProp < group_edge) and a high group [pProp >= group_edge). Reported, for
    both train and val callers:
      - whole set, UNWEIGHTED  (`pearson`, `spearman`)
      - whole set, WEIGHTED so the two groups contribute equally: each molecule's
        weight is inversely proportional to its group's size,
        w = N / (2 * n_group), normalized to mean 1. This upweights the smaller
        high-pProp group. (`pearson_weighted`, `spearman_weighted`)
      - each group separately, unweighted: low group (`*_group_lt`) and high group
        (`*_group_ge`). These two groups span wide pProp ranges, so they avoid the
        range-restriction artifact that makes narrow per-bin correlations
        unreliable.

    Parameters
    ----------
    y_true_pprop : (N,) true continuous pProp values
    pred_pprop   : (N,) predicted pProp values
    group_edge   : pProp boundary splitting the low/high groups (default 5.0)

    Returns a dict; any value is np.nan when its subset has < 2 points or no
    variance.
    """
    y = np.asarray(y_true_pprop, dtype=np.float64)
    p = np.asarray(pred_pprop, dtype=np.float64)
    w, ge = _two_group_weights(y, group_edge)
    ones = np.ones(len(y), dtype=np.float64)
    return {
        "pearson": _pearson(y, p, ones),
        "spearman": _spearman(y, p, ones),
        "pearson_weighted": _pearson(y, p, w),
        "spearman_weighted": _spearman(y, p, w),
        "pearson_group_lt": _pearson(y[~ge], p[~ge], ones[~ge]),
        "spearman_group_lt": _spearman(y[~ge], p[~ge], ones[~ge]),
        "pearson_group_ge": _pearson(y[ge], p[ge], ones[ge]),
        "spearman_group_ge": _spearman(y[ge], p[ge], ones[ge]),
        "group_edge": group_edge,
    }


def compute_error_metrics(y_true_pprop, pred_pprop, group_edge=5.0):
    """
    Mean Absolute Error between predicted and true pProp.

    MAE (not MSE) is reported as the standalone error metric because the eval
    loss already IS the MSE; MAE adds a robust, pProp-unit "typical miss" that
    isn't dominated by the rare high-pProp tail. Mirrors the correlation layout:
      - whole set, UNWEIGHTED (`mae`)
      - whole set, WEIGHTED by inverse group size so the low [<edge) and high
        [>=edge) groups count equally (`mae_weighted`)
      - each group, unweighted (`mae_group_lt`, `mae_group_ge`)

    Returns a dict; any value is np.nan when its subset is empty.
    """
    y = np.asarray(y_true_pprop, dtype=np.float64)
    p = np.asarray(pred_pprop, dtype=np.float64)
    err = np.abs(p - y)
    w, ge = _two_group_weights(y, group_edge)

    def _mae(mask):
        e = err[mask]
        return float(e.mean()) if e.size else np.nan

    full = slice(None)
    return {
        "mae": _mae(full),
        "mae_weighted": float((w * err).sum() / w.sum()) if w.sum() > 0 else np.nan,
        "mae_group_lt": _mae(~ge),
        "mae_group_ge": _mae(ge),
        "group_edge": group_edge,
    }


def enrichment_factor(y_true_pprop, scores, thresholds, fractions):
    """
    Enrichment Factor at top fractions, for "active = true pProp >= threshold".

    Ranks molecules by `scores` descending (higher = predicted more elite), takes
    the top `fraction` of the list, and computes
        EF = (actives in top / size of top) / (total actives / N).
    EF = 1 is random selection; higher means the model concentrates actives near
    the top. The maximum possible EF at a given threshold is 1 / (active rate), so
    low thresholds (where most molecules are active) have a low ceiling.

    Parameters
    ----------
    y_true_pprop : (N,) true continuous pProp values
    scores       : (N,) ranking score, higher = ranked first. For the regression
                   model this is simply the predicted pProp.
    thresholds   : list[float] pProp cutoffs defining "active" (active = pprop >= t)
    fractions    : list[float] top fractions in (0, 1], e.g. [0.01] for top 1%

    Returns
    -------
    dict keyed by (threshold, fraction) -> EF float (np.nan if no actives exist).
    """
    y_true_pprop = np.asarray(y_true_pprop, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.argsort(-scores, kind="stable")  # descending by score
    out = {}
    for thr in thresholds:
        active = y_true_pprop >= thr
        n_active = int(active.sum())
        ranked_active = active[order]
        base_rate = n_active / n if n else 0.0
        for frac in fractions:
            k = max(1, int(round(frac * n)))
            if n_active == 0 or n == 0:
                out[(thr, frac)] = np.nan
            else:
                hits = int(ranked_active[:k].sum())
                out[(thr, frac)] = float((hits / k) / base_rate)
    return out


def compute_regression_metrics(y_true_pprop, pred_pprop, class_edges, class_names):
    """
    AUC / AP metrics for a regression model that predicts a scalar pProp value.

    Rather than class probabilities, we use a soft per-class score: negative
    distance from the predicted pProp to the nearest bin boundary. A prediction
    inside the bin scores 0 (maximum); predictions outside score proportionally
    more negative as they move further away. This plugs directly into the same
    one-vs-rest AUC/AP pipeline used by the classifier.

    Parameters
    ----------
    y_true_pprop : (N,) float — true continuous pProp values
    pred_pprop   : (N,) float — model's scalar pProp predictions
    class_edges  : list of (lo, hi) tuples defining each bin; lo/hi may be ±inf
    class_names  : list[str] length C, e.g. ["0-4.5", "4.5-5.5", ...]

    Returns
    -------
    Same dict structure as compute_metrics, plus:
        bin_accuracy  fraction of samples whose predicted bin matches the true bin
    """
    y_true_pprop = np.asarray(y_true_pprop, dtype=np.float64)
    pred_pprop = np.asarray(pred_pprop, dtype=np.float64)
    n_classes = len(class_names)

    # Binary ground-truth labels: molecule i is positive for class c iff
    # its true pProp falls in that class's bin.
    def assign_bin(pprop_vals):
        out = np.full(len(pprop_vals), -1, dtype=np.int64)
        for c, (lo, hi) in enumerate(class_edges):
            mask = (pprop_vals >= lo) & (pprop_vals < hi)
            out[mask] = c
        return out

    true_bins = assign_bin(y_true_pprop)
    pred_bins = assign_bin(pred_pprop)
    bin_accuracy = float(np.mean(true_bins == pred_bins))

    Y = np.zeros((len(y_true_pprop), n_classes), dtype=np.float64)
    for c in range(n_classes):
        Y[:, c] = (true_bins == c).astype(np.float64)

    # Soft score for class c: 0 inside the bin, negative outside (by distance).
    scores = np.zeros((len(pred_pprop), n_classes), dtype=np.float64)
    for c, (lo, hi) in enumerate(class_edges):
        below = np.maximum(0.0, lo - pred_pprop)   # 0 if pred >= lo (or lo=-inf)
        above = np.maximum(0.0, pred_pprop - hi)   # 0 if pred < hi (or hi=+inf)
        scores[:, c] = -(below + above)

    support = Y.sum(axis=0)
    n = len(y_true_pprop)

    per_auc, per_ap = {}, {}
    for c, name in enumerate(class_names):
        if support[c] == 0 or support[c] == n:
            per_auc[name] = np.nan
            per_ap[name] = np.nan
        else:
            per_auc[name] = float(roc_auc_score(Y[:, c], scores[:, c]))
            per_ap[name] = float(average_precision_score(Y[:, c], scores[:, c]))

    present = (support > 0) & (support < n)
    w = support[present].astype(float)
    auc_vals = np.array([per_auc[class_names[c]] for c in np.where(present)[0]])
    ap_vals = np.array([per_ap[class_names[c]] for c in np.where(present)[0]])
    weighted_auc = float(np.average(auc_vals, weights=w)) if w.sum() else np.nan
    weighted_ap = float(np.average(ap_vals, weights=w)) if w.sum() else np.nan
    macro_auc = float(np.mean(auc_vals)) if auc_vals.size else np.nan
    macro_ap = float(np.mean(ap_vals)) if ap_vals.size else np.nan

    return {
        "weighted_auc": weighted_auc,
        "weighted_ap": weighted_ap,
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "auc": per_auc,
        "ap": per_ap,
        "bin_accuracy": bin_accuracy,
    }


def compute_metrics(y_true, probs, class_names):
    """
    Parameters
    ----------
    y_true : (N,) int labels in [0, C)
    probs  : (N, C) predicted class probabilities (rows sum to 1)
    class_names : list[str] length C, e.g. ["0-4.5", ...] for legible keys

    Returns
    -------
    dict with:
        weighted_auc, weighted_ap          support-weighted over present classes
        macro_auc, macro_ap                unweighted mean over present classes
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

    return {
        "weighted_auc": weighted_auc,
        "weighted_ap": weighted_ap,
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "auc": per_auc,
        "ap": per_ap,
    }

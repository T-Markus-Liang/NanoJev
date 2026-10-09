"""Standalone numpy-only numerical baselines for NanoJev T11.

Algorithms only. No data loading, metadata access, heldout access, or network.

Public API
----------
fit_standardizer(X) -> (mean, scale)
transform(X, mean, scale) -> X_standardized
fit_linear(X, y, objective="ce", steps=200, lr=0.05, l2=0.001, seed=1729) -> model
predict_linear(model, X) -> probabilities
fit_gbm(X, y, rounds=30, lr=0.1) -> model
predict_gbm(model, X) -> probabilities
metrics(y, p) -> {"nll", "brier", "accuracy", "ece", "count"}
paired_group_ci(y, p_candidate, p_baseline, groups, seed=1729, replicates=1000)
"""

from __future__ import annotations

import numpy as np

_LOG_EPS = 1e-15
_GBM_NAME = "gradient_boosted_stumps"
_MINIBATCH = 16
_SIGMOID_CLIP = 500.0


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _as_2d(X, name="X"):
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_1d(v, name="y", size=None):
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if size is not None and arr.shape[0] != size:
        raise ValueError(f"{name} length mismatch")
    return arr


def _check_binary(y):
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y must be binary with values in {0, 1}")
    return y


def _check_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    ivalue = int(value)
    if ivalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return ivalue


def _check_float(name, value, minimum, strict=False):
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number")
    fvalue = float(value)
    if not np.isfinite(fvalue):
        raise ValueError(f"{name} must be finite")
    if strict:
        if fvalue <= minimum:
            raise ValueError(f"{name} must be > {minimum}")
    elif fvalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return fvalue


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -_SIGMOID_CLIP, _SIGMOID_CLIP)))


def _nll(y, p):
    pc = np.clip(p, _LOG_EPS, 1.0 - _LOG_EPS)
    return float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))


# --------------------------------------------------------------------------- #
# standardization (train-only statistics; caller supplies fitted values)
# --------------------------------------------------------------------------- #
def fit_standardizer(X):
    """Return per-column (mean, scale) with zero-variance columns mapped to 1."""
    arr = _as_2d(X, "X")
    mean = arr.mean(axis=0)
    scale = arr.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean, scale


def transform(X, mean, scale):
    """Apply externally supplied (train-only) mean/scale to X."""
    arr = np.asarray(X, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("X must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X must contain only finite values")
    if mean.ndim != 1 or scale.ndim != 1:
        raise ValueError("mean and scale must be 1-dimensional")
    if mean.shape != scale.shape or mean.shape[0] != arr.shape[1]:
        raise ValueError("mean/scale shape mismatch with X")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("mean and scale must contain only finite values")
    if np.any(scale <= 0.0):
        raise ValueError("scale must be positive")
    return (arr - mean) / scale


# --------------------------------------------------------------------------- #
# linear model: logistic CE or exact binary Brier, full reproducibility
# --------------------------------------------------------------------------- #
def fit_linear(X, y, objective="ce", steps=200, lr=0.05, l2=0.001, seed=1729):
    """Fit a linear probabilistic model by deterministic mini-batch SGD.

    objective="ce"    -> logistic cross-entropy; gradient dL/dz = p - y.
    objective="brier" -> exact binary Brier; gradient dL/dz = 2*(p-y)*p*(1-p).
    Initialization is always exactly zero, independent of seed/objective. The
    shuffle order is derived solely from ``seed`` so paired objectives share it.
    L2 regularizes weights only (bias excluded).
    """
    if objective not in ("ce", "brier"):
        raise ValueError("objective must be 'ce' or 'brier'")
    X = _as_2d(X, "X")
    y = _check_binary(_as_1d(y, "y", X.shape[0]))
    steps = _check_int("steps", steps, 1)
    lr = _check_float("lr", lr, 0.0, strict=True)
    l2 = _check_float("l2", l2, 0.0)
    seed = _check_int("seed", seed, 0)

    n, d = X.shape
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    initial_w = w.copy()
    initial_b = float(b)

    batch = max(1, min(n, _MINIBATCH))
    for step in range(steps):
        start = (step * batch) % n
        idx = order[(start + np.arange(batch)) % n]
        xb = X[idx]
        yb = y[idx]
        p = _sigmoid(xb @ w + b)
        m = xb.shape[0]
        if objective == "ce":
            grad_z = (p - yb) / m
        else:
            grad_z = (2.0 * (p - yb) / m) * (p * (1.0 - p))
        grad_w = xb.T @ grad_z + l2 * w
        grad_b = float(np.sum(grad_z))
        w = w - lr * grad_w
        b = b - lr * grad_b

    if not np.all(np.isfinite(w)) or not np.isfinite(b):
        raise ValueError("linear fit diverged")
    return {
        "weights": w,
        "bias": float(b),
        "initial_weights": initial_w,
        "initial_bias": initial_b,
        "objective": objective,
        "steps": steps,
        "lr": lr,
        "l2": l2,
        "seed": seed,
    }


def predict_linear(model, X):
    if not isinstance(model, dict) or "weights" not in model or "bias" not in model:
        raise ValueError("invalid linear model")
    X = _as_2d(X, "X")
    w = np.asarray(model["weights"], dtype=np.float64)
    b = float(model["bias"])
    if w.ndim != 1 or w.shape[0] != X.shape[1]:
        raise ValueError("model/feature mismatch")
    if not np.all(np.isfinite(w)) or not np.isfinite(b):
        raise ValueError("model parameters must be finite")
    return _sigmoid(X @ w + b)


# --------------------------------------------------------------------------- #
# gradient boosted regression stumps on Bernoulli negative gradients
# --------------------------------------------------------------------------- #
def _best_stump(X, residual):
    """Deterministic train-only best regression stump, or None if unsplittable."""
    n, d = X.shape
    best = None
    best_sse = np.inf
    for j in range(d):
        x = X[:, j]
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        rs = residual[order]
        if n < 2 or xs[0] == xs[-1]:
            continue
        csum = np.cumsum(rs)
        sumsq = np.cumsum(rs * rs)
        total = csum[-1]
        total_sq = sumsq[-1]
        k = np.arange(0, n - 1)
        k = k[xs[:-1] < xs[1:]]
        if k.size == 0:
            continue
        left_n = (k + 1).astype(np.float64)
        right_n = (n - k - 1).astype(np.float64)
        left_sum = csum[k]
        right_sum = total - left_sum
        sse = (sumsq[k] - left_sum * left_sum / left_n) + (
            (total_sq - sumsq[k]) - right_sum * right_sum / right_n
        )
        pos = int(np.argmin(sse))
        value = float(sse[pos])
        if value < best_sse:
            best_sse = value
            kk = int(k[pos])
            # Using the left endpoint exactly realizes the evaluated partition,
            # even for adjacent floats whose midpoint rounds up.
            threshold = float(xs[kk])
            left_value = float(left_sum[pos] / left_n[pos])
            right_value = float(right_sum[pos] / right_n[pos])
            best = (j, threshold, left_value, right_value)
    return best


def fit_gbm(X, y, rounds=30, lr=0.1):
    """Fit gradient boosted regression stumps on logistic pseudo-residuals.

    This is explicitly an implementation of boosted stumps
    ("gradient_boosted_stumps"), not a full GBM library.
    """
    X = _as_2d(X, "X")
    y = _check_binary(_as_1d(y, "y", X.shape[0]))
    rounds = _check_int("rounds", rounds, 1)
    lr = _check_float("lr", lr, 0.0, strict=True)

    n = X.shape[0]
    base_p = float(np.mean(y))
    base_p = min(max(base_p, _LOG_EPS), 1.0 - _LOG_EPS)
    base = float(np.log(base_p / (1.0 - base_p)))

    F = np.full(n, base, dtype=np.float64)
    stumps = []
    for _ in range(rounds):
        p = _sigmoid(F)
        residual = y - p
        stump = _best_stump(X, residual)
        if stump is None:
            break
        j, threshold, left_value, right_value = stump
        update = np.where(X[:, j] <= threshold, left_value, right_value)
        F = F + lr * update
        stumps.append((j, threshold, left_value, right_value))

    return {
        "name": _GBM_NAME,
        "n_features": X.shape[1],
        "base": base,
        "lr": lr,
        "rounds": len(stumps),
        "stumps": stumps,
    }


def predict_gbm(model, X):
    if not isinstance(model, dict) or "stumps" not in model or "base" not in model:
        raise ValueError("invalid gbm model")
    X = _as_2d(X, "X")
    if model.get("n_features") != X.shape[1]:
        raise ValueError("model/feature mismatch")
    F = np.full(X.shape[0], float(model["base"]), dtype=np.float64)
    lr = float(model["lr"])
    if not np.all(np.isfinite(F)) or not np.isfinite(lr) or lr <= 0:
        raise ValueError("invalid gbm parameters")
    for j, threshold, left_value, right_value in model["stumps"]:
        if type(j) is not int or not 0 <= j < X.shape[1]:
            raise ValueError("model/feature mismatch")
        if not all(np.isfinite(v) for v in (threshold, left_value, right_value)):
            raise ValueError("invalid stump parameters")
        F = F + lr * np.where(X[:, j] <= threshold, left_value, right_value)
    return _sigmoid(F)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics(y, p):
    y = _check_binary(_as_1d(y, "y"))
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("p must be 1-dimensional")
    if p.shape[0] == 0:
        raise ValueError("p must be non-empty")
    if p.shape[0] != y.shape[0]:
        raise ValueError("y/p length mismatch")
    if not np.all(np.isfinite(p)):
        raise ValueError("p must contain only finite values")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p must be in [0, 1]")

    n = int(y.shape[0])
    clipped = np.clip(p, _LOG_EPS, 1.0 - _LOG_EPS)
    nll = -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    brier = float(np.mean((p - y) ** 2))
    accuracy = float(np.mean((p >= 0.5) == (y == 1.0)))

    bin_index = np.minimum((p * 10.0).astype(np.int64), 9)
    ece = 0.0
    for b in range(10):
        mask = bin_index == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        confidence = float(np.mean(p[mask]))
        observed = float(np.mean(y[mask]))
        ece += (count / n) * abs(observed - confidence)

    return {
        "nll": float(nll),
        "brier": brier,
        "accuracy": accuracy,
        "ece": float(ece),
        "count": n,
    }


# --------------------------------------------------------------------------- #
# paired group bootstrap CI
# --------------------------------------------------------------------------- #
def paired_group_ci(y, p_candidate, p_baseline, groups, seed=1729, replicates=1000):
    """Cluster bootstrap CI for paired metric differences.

    Entire groups (clusters) are resampled with replacement using shared row
    indices for candidate and baseline. Row-level i.i.d. bootstrap is never used.
    """
    y = _check_binary(_as_1d(y, "y"))
    candidate = np.asarray(p_candidate, dtype=np.float64)
    baseline = np.asarray(p_baseline, dtype=np.float64)
    for name, arr in (("p_candidate", candidate), ("p_baseline", baseline)):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional")
        if arr.shape[0] != y.shape[0]:
            raise ValueError(f"{name} length mismatch")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(f"{name} must be in [0, 1]")
    seed = _check_int("seed", seed, 0)
    replicates = _check_int("replicates", replicates, 1)

    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be 1-dimensional")
    if groups.shape[0] != y.shape[0]:
        raise ValueError("groups length mismatch")
    if groups.dtype.kind in 'fc' and not np.all(np.isfinite(groups)):
        raise ValueError("group IDs must be finite")
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = int(inverse.max()) + 1 if inverse.size else 0
    if n_groups < 2:
        raise ValueError("at least 2 groups are required")

    rows_per_group = [np.nonzero(inverse == i)[0] for i in range(n_groups)]
    rng = np.random.default_rng(seed)
    nll_deltas = np.empty(replicates, dtype=np.float64)
    brier_deltas = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        sampled = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([rows_per_group[i] for i in sampled])
        yy = y[rows]
        pc = candidate[rows]
        pb = baseline[rows]
        nll_deltas[r] = _nll(yy, pc) - _nll(yy, pb)
        brier_deltas[r] = float(np.mean((pc - yy) ** 2) - np.mean((pb - yy) ** 2))

    nll_point = _nll(y, candidate) - _nll(y, baseline)
    brier_point = float(
        np.mean((candidate - y) ** 2) - np.mean((baseline - y) ** 2)
    )
    return {
        "nll": {
            "delta": float(nll_point),
            "lower": float(np.percentile(nll_deltas, 2.5)),
            "upper": float(np.percentile(nll_deltas, 97.5)),
        },
        "brier": {
            "delta": float(brier_point),
            "lower": float(np.percentile(brier_deltas, 2.5)),
            "upper": float(np.percentile(brier_deltas, 97.5)),
        },
    }

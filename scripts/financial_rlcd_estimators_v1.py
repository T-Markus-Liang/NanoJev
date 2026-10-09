"""Numpy-only RLCD-like estimator objectives for NanoJev T14 (B4).

Estimator-level counterparts of the tensor objectives in
``scripts/calibrated_objectives.py`` (see docs/RLCD_EXPERIMENT.md for the
derivation).  Binary event version: candidates {0, 1}, p = sigmoid(z).

The paired proper-reward score-function estimator uses M >= 2 independent
draws A_i ~ Bernoulli(p) with replacement and a detached conditional
baseline depending only on the other M-1 draws:

    r_i = (2/M) 1[A_i = Y] - (2/(M(M-1))) (c[A_i] - 1)
    b_i = (2/M) p[Y]      - (2/(M(M-1))) sum_{j != i} p[A_j]
    L   = -sum_i sg(r_i - b_i) log p[A_i]
    E[grad_z L] = grad_z ||p - q||^2 = 4 (p - q) p (1 - p)

The correctness-only variant drops the agreement penalty and keeps only
1[A_i = Y] with conditional baseline p[Y]; its expected reward is linear in
p, so it is a deliberately biased negative control, not an RLCD candidate.

Algorithms only.  No data loading, metadata access, or network.
"""
from __future__ import annotations

import numpy as np

from financial_baseline_estimators_v1 import (
    _MINIBATCH,
    _SIGMOID_CLIP,
    _as_1d,
    _as_2d,
    _check_binary,
    _check_float,
    _check_int,
    _sigmoid,
)

SAMPLING_SEED_MULTIPLIER = 1_000_003
SAMPLING_SEED_OFFSET = 17
ESTIMATORS = ("paired_brier_pg", "correctness_pg")


def sampling_seed(training_seed):
    """Frozen derivation of the reward-sampling stream from a training seed.

    The permutation stream stays ``default_rng(training_seed)`` exactly as in
    ``financial_baseline_estimators_v1.fit_linear`` so every objective shares
    the same batch order for a given seed; sampling uses an independent
    generator so draws never perturb the shared schedule.
    """
    seed = _check_int("training_seed", training_seed, 0)
    return seed * SAMPLING_SEED_MULTIPLIER + SAMPLING_SEED_OFFSET


def _draw_rewards(actions, y, p):
    """Per-draw local rewards and detached conditional baselines (binary).

    actions: (m, M) 0/1 draws, y: (m,), p: (m,) with p = P(action = 1).
    Returns (local_reward, baseline), each (m, M).
    """
    m, M = actions.shape
    counts1 = actions.sum(axis=1)                       # (m,)
    # c[A_i] - 1 = number of *other* draws equal to A_i.
    other_same = np.where(actions == 1, counts1[:, None] - 1.0,
                          (M - counts1)[:, None] - 1.0)
    hit = (actions == y[:, None]).astype(np.float64)
    coef = 2.0 / (M * (M - 1))
    local_reward = (2.0 / M) * hit - coef * other_same
    p_of_action = np.where(actions == 1, p[:, None], 1.0 - p[:, None])
    p_of_y = np.where(y[:, None] == 1.0, p[:, None], 1.0 - p[:, None])
    other_probability_sum = p_of_action.sum(axis=1, keepdims=True) - p_of_action
    baseline = (2.0 / M) * p_of_y - coef * other_probability_sum
    return local_reward, baseline


def _pg_batch_grad_z(X, y, w, b, rng, estimator, samples):
    """Mean surrogate gradient dL/dz for one minibatch under a PG estimator."""
    p = _sigmoid(X @ w + b)                             # (m,)
    m = X.shape[0]
    actions = (rng.random((m, samples)) < p[:, None]).astype(np.float64)
    if estimator == "paired_brier_pg":
        advantage, baseline = _draw_rewards(actions, y, p)
        advantage = advantage - baseline
    elif estimator == "correctness_pg":
        # Negative control: reward 1[A = Y] only, conditional baseline p[Y].
        hit = (actions == y[:, None]).astype(np.float64)
        p_of_y = np.where(y[:, None] == 1.0, p[:, None], 1.0 - p[:, None])
        advantage = hit - p_of_y
    else:
        raise ValueError(f"unknown PG estimator: {estimator}")
    # d/dz [-adv * log p_A] = -adv * (A - p).  Mean over rows like fit_linear.
    grad_z = -(advantage * (actions - p[:, None])).sum(axis=1) / m
    return grad_z


def fit_linear_pg(X, y, estimator="paired_brier_pg", steps=200, lr=0.05,
                  l2=0.001, seed=1729, reward_samples=32):
    """Fit the same linear model as ``fit_linear`` with a sampled PG objective.

    Identical schedule to ``fit_linear``: zero initialization, the same
    seed-derived permutation and minibatch slicing, L2 on weights only.
    ``estimator`` is "paired_brier_pg" (unbiased Brier-gradient estimator) or
    "correctness_pg" (biased negative control; expected reward linear in p).
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    X = _as_2d(X, "X")
    y = _check_binary(_as_1d(y, "y", X.shape[0]))
    steps = _check_int("steps", steps, 1)
    lr = _check_float("lr", lr, 0.0, strict=True)
    l2 = _check_float("l2", l2, 0.0)
    seed = _check_int("seed", seed, 0)
    reward_samples = _check_int("reward_samples", reward_samples, 2)

    n, d = X.shape
    rng_perm = np.random.default_rng(seed)
    rng_draw = np.random.default_rng(sampling_seed(seed))
    order = rng_perm.permutation(n)

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
        grad_z = _pg_batch_grad_z(xb, yb, w, b, rng_draw, estimator, reward_samples)
        grad_w = xb.T @ grad_z + l2 * w
        grad_b = float(np.sum(grad_z))
        w = w - lr * grad_w
        b = b - lr * grad_b

    if not np.all(np.isfinite(w)) or not np.isfinite(b):
        raise ValueError("linear PG fit diverged")
    return {
        "weights": w,
        "bias": float(b),
        "initial_weights": initial_w,
        "initial_bias": initial_b,
        "objective": estimator,
        "steps": steps,
        "lr": lr,
        "l2": l2,
        "seed": seed,
        "reward_samples": reward_samples,
    }


# --------------------------------------------------------------------------- #
# bias check: E[grad_z L_PG] = 4 (p - q) p (1 - q-free exact Brier gradient)
# --------------------------------------------------------------------------- #
def exact_brier_grad_z(p, q):
    """d/dz E[(sigmoid(z) - Y)^2 | Y ~ Bernoulli(q)] = 4 (p - q) p (1 - p).

    For the binary candidate set ||p - q||^2 = 2 (p - q)^2 and E[R] =
    ||q||^2 - ||p - q||^2, so the expected surrogate gradient equals the
    gradient of the squared distance term.
    """
    p = float(p)
    q = float(q)
    if not (0.0 < p < 1.0) or not (0.0 <= q <= 1.0):
        raise ValueError("p must be in (0, 1) and q in [0, 1]")
    return 4.0 * (p - q) * p * (1.0 - p)


def paired_gradient_monte_carlo(p, y, samples, replicates, seed=1729):
    """R independent realizations of the single-row PG surrogate gradient.

    Returns {"mean", "se", "replicates"} for dL/dz at p with outcome y.
    """
    p = _check_float("p", p, 0.0, strict=True)
    if p >= 1.0:
        raise ValueError("p must be < 1")
    y = _check_binary(_as_1d(np.asarray([y]), "y", 1))[0]
    samples = _check_int("samples", samples, 2)
    replicates = _check_int("replicates", replicates, 2)
    rng = np.random.default_rng(_check_int("seed", seed, 0))
    grads = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        actions = (rng.random(samples) < p).astype(np.float64)[None, :]
        adv, base = _draw_rewards(actions, np.asarray([y]), np.asarray([p]))
        adv = adv - base
        grads[r] = -(adv * (actions - p)).sum()
    return {"mean": float(grads.mean()), "se": float(grads.std(ddof=1) / np.sqrt(replicates)),
            "replicates": replicates}


def paired_gradient_bias_check(grid_p, grid_q, samples, replicates, seed=1729,
                               max_standard_errors=5.0):
    """Data-free check that the PG surrogate gradient is unbiased for Brier.

    For each frozen (p, q) grid point and each y in {0, 1}, compare the
    Monte Carlo mean surrogate gradient with the analytic conditional
    expectation E[grad | y] (not the marginal over Y ~ q).  The conditional
    expectation of the score-function surrogate equals
    grad_z E_A[R | y] = grad_z (2 p_y - ||p||^2).
    """
    results = []
    worst = 0.0
    for p in grid_p:
        for y in (0.0, 1.0):
            mc = paired_gradient_monte_carlo(p, y, samples, replicates, seed=seed)
            # E_A[R | y] = (2/M) M p_y - (2/(M(M-1))) M(M-1) E[A_i=A_j, i!=j]
            #            = 2 p_y - (p^2 + (1-p)^2);  d/dz below.
            p_y = p if y == 1.0 else 1.0 - p
            d_r = 2.0 * (p * (1.0 - p)) * (1.0 if y == 1.0 else -1.0)
            d_norm = 2.0 * (2.0 * p - 1.0) * p * (1.0 - p)
            exact = -(d_r - d_norm)  # L = -E[R|y] surrogate expectation
            discrepancy = abs(mc["mean"] - exact)
            se = max(mc["se"], 1e-300)
            results.append({"p": float(p), "y": y, "mc_mean": mc["mean"],
                            "exact": float(exact),
                            "standard_errors": float(discrepancy / se)})
            worst = max(worst, discrepancy / se)
    return {"grid": results, "max_standard_errors": float(worst),
            "samples": samples, "replicates": replicates,
            "passed": bool(worst <= max_standard_errors)}


# --------------------------------------------------------------------------- #
# estimator gradient-variance diagnostic (descriptive, no selection use)
# --------------------------------------------------------------------------- #
def gradient_variance_diagnostic(model, X, y, estimator, samples,
                                 replicates=64, seed=1729):
    """Dispersion of the PG surrogate gradient at fitted parameters.

    Draws ``replicates`` independent realizations of the per-step minibatch
    gradient on the full supplied batch and reports the mean componentwise
    variance of grad_z contributions plus the exact-Brier gradient norm for
    scale.  Purely descriptive; never used for selection.
    """
    X = _as_2d(X, "X")
    y = _check_binary(_as_1d(y, "y", X.shape[0]))
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    replicates = _check_int("replicates", replicates, 2)
    w = np.asarray(model["weights"], dtype=np.float64)
    b = float(model["bias"])
    rng = np.random.default_rng(_check_int("seed", seed, 0))
    m = X.shape[0]
    grads = np.empty((replicates, m), dtype=np.float64)
    for r in range(replicates):
        grads[r] = _pg_batch_grad_z(X, y, w, b, rng, estimator, samples) * m
    p = _sigmoid(X @ w + b)
    exact_grad_z = (2.0 * (p - y)) * (p * (1.0 - p))
    return {"estimator": estimator, "replicates": replicates, "rows": int(m),
            "per_row_grad_variance_mean": float(grads.var(axis=0, ddof=1).mean()),
            "exact_brier_grad_l2": float(np.linalg.norm(exact_grad_z)),
            "pg_grad_mean_l2": float(np.linalg.norm(grads.mean(axis=0)))}


# --------------------------------------------------------------------------- #
# separated reward decomposition (B4: no single aggregate may hide a mode)
# --------------------------------------------------------------------------- #
def reward_parts(y, p):
    """Exact population components of the proper reward at predictions p.

    hit_component      = 2 * mean(p_y)           (prediction quality reward)
    agreement_component = mean(p^2 + (1 - p)^2)   (self-agreement penalty)
    expected_reward    = hit - agreement = E[R] under empirical outcomes
    exact_brier        = mean((p - y)^2)          (proper loss being estimated)
    """
    y = _check_binary(_as_1d(y, "y"))
    p = _as_1d(p, "p", y.shape[0])
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p must be in [0, 1]")
    p_y = np.where(y == 1.0, p, 1.0 - p)
    hit = 2.0 * float(np.mean(p_y))
    agreement = float(np.mean(p * p + (1.0 - p) * (1.0 - p)))
    return {"hit_component": hit, "agreement_component": agreement,
            "expected_reward": hit - agreement,
            "exact_brier": float(np.mean((p - y) ** 2))}


# --------------------------------------------------------------------------- #
# selective risk: abstention is a first-class action (B6), rank-based coverages
# --------------------------------------------------------------------------- #
def selective_risk(y, p, coverages):
    """Metrics on the most confident covered subsets at frozen coverages.

    Confidence s_i = 2 |p_i - 0.5|.  For each coverage c the covered set is
    the top ceil(c * n) rows by confidence, ties broken by row order; this is
    a rank rule, not a fitted threshold, so no threshold tuning is possible.
    """
    from financial_baseline_estimators_v1 import metrics

    y = _check_binary(_as_1d(y, "y"))
    p = _as_1d(p, "p", y.shape[0])
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p must be in [0, 1]")
    if not isinstance(coverages, (list, tuple)) or not coverages:
        raise ValueError("coverages must be a nonempty list")
    n = y.shape[0]
    confidence = 2.0 * np.abs(p - 0.5)
    # Stable descending-confidence order, ties keep original row order.
    order = np.argsort(-confidence, kind="mergesort")
    out = []
    for c in coverages:
        c = _check_float("coverage", c, 0.0, strict=True)
        if c > 1.0:
            raise ValueError("coverage must be <= 1")
        k = int(np.ceil(c * n))
        covered = order[:k]
        entry = {"coverage": float(c), "covered_count": int(k)}
        entry.update(metrics(y[covered], p[covered]))
        out.append(entry)
    return out

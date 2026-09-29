"""Synthetic, numpy-only tests for financial_baseline_estimators_v1.

No real data or weights. Run with:
    PYTHONDONTWRITEBYTECODE=1 python test_financial_baseline_estimators_v1.py
"""

import unittest
try:
    import numpy as np
    import financial_baseline_estimators_v1 as fe
except ModuleNotFoundError as exc:
    if exc.name != 'numpy':
        raise
    np = None


def _easy_data(seed=7, n=400, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
    return X, y


def _base_rate_nll(y):
    p = float(np.mean(y))
    return fe.metrics(y, np.full(y.shape[0], p))["nll"]


def test_ce_reproducibility():
    X, y = _easy_data()
    a = fe.fit_linear(X, y, objective="ce", seed=11)
    b = fe.fit_linear(X, y, objective="ce", seed=11)
    assert np.array_equal(a["weights"], b["weights"])
    assert a["bias"] == b["bias"]
    assert np.array_equal(fe.predict_linear(a, X), fe.predict_linear(b, X))


def test_brier_reproducibility():
    X, y = _easy_data()
    a = fe.fit_linear(X, y, objective="brier", seed=11)
    b = fe.fit_linear(X, y, objective="brier", seed=11)
    assert np.array_equal(a["weights"], b["weights"])
    assert a["bias"] == b["bias"]


def test_matched_zero_initial_weights_across_seeds_and_objectives():
    X, y = _easy_data()
    models = [
        fe.fit_linear(X, y, objective="ce", seed=1),
        fe.fit_linear(X, y, objective="ce", seed=99),
        fe.fit_linear(X, y, objective="brier", seed=1),
        fe.fit_linear(X, y, objective="brier", seed=99),
    ]
    for m in models:
        assert np.all(m["initial_weights"] == 0.0)
        assert m["initial_bias"] == 0.0
        assert m["weights"].shape == (X.shape[1],)


def test_linear_improves_on_easy_data_ce_and_brier():
    X, y = _easy_data()
    baseline = _base_rate_nll(y)
    ce = fe.fit_linear(X, y, objective="ce", steps=400, lr=0.1, seed=3)
    brier = fe.fit_linear(X, y, objective="brier", steps=400, lr=0.1, seed=3)
    p_ce = fe.predict_linear(ce, X)
    p_brier = fe.predict_linear(brier, X)
    assert fe.metrics(y, p_ce)["nll"] < baseline
    assert fe.metrics(y, p_brier)["nll"] < baseline
    assert fe.metrics(y, p_ce)["accuracy"] > 0.8
    assert np.all((p_ce >= 0.0) & (p_ce <= 1.0))


def test_constant_feature_handled():
    X, y = _easy_data()
    X = X.copy()
    X[:, 2] = 5.0
    mean, scale = fe.fit_standardizer(X)
    assert scale[2] == 1.0
    Z = fe.transform(X, mean, scale)
    assert np.all(np.isfinite(Z))
    model = fe.fit_linear(Z, y, steps=200, seed=5)
    assert np.all(np.isfinite(model["weights"]))
    assert np.all(np.isfinite(fe.predict_linear(model, Z)))


def test_standardizer_train_only_scaling():
    rng = np.random.default_rng(21)
    train = rng.normal(loc=3.0, scale=2.0, size=(200, 3))
    test = rng.normal(loc=10.0, scale=5.0, size=(50, 3))
    mean, scale = fe.fit_standardizer(train)
    Z_train = fe.transform(train, mean, scale)
    Z_test = fe.transform(test, mean, scale)
    assert np.allclose(Z_train.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z_train.std(axis=0), 1.0, atol=1e-12)
    # test set uses train statistics only
    assert np.allclose(Z_test, (test - mean) / scale)
    assert not np.isclose(Z_test.mean(axis=0), 0.0).all()


def test_gbm_improves_and_named():
    X, y = _easy_data()
    baseline = _base_rate_nll(y)
    model = fe.fit_gbm(X, y, rounds=30, lr=0.2)
    assert model["name"] == "gradient_boosted_stumps"
    assert model["rounds"] >= 1
    p = fe.predict_gbm(model, X)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert fe.metrics(y, p)["nll"] < baseline


def test_gbm_deterministic():
    X, y = _easy_data()
    a = fe.fit_gbm(X, y, rounds=15, lr=0.1)
    b = fe.fit_gbm(X, y, rounds=15, lr=0.1)
    assert a["stumps"] == b["stumps"]
    assert np.array_equal(fe.predict_gbm(a, X), fe.predict_gbm(b, X))


def test_metrics_values():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    p = np.array([0.1, 0.9, 0.6, 0.4])
    m = fe.metrics(y, p)
    assert m["count"] == 4
    assert np.isclose(m["brier"], float(np.mean((p - y) ** 2)))
    assert m["accuracy"] == 1.0
    assert 0.0 <= m["ece"] <= 1.0
    perfect = fe.metrics(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert np.isfinite(perfect["nll"])
    assert perfect["nll"] < 1e-12


def test_ci_identity_zero():
    rng = np.random.default_rng(0)
    n = 60
    y = (rng.uniform(size=n) < 0.5).astype(np.float64)
    p = rng.uniform(0.05, 0.95, size=n)
    groups = np.repeat(np.arange(6), 10)
    out = fe.paired_group_ci(y, p, p.copy(), groups, seed=4, replicates=200)
    assert out["nll"]["delta"] == 0.0
    assert out["nll"]["lower"] == 0.0
    assert out["nll"]["upper"] == 0.0
    assert out["brier"]["delta"] == 0.0
    assert out["brier"]["lower"] == 0.0
    assert out["brier"]["upper"] == 0.0


def test_ci_deterministic_and_directional():
    rng = np.random.default_rng(1)
    n = 80
    y = (rng.uniform(size=n) < 0.5).astype(np.float64)
    good = np.clip(y * 0.8 + 0.1, 0.0, 1.0)
    bad = np.full(n, 0.5)
    groups = np.repeat(np.arange(8), 10)
    a = fe.paired_group_ci(y, good, bad, groups, seed=2, replicates=300)
    b = fe.paired_group_ci(y, good, bad, groups, seed=2, replicates=300)
    assert a == b
    assert a["nll"]["delta"] < 0.0
    assert a["nll"]["lower"] <= a["nll"]["upper"]


def test_invalid_inputs():
    X, y = _easy_data()
    cases = [
        (lambda: fe.fit_standardizer(np.empty((0, 3)))),
        (lambda: fe.fit_standardizer(np.array([[np.nan, 1.0]]))),
        (lambda: fe.fit_linear(X, y[:-1])),
        (lambda: fe.fit_linear(X, np.where(y > 0.5, 2.0, y))),
        (lambda: fe.fit_linear(X, y, objective="mse")),
        (lambda: fe.fit_linear(X, y, steps=0)),
        (lambda: fe.fit_linear(X, y, lr=0.0)),
        (lambda: fe.fit_linear(X, y, l2=-1.0)),
        (lambda: fe.fit_linear(X, y, seed=-1)),
        (lambda: fe.fit_linear(X, y, steps=True)),
        (lambda: fe.fit_linear(X, y, lr=True)),
        (lambda: fe.fit_linear(X, y, l2=False)),
        (lambda: fe.fit_linear(X, y, seed=True)),
        (lambda: fe.fit_gbm(X, y, rounds=0)),
        (lambda: fe.fit_gbm(X, y, rounds=True)),
        (lambda: fe.fit_gbm(X, y, lr=True)),
        (lambda: fe.transform(X, np.zeros(2), np.ones(3))),
        (lambda: fe.transform(X, np.zeros(3), np.array([1.0, 0.0, 1.0]))),
        (lambda: fe.metrics(np.array([]), np.array([]))),
        (lambda: fe.metrics(y, np.full(y.shape[0], np.nan))),
        (lambda: fe.metrics(y, np.full(y.shape[0], 1.5))),
        (lambda: fe.metrics(y, y[:5])),
        (lambda: fe.paired_group_ci(y, y, y, np.zeros(y.shape[0]), replicates=100)),
        (lambda: fe.paired_group_ci(y, y, y, np.zeros(y.shape[0]), seed=True)),
        (lambda: fe.paired_group_ci(y, y, y, np.zeros(y.shape[0]), replicates=0)),
        (lambda: fe.predict_linear({"weights": np.zeros(2), "bias": 0.0}, X)),
    ]
    for i, fn in enumerate(cases):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"case {i} did not raise ValueError")


def test_exact_first_step_gradients_and_paired_budget():
    X = np.array([[1.0], [2.0], [-1.0]])
    y = np.array([1.0, 1.0, 0.0])
    for objective, multiplier in [('ce', 1.0), ('brier', 0.5)]:
        m = fe.fit_linear(X, y, objective=objective, steps=1, lr=0.1, l2=0)
        grad = multiplier * (0.5-y)
        assert np.allclose(m['weights'], -0.1 * X.T @ grad / len(y))
        assert np.isclose(m['bias'], -0.1 * grad.mean())


def test_minibatch_wrap_keeps_fixed_budget():
    X = np.arange(17, dtype=float).reshape(-1, 1) / 17
    y = (X[:, 0] > 0.5).astype(float)
    order = np.random.default_rng(5).permutation(17)
    w = np.zeros(1)
    b = 0.0
    for step in range(2):
        ids = order[(step*16 + np.arange(16)) % 17]
        p = 1/(1+np.exp(-(X[ids] @ w + b)))
        w -= 0.1 * X[ids].T @ (p-y[ids]) / 16
        b -= 0.1 * np.mean(p-y[ids])
    model = fe.fit_linear(X,y,steps=2,lr=0.1,l2=0,seed=5)
    assert np.allclose(model['weights'],w)
    assert np.isclose(model['bias'],b)


def test_gbm_rejects_corrupted_model_and_wrong_feature_width():
    X,y = _easy_data()
    model = fe.fit_gbm(X,y)
    with np.testing.assert_raises(ValueError):
        fe.predict_gbm(model,np.ones((2,6)))
    model['base'] = float('nan')
    with np.testing.assert_raises(ValueError):
        fe.predict_gbm(model,X)


def load_tests(loader, tests, pattern):
    if np is None:
        @unittest.skip('numpy unavailable; run with the project .venv/bin/python')
        def requires_numpy():
            pass
        return unittest.TestSuite([unittest.FunctionTestCase(requires_numpy)])
    return unittest.TestSuite(unittest.FunctionTestCase(obj) for name, obj in sorted(globals().items())
                              if name.startswith('test_') and callable(obj))


def _run_all():
    tests = [
        obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    unittest.main()

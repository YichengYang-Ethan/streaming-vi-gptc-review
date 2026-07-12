"""Tests for the streaming Pathfinder driver (compiles PyTensor; small models)."""

import numpy as np
import pytest

from models import gaussian_regression, logistic_regression, make_loader, sample_logistic

from pymc_streaming_lab import fit_streaming_pathfinder
from pymc_streaming_lab.stochastic_lbfgs import StochasticLBFGSConfig

pytestmark = pytest.mark.usefixtures("fast_compile")

CFG = StochasticLBFGSConfig(maxcor=6)


def test_set_data_changes_compiled_objective():
    """The compiled gradient closes over the pm.Data batch: set_data changes it.

    This is the seam the whole streaming design rests on — one compiled graph,
    different minibatches fed between calls.
    """
    from pymc_extras.inference.pathfinder.bfgs_sample import get_neg_logp_dlogp_of_ravel_inputs

    rng = np.random.default_rng(0)
    X, y, _ = sample_logistic(k=2, n=200, rng=rng)
    model, packed = logistic_regression(X, y)
    vg = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=True)

    x = np.zeros(2)
    _, g_all = vg(x)
    model.set_data("batch", packed[:50])  # a sub-batch
    _, g_sub = vg(x)
    assert not np.allclose(g_all, g_sub), "gradient did not respond to set_data"


def test_smoke_streaming_logistic():
    """A short streaming fit on logistic data returns finite, correctly shaped draws."""
    rng = np.random.default_rng(1)
    X, y, _ = sample_logistic(k=2, n=400, rng=rng)
    model, packed = logistic_regression(X, y)
    loader = make_loader(packed, batch_size=64)
    res = fit_streaming_pathfinder(
        model, loader, num_iters=15, num_draws=500, eval_rows=200, random_seed=2, lbfgs_config=CFG
    )
    assert res.samples.shape == (500, 2)
    assert np.isfinite(res.samples).all()
    assert res.elbo_trace.size == len(res.elbo_trace)
    assert 0.0 <= res.violation_rate <= 1.0


def test_crn_determinism():
    """Same seed gives an identical fit (loader, optimizer, and draws are all seeded)."""
    rng = np.random.default_rng(3)
    X, y, _ = sample_logistic(k=2, n=300, rng=rng)
    model, packed = logistic_regression(X, y)

    def run():
        loader = make_loader(packed, batch_size=64)
        return fit_streaming_pathfinder(
            model,
            loader,
            num_iters=20,
            num_draws=400,
            eval_rows=150,
            random_seed=7,
            lbfgs_config=CFG,
        )

    a, b = run(), run()
    np.testing.assert_array_equal(a.elbo_trace, b.elbo_trace)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_elbo_argmax_selects_max():
    """The reported argmax is the index of the largest ELBO in the trace."""
    rng = np.random.default_rng(4)
    X, y, _ = sample_logistic(k=2, n=300, rng=rng)
    model, packed = logistic_regression(X, y)
    loader = make_loader(packed, batch_size=64)
    res = fit_streaming_pathfinder(
        model, loader, num_iters=25, num_draws=300, eval_rows=150, random_seed=5, lbfgs_config=CFG
    )
    assert res.elbo_argmax == int(np.argmax(res.elbo_trace))


def test_missing_batch_var_raises():
    """A model without the named placeholder gives an actionable error."""
    rng = np.random.default_rng(6)
    X, y, _ = sample_logistic(k=2, n=100, rng=rng)
    model, packed = logistic_regression(X, y)
    loader = make_loader(packed, batch_size=32)
    with pytest.raises(KeyError, match=r"pm\.Data"):
        fit_streaming_pathfinder(model, loader, batch_var="nope", num_iters=3)


@pytest.mark.slow
def test_gaussian_equivalence_to_analytic():
    """On a Gaussian posterior the streaming fit recovers the analytic mean and sd.

    Full-batch loader = deterministic Pathfinder; the target is exactly Gaussian,
    which Pathfinder is built to represent, so mean/sd must match the closed-form
    posterior within Monte-Carlo error across seeds.
    """
    rng = np.random.default_rng(10)
    k, n, sigma = 3, 500, 1.0
    beta_true = np.array([1.5, -2.0, 0.5])
    X = rng.normal(size=(n, k))
    y = X @ beta_true + rng.normal(0, sigma, size=n)
    model, packed, amean, acov = gaussian_regression(X, y, sigma)
    asd = np.sqrt(np.diag(acov))

    for seed in range(3):
        loader = make_loader(packed, batch_size=n)  # full batch
        res = fit_streaming_pathfinder(
            model,
            loader,
            num_iters=60,
            num_draws=4000,
            eval_rows=n,
            random_seed=seed,
            lbfgs_config=CFG,
        )
        post_mean = res.samples.mean(0)
        post_sd = res.samples.std(0)
        se = asd / np.sqrt(res.samples.shape[0])
        assert np.all(np.abs(post_mean - amean) < 6 * se + 0.01), (
            f"seed {seed}: mean {post_mean} vs analytic {amean}"
        )
        assert np.all(np.abs(post_sd - asd) < 0.3 * asd), (
            f"seed {seed}: sd {post_sd} vs analytic {asd}"
        )


@pytest.mark.slow
def test_batch_size_robustness():
    """Posterior means agree across batch sizes, down to small minibatches."""
    rng = np.random.default_rng(11)
    X, y, _ = sample_logistic(k=3, n=2000, rng=rng)
    model, packed = logistic_regression(X, y)

    means = {}
    for bs in (2000, 256):
        loader = make_loader(packed, batch_size=bs, shuffle=(bs < 2000), seed=0)
        res = fit_streaming_pathfinder(
            model,
            loader,
            num_iters=150,
            num_draws=3000,
            eval_rows=1000,
            random_seed=0,
            lbfgs_config=CFG,
        )
        means[bs] = res.samples.mean(0)
    # minibatch mean tracks the full-batch mean to within a modest tolerance
    assert np.all(np.abs(means[256] - means[2000]) < 0.25 + 0.15 * np.abs(means[2000]))


@pytest.mark.slow
def test_violation_rate_below_20pct():
    """Same-batch pairing keeps the curvature-violation rate under the proposal's 20%."""
    rng = np.random.default_rng(12)
    X, y, _ = sample_logistic(k=4, n=4000, rng=rng)
    model, packed = logistic_regression(X, y)
    loader = make_loader(packed, batch_size=256, shuffle=True, seed=1)
    res = fit_streaming_pathfinder(
        model, loader, num_iters=200, num_draws=500, eval_rows=1000, random_seed=0, lbfgs_config=CFG
    )
    assert res.violation_rate < 0.20, f"violation rate {res.violation_rate:.3f}"


def test_full_data_logp_exact_with_tail():
    """Streaming full-data logP equals model.logp over the whole dataset, including the
    trailing partial batch the training loader drops (G8 regression)."""
    from models import gaussian_regression
    from pymc_extras.inference.pathfinder.bfgs_sample import get_neg_logp_dlogp_of_ravel_inputs

    from pymc_streaming_lab.streaming_pathfinder import _compile_batched_logp, _full_data_logp

    rng = np.random.default_rng(0)
    k, n = 3, 350  # 350 is not divisible by 128 -> a genuine partial tail
    X = rng.normal(size=(n, k))
    y = X @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 1.0, size=n)
    model, packed, *_ = gaussian_regression(X, y, 1.0)
    loader = make_loader(packed, batch_size=128, shuffle=True, seed=1)

    # the complete pass visits every row exactly once, tail included
    assert sum(b.shape[0] for b in loader.complete_batches()) == n

    prior_fn = _compile_batched_logp(model, model.free_RVs, jacobian=True)
    obs_fn = _compile_batched_logp(model, model.observed_RVs, jacobian=False)
    nlp = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=True)
    phi = rng.normal(size=(5, k))
    got = _full_data_logp(phi, loader, model, "batch", prior_fn, obs_fn, n)
    for i in range(phi.shape[0]):
        model.set_data("batch", packed)  # full data in the placeholder = the exact truth
        truth = -nlp(phi[i].astype(np.float64))[0]
        assert abs(got[i] - truth) < 1e-6, (i, got[i], truth)

"""Tiny models and loaders shared by the streaming-pathfinder tests."""

import numpy as np

from pymc_streaming_lab import DataLoader


def make_loader(data, batch_size, shuffle=False, seed=0):
    """Wrap a dense (N, cols) array in a vendored DataLoader.

    ``len(loader) == N`` (the row count), which is what the model passes as
    ``total_size`` and what the streaming driver reads as N.
    """
    n, cols = data.shape

    def factory():
        yield np.asarray(data, dtype=np.float64)

    return DataLoader(
        factory,
        batch_size=batch_size,
        sample_shape=(cols,),
        total_size=n,
        shuffle=shuffle,
        seed=seed,
    )


def gaussian_regression(X, y, sigma, prior_sd=10.0):
    """Linear regression with fixed noise; all-Normal so unconstrained == constrained.

    Returns ``(model, packed_data, analytic_mean, analytic_cov)``. The packed data
    is ``hstack([X, y])`` for the loader; the analytic Gaussian posterior of ``beta``
    is the ground truth the streaming fit must recover.
    """
    import pymc as pm

    n, k = X.shape
    packed = np.hstack([X, y[:, None]]).astype(np.float64)
    with pm.Model() as model:
        batch = pm.Data("batch", packed)
        beta = pm.Normal("beta", 0.0, prior_sd, shape=k)
        mu = pm.math.dot(batch[:, :k], beta)
        pm.Normal("y", mu, sigma, observed=batch[:, k], total_size=n)

    prec = X.T @ X / sigma**2 + np.eye(k) / prior_sd**2
    cov = np.linalg.inv(prec)
    mean = cov @ (X.T @ y / sigma**2)
    return model, packed, mean, cov


def logistic_regression(X, y, prior_sd=2.0):
    """Bayesian logistic regression with a pm.Data batch placeholder.

    Returns ``(model, packed_data)`` where packed is ``hstack([X, y])``.
    """
    import pymc as pm

    n, k = X.shape
    packed = np.hstack([X, y[:, None]]).astype(np.float64)
    with pm.Model() as model:
        batch = pm.Data("batch", packed)
        beta = pm.Normal("beta", 0.0, prior_sd, shape=k)
        logit = pm.math.dot(batch[:, :k], beta)
        pm.Bernoulli("y", logit_p=logit, observed=batch[:, k], total_size=n)
    return model, packed


def sample_logistic(k, n, rng, beta_true=None):
    """Draw a logistic-regression dataset (X, y, beta_true)."""
    beta_true = rng.normal(size=k) if beta_true is None else beta_true
    X = rng.normal(size=(n, k))
    p = 1.0 / (1.0 + np.exp(-(X @ beta_true)))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    return X, y, beta_true

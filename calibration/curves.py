"""Simulated loss curves and a vectorized CUSUM twin for calibration.

The generators produce synthetic ``pm.fit``-style loss traces in two families:

- **non-converged** curves whose improvement rate stays above the monitor's
  allowance for the whole horizon (any trigger on these is a false positive);
- **converged** curves whose improvement dies at a known step ``t_star``
  (trigger step minus ``t_star`` is the detection delay).

``simulate_monitor`` re-implements ``StreamingConvergenceMonitor``'s exact
recurrence vectorized across curves, so the calibration study can process
thousands of 20k-step traces in seconds. ``tests/test_calibration.py``
verifies step-for-step equality between the twin and the real class.
"""

import numpy as np

HORIZON = 20_000
_SQRT_PI_OVER_2 = float(np.sqrt(np.pi) / 2.0)

#: Non-converged curve kinds used by the FPR study.
NONCONVERGED_KINDS = ("linear", "powerlaw", "heteroscedastic", "heavytail")


def _finish(deltas, start=1000.0):
    return start - np.cumsum(deltas, axis=-1)


def make_nonconverged(kind, n_curves, horizon=HORIZON, rng=None):
    """Loss traces that keep improving for the whole horizon.

    Parameters are randomized per curve within ranges that keep the mean
    standardized improvement comfortably above the default allowance
    (``kappa=0.25``) at every step up to ``horizon``.
    """
    rng = np.random.default_rng(rng)
    t = np.arange(1, horizon + 1, dtype=float)

    if kind == "linear":
        rate = rng.uniform(0.5, 3.0, size=(n_curves, 1))
        noise = rng.uniform(0.3, 1.5, size=(n_curves, 1))
        deltas = rng.normal(rate, noise, size=(n_curves, horizon))
    elif kind == "powerlaw":
        # improvement ~ c * t^-0.5 with c chosen so z stays in [1, 4] at the horizon
        noise = rng.uniform(0.3, 1.5, size=(n_curves, 1))
        z_end = rng.uniform(1.0, 4.0, size=(n_curves, 1))
        c = z_end * noise * np.sqrt(horizon)
        deltas = rng.normal(c * t**-0.5, noise, size=(n_curves, horizon))
    elif kind == "heteroscedastic":
        # noise scale drifts by ~4x over the horizon; improvement tracks it,
        # so the curve is always improving at ~2 robust sd per step
        base = rng.uniform(0.3, 1.5, size=(n_curves, 1))
        phase = rng.uniform(0.0, 2 * np.pi, size=(n_curves, 1))
        scale = base * (1.0 + 1.5 * (1 + np.sin(2 * np.pi * t / horizon + phase)) / 2)
        deltas = rng.normal(2.0 * scale, scale)
    elif kind == "heavytail":
        rate = rng.uniform(1.0, 3.0, size=(n_curves, 1))
        noise = rng.uniform(0.3, 1.0, size=(n_curves, 1))
        # Student-t(3) noise scaled to unit variance
        tnoise = rng.standard_t(3, size=(n_curves, horizon)) / np.sqrt(3.0)
        deltas = rate + noise * tnoise
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {NONCONVERGED_KINDS}")
    return _finish(deltas)


def make_converged(n_curves, t_star, horizon=HORIZON, rng=None):
    """Loss traces whose improvement dies exactly at ``t_star``."""
    rng = np.random.default_rng(rng)
    rate = rng.uniform(0.5, 3.0, size=(n_curves, 1))
    noise = rng.uniform(0.3, 1.5, size=(n_curves, 1))
    deltas = rng.normal(rate, noise, size=(n_curves, horizon))
    deltas[:, t_star:] = rng.normal(0.0, noise, size=(n_curves, horizon - t_star))
    return _finish(deltas)


def simulate_monitor(
    losses, kappa=0.25, h=20.0, halflife=200.0, min_steps=1000, z_clip=4.0, sigma_floor=1e-12
):
    """Vectorized twin of ``StreamingConvergenceMonitor`` across many curves.

    Mirrors the class recurrence step for step (von Neumann scale estimate,
    winsorized standardized increments, arming at ``min_steps``).
    ``tests/test_calibration.py`` asserts exact equality with the class.

    Parameters
    ----------
    losses : ndarray, shape (n_curves, horizon)
        Finite loss traces (the class's NaN handling is not replicated here).

    Returns
    -------
    stop_step : ndarray of int, shape (n_curves,)
        Step index at which each curve triggered, or -1 if it never did.
    """
    losses = np.asarray(losses, dtype=float)
    n_curves, horizon = losses.shape
    lam = np.exp(np.log(0.5) / halflife)

    scale = np.abs((losses[:, 1] - losses[:, 2]) - (losses[:, 0] - losses[:, 1]))  # primed at i=2
    prev_delta = losses[:, 1] - losses[:, 2]
    S = np.zeros(n_curves)
    stop_step = np.full(n_curves, -1, dtype=int)
    active = np.ones(n_curves, dtype=bool)

    # i = 0 primes prev_loss; i = 1 primes prev_delta; i = 2 primes scale;
    # accumulation begins once i >= min_steps.
    for i in range(3, horizon):
        delta = losses[:, i - 1] - losses[:, i]
        abs_diff = np.abs(delta - prev_delta)
        prev_delta = delta
        sigma = scale * _SQRT_PI_OVER_2 + sigma_floor
        scale = lam * scale + (1.0 - lam) * abs_diff
        z = np.clip(delta / sigma, -z_clip, z_clip)
        if i >= min_steps:
            S = np.maximum(0.0, S + (kappa - z))
            fired = active & (S > h)
            stop_step[fired] = i
            active &= ~fired
            if not active.any():
                break
            S[~active] = 0.0  # frozen; keeps the loop cheap and harmless
    return stop_step

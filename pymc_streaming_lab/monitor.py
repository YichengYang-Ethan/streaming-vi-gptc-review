"""Loss-based early stopping for ``pm.fit`` on noisy (streaming) ELBO traces.

Both classes follow PyMC's variational callback protocol: they are callables
invoked once per optimization step as ``callback(approx, losses, i)``, where
``losses`` is the array of per-step losses so far (each a one-MC-sample
negative-ELBO estimate) and stopping is signalled by raising
:class:`StopIteration`, which ``pm.fit`` catches, logs, and returns the partial
result. This module imports only numpy, so it can be lifted into
``pymc/variational/callbacks.py`` as-is.

Why not an "ELBO plateau" check
-------------------------------
Deciding that SGD on a noisy ELBO has converged is a sequential
change-detection problem: the per-step loss carries Monte-Carlo and minibatch
noise that is typically one to two orders of magnitude larger than the
per-step improvement, and the values are autocorrelated through the parameter
trajectory. Threshold checks on windowed means have no controllable false
alarm rate there. :class:`StreamingConvergenceMonitor` instead runs a
one-sided CUSUM (Page, 1954) on the standardized per-step improvement, which
accumulates evidence that the improvement rate has fallen below a small
allowance and fires only when that evidence is sustained.
"""

import numpy as np

__all__ = ["PatienceStopping", "StreamingConvergenceMonitor"]

# E|X| = sigma * sqrt(2/pi) for X ~ N(0, sigma); the mean absolute successive
# difference of an i.i.d. series estimates sqrt(2) * sigma_noise, so
# sigma_noise = mean|delta_t - delta_{t-1}| * sqrt(pi) / 2.
_SQRT_PI_OVER_2 = float(np.sqrt(np.pi) / 2.0)


class StreamingConvergenceMonitor:
    """Stop ``pm.fit`` when the loss improvement rate decays to noise level.

    Let ``delta_t = losses[t-1] - losses[t]`` (positive while optimizing). The
    monitor standardizes each increment by a robust exponentially-weighted
    scale estimate, ``z_t = delta_t / sigma_t``, and accumulates a one-sided
    CUSUM statistic::

        S_t = max(0, S_{t-1} + (kappa - z_t))

    While optimization improves at more than ``kappa`` robust standard
    deviations per step, the increments are negative and ``S`` stays pinned at
    zero. Once improvement drops below the allowance, ``S`` grows by about
    ``kappa`` per step, and convergence is declared when ``S > h`` — i.e. only
    after sustained evidence, never from a single noisy step. Accumulation is
    armed only after ``min_steps``, so the scale estimate warms up first and
    the monitor cannot fire during early transients.

    Two robustness details, both forced by the calibration study:

    - ``sigma`` is estimated from *successive differences* of the increments
      (von Neumann, 1941): ``sigma ~ EW-mean |delta_t - delta_{t-1}| * sqrt(pi)/2``.
      Differencing removes any slowly-varying improvement trend, so a
      fast-decaying loss (early ADVI) cannot inflate the scale estimate and
      fake small ``z``.
    - ``z`` is winsorized at ``+/- z_clip`` before accumulating, so a single
      heavy-tailed loss spike contributes bounded evidence in either
      direction.

    Parameters
    ----------
    kappa : float
        Allowance (reference value) in robust standard deviations per step.
        Improvement below ``kappa * sigma`` counts as evidence of convergence.
    h : float
        CUSUM decision threshold. Larger values trade detection delay for a
        lower false-alarm rate.
    halflife : float
        Half-life, in steps, of the exponentially-weighted scale estimate.
    min_steps : int
        Number of steps before the CUSUM is armed. Must be large enough for
        the scale estimate to stabilize (a few half-lives).
    z_clip : float
        Winsorization bound on the standardized increment.
    sigma_floor : float
        Lower bound on the scale estimate, guarding against exactly-constant
        losses driving ``z`` to infinity.
    keep_diagnostics : bool
        Record per-step ``delta``, ``sigma``, ``z`` and ``S`` trajectories,
        retrievable via :meth:`diagnostics`. Core state is O(1) regardless.

    Notes
    -----
    The defaults ``kappa=0.25, h=20, halflife=200`` are frozen from the
    calibration study in ``calibration/``: over 1000 still-improving traces in
    each of four families (linear, decaying power-law, heteroscedastic, and
    Student-t heavy-tailed noise), the worst-family false-positive rate is
    0.6%, with a median detection delay of ~70 steps once improvement stops
    (see ``figures/cusum_oc.png``). ``halflife`` has almost no effect because
    the von-Neumann scale estimate is trend-robust on its own. Like every
    sequential test, the monitor has an indifference region: a trace whose
    improvement rate sits *below* roughly ``kappa`` robust standard deviations
    per step is, by the allowance's definition, treated as converged — pick
    ``kappa`` to match the smallest improvement rate worth waiting for. A jump
    *upward* in loss (e.g. the stream's distribution shifts at an epoch
    boundary) makes ``z`` negative, which drains ``S`` — the monitor
    automatically withdraws its convergence evidence when the objective moves.

    Examples
    --------
    .. code-block:: python

        monitor = StreamingConvergenceMonitor()
        approx = pm.fit(100_000, callbacks=[monitor])  # stops early if converged
    """

    def __init__(
        self,
        kappa=0.25,
        h=20.0,
        halflife=200.0,
        min_steps=1000,
        z_clip=4.0,
        sigma_floor=1e-12,
        keep_diagnostics=True,
    ):
        if kappa <= 0 or h <= 0 or halflife <= 0:
            raise ValueError("kappa, h and halflife must all be positive")
        if z_clip <= kappa:
            raise ValueError(f"z_clip ({z_clip}) must exceed kappa ({kappa})")
        if min_steps < 0:
            raise ValueError(f"min_steps must be non-negative, got {min_steps!r}")
        self.kappa = float(kappa)
        self.h = float(h)
        self.halflife = float(halflife)
        self.min_steps = int(min_steps)
        self.z_clip = float(z_clip)
        self.sigma_floor = float(sigma_floor)
        self.keep_diagnostics = bool(keep_diagnostics)

        self._lam = float(np.exp(np.log(0.5) / self.halflife))
        self.n_nonfinite = 0
        self._prev_loss = None
        self._prev_delta = None  # previous improvement, for successive differencing
        self._scale = None  # EW mean of |delta_t - delta_{t-1}|
        self._S = 0.0
        self._armed = False
        self._history = {"step": [], "delta": [], "sigma": [], "z": [], "S": []}

    def __call__(self, approx, losses, i):
        if losses is None:
            raise TypeError(
                f"{type(self).__name__} needs per-step losses; run pm.fit with score=True "
                "(the default for ADVI) or remove this callback."
            )
        loss = float(losses[-1])
        if not np.isfinite(loss):
            self.n_nonfinite += 1
            return
        if self._prev_loss is None:
            self._prev_loss = loss
            return

        delta = self._prev_loss - loss
        self._prev_loss = loss

        if self._prev_delta is None:
            self._prev_delta = delta
            return
        abs_diff = abs(delta - self._prev_delta)
        self._prev_delta = delta

        # Standardize with the *previous* scale so a step never judges itself,
        # then fold the successive difference into the estimate.
        if self._scale is None:
            self._scale = abs_diff
            return
        sigma = self._scale * _SQRT_PI_OVER_2 + self.sigma_floor
        self._scale = self._lam * self._scale + (1.0 - self._lam) * abs_diff
        z = float(np.clip(delta / sigma, -self.z_clip, self.z_clip))

        if i >= self.min_steps:
            if not self._armed:
                self._S = 0.0
                self._armed = True
            self._S = max(0.0, self._S + (self.kappa - z))

        if self.keep_diagnostics:
            hist = self._history
            hist["step"].append(i)
            hist["delta"].append(delta)
            hist["sigma"].append(sigma)
            hist["z"].append(z)
            hist["S"].append(self._S)

        if self._armed and self._S > self.h:
            raise StopIteration(
                f"StreamingConvergenceMonitor: converged at step {i} "
                f"(S={self._S:.2f} > h={self.h:g})"
            )

    def diagnostics(self):
        """Return recorded per-step trajectories as numpy arrays.

        Keys: ``step``, ``delta``, ``sigma``, ``z``, ``S``. Empty arrays when
        ``keep_diagnostics=False``.
        """
        return {key: np.asarray(values) for key, values in self._history.items()}

    def reset(self):
        """Forget all state, e.g. after an optimizer restart."""
        self.n_nonfinite = 0
        self._prev_loss = None
        self._prev_delta = None
        self._scale = None
        self._S = 0.0
        self._armed = False
        self._history = {key: [] for key in self._history}


class PatienceStopping:
    """Stop when the smoothed loss makes no new best for ``patience`` steps.

    The deliberately boring baseline: an exponentially-weighted moving average
    of the loss, and a Keras-style patience rule on its running minimum. Fewer
    assumptions than the CUSUM monitor, longer detection delay.

    Parameters
    ----------
    patience : int
        Steps without a new best smoothed loss before stopping.
    halflife : float
        Half-life, in steps, of the loss smoother.
    min_steps : int
        Never stop before this step.
    """

    def __init__(self, patience=500, halflife=100.0, min_steps=0):
        if patience <= 0 or halflife <= 0:
            raise ValueError("patience and halflife must be positive")
        self.patience = int(patience)
        self.halflife = float(halflife)
        self.min_steps = int(min_steps)
        self._lam = float(np.exp(np.log(0.5) / self.halflife))
        self.n_nonfinite = 0
        self._smoothed = None
        self._best = np.inf
        self._best_step = 0

    def __call__(self, approx, losses, i):
        if losses is None:
            raise TypeError(
                f"{type(self).__name__} needs per-step losses; run pm.fit with score=True "
                "(the default for ADVI) or remove this callback."
            )
        loss = float(losses[-1])
        if not np.isfinite(loss):
            self.n_nonfinite += 1
            return
        if self._smoothed is None:
            self._smoothed = loss
        else:
            self._smoothed = self._lam * self._smoothed + (1.0 - self._lam) * loss
        if self._smoothed < self._best:
            self._best = self._smoothed
            self._best_step = i
        if i >= self.min_steps and i - self._best_step > self.patience:
            raise StopIteration(
                f"PatienceStopping: no improvement for {self.patience} steps "
                f"(best smoothed loss {self._best:.4f} at step {self._best_step})"
            )

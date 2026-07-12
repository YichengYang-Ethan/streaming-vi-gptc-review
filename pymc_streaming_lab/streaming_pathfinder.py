"""Streaming (minibatch) Pathfinder for PyMC.

Runs Pathfinder where the log-density gradients come from minibatches yielded by
a :class:`DataLoader`, rather than the full dataset. The model carries the data in
a ``pm.Data`` placeholder scaled with ``total_size=len(loader)`` (the same pattern
as the streaming ADVI Trainer), so ``model.logp()`` already returns the correctly
rescaled full-data log-density for whatever batch is currently set.

Three phases, one compiled graph:

1. **Optimize.** Drive :func:`run_stochastic_lbfgs`; before each step a fresh
   minibatch is written into the placeholder, so every gradient in a step (and
   therefore each Schraudolph curvature pair) is on one batch. Each accepted step
   stores the sampler-ready ``(x, g, alpha, s_win, z_win)``.
2. **Select.** Set one fixed evaluation batch, draw the Monte-Carlo normals *once*
   (common random numbers), and score every stored iterate's Gaussian ELBO through
   the same compiled sampler. A shared evaluation batch + shared draws make the
   comparison paired, so the argmax is not chosen by whichever iterate happened to
   see the luckiest batch.
3. **Draw.** Re-sample the best iterate's Gaussian at full width and, optionally,
   Pareto-smoothed importance resample it.

The optimizer core is in :mod:`pymc_streaming_lab.stochastic_lbfgs`; the compiled
log-density, Gaussian sampler, and PSIS are reused unchanged from
:mod:`pymc_extras.inference.pathfinder`.
"""

from dataclasses import dataclass

import numpy as np

from pymc_streaming_lab.stochastic_lbfgs import StochasticLBFGSConfig, run_stochastic_lbfgs

__all__ = ["StreamingPathfinderResult", "fit_streaming_pathfinder"]


@dataclass
class StreamingPathfinderResult:
    """Output of :func:`fit_streaming_pathfinder`.

    Attributes
    ----------
    samples : ndarray, shape (num_draws, N)
        Posterior draws in the model's raveled unconstrained space.
    logP, logQ : ndarray, shape (num_draws,)
        Target and proposal log-densities of the returned draws.
    elbo_trace : ndarray
        ELBO of every stored iterate, evaluated on the shared evaluation batch.
    elbo_argmax : int
        Index of the selected iterate within ``elbo_trace``.
    pareto_k : float or None
        PSIS Pareto shape diagnostic (None when importance sampling is disabled).
    violation_rate : float
        Curvature-rejection rate of the optimizer (proposal target < 0.20).
    n_ls_failures : int
        Number of line-search failures during optimization.
    """

    samples: np.ndarray
    logP: np.ndarray
    logQ: np.ndarray
    elbo_trace: np.ndarray
    elbo_argmax: int
    pareto_k: float | None
    violation_rate: float
    n_ls_failures: int


def _elbo(logP, logQ):
    """Mean ELBO over the draws, matching upstream ``LBFGSStreamingCallback``.

    A single draw with non-finite ``logP`` (proposal mass where the target has zero
    density) collapses the estimate to ``-inf`` rather than being dropped, so a
    support-violating iterate cannot outscore a valid one.
    """
    logP = np.asarray(logP)
    logQ = np.asarray(logQ)
    finite = np.isfinite(logP)
    if not np.any(finite):
        return -np.inf
    logP_safe = np.where(finite, logP, -np.inf)
    elbo = float(np.mean(logP_safe - logQ))
    return elbo if np.isfinite(elbo) else -np.inf


def fit_streaming_pathfinder(
    model,
    loader,
    *,
    batch_var="batch",
    num_iters=200,
    num_elbo_draws=10,
    num_draws=1000,
    eval_rows=2000,
    jitter=2.0,
    jacobian_correction=True,
    importance_sampling="psis",
    lbfgs_config=None,
    random_seed=None,
):
    """Fit a streaming Pathfinder approximation to ``model`` using ``loader``.

    Parameters
    ----------
    model : pymc.Model
        A model whose data enter through a ``pm.Data`` placeholder named
        ``batch_var`` and whose likelihood passes ``total_size=len(loader)``.
    loader : iterable
        Yields minibatches (arrays whose leading axis is the batch rows) and
        supports ``len(loader) == N`` (the dataset row count).
    batch_var : str
        Name of the ``pm.Data`` placeholder to stream into.
    num_iters : int
        Number of stochastic L-BFGS steps.
    num_elbo_draws : int
        Monte-Carlo draws per iterate for ELBO selection.
    num_draws : int
        Draws returned from the selected Gaussian.
    eval_rows : int
        Rows in the fixed evaluation batch used for iterate selection.
    jitter : float
        Uniform jitter added to the prior initial point.
    jacobian_correction : bool
        Include the change-of-variables Jacobian so logp is the unconstrained
        joint density (matches ``fit_pathfinder``).
    importance_sampling : {"psis", "psir", "identity", None}
        Post-hoc reweighting of the returned draws.
    lbfgs_config : StochasticLBFGSConfig, optional
    random_seed : int, optional

    Returns
    -------
    StreamingPathfinderResult
    """
    import pymc as pm

    from pymc.blocking import DictToArrayBijection
    from pymc.initial_point import make_initial_point_fn
    from pymc.model.core import Point
    from pymc_extras.inference.pathfinder.bfgs_sample import (
        get_neg_logp_dlogp_of_ravel_inputs,
        make_pathfinder_sample_fn,
    )
    from pymc_extras.inference.pathfinder.importance_sampling import importance_sampling as psis_fn

    model = pm.modelcontext(model)
    if batch_var not in model.named_vars:
        raise KeyError(
            f"batch_var {batch_var!r} is not a variable in the model; add a "
            f"pm.Data({batch_var!r}, ...) placeholder that the data stream feeds."
        )
    cfg = lbfgs_config or StochasticLBFGSConfig()
    J = cfg.maxcor

    init_ss, elbo_ss, final_ss = np.random.SeedSequence(random_seed).spawn(3)

    # --- compile once (both close over the batch_var pm.Data shared variable) ---
    neg_logp_dlogp = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=jacobian_correction)
    ip = Point(make_initial_point_fn(model=model)(None), model=model)
    x_base = DictToArrayBijection.map(ip).data
    N = x_base.shape[0]
    sample_logp = make_pathfinder_sample_fn(model, N=N, J=J, jacobian=jacobian_correction)

    def value_grad_fn(x):
        value, grad = neg_logp_dlogp(np.asarray(x, dtype=np.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    # --- data stream: a cycling iterator over the loader ---
    epoch = iter(loader)

    def next_batch():
        nonlocal epoch
        try:
            return next(epoch)
        except StopIteration:
            epoch = iter(loader)
            return next(epoch)

    # Fixed held-out evaluation batch: the first eval_rows rows (capped at N).
    n_total = len(loader)
    target_rows = min(eval_rows, n_total)
    chunks, rows = [], 0
    while rows < target_rows:
        b = next_batch()
        chunks.append(b)
        rows += b.shape[0]
    eval_batch = np.concatenate(chunks, axis=0)[:target_rows]

    # --- phase 1: optimize on the stream ---
    init_rng = np.random.default_rng(init_ss)
    x0 = x_base + init_rng.uniform(-jitter, jitter, size=N)
    model.set_data(batch_var, next_batch())  # prime the first training batch

    traj = run_stochastic_lbfgs(
        value_grad_fn, lambda: model.set_data(batch_var, next_batch()), x0, num_iters, cfg
    )
    if not traj.iterates:
        raise RuntimeError(
            "Streaming L-BFGS produced no accepted steps; try more iterations, a "
            "smaller jitter, or a larger batch size."
        )

    # --- phase 2: select the best iterate on a shared batch with common random numbers ---
    model.set_data(batch_var, eval_batch)
    u_elbo = np.random.default_rng(elbo_ss).standard_normal((num_elbo_draws, N))
    elbo_trace = np.empty(len(traj.iterates))
    for k, it in enumerate(traj.iterates):
        _, logQ, logP, _ = sample_logp(
            it["x"], it["g"], it["alpha"], it["s_win"], it["z_win"], u_elbo
        )
        elbo_trace[k] = _elbo(logP, logQ)
    elbo_argmax = int(np.argmax(elbo_trace))
    best = traj.iterates[elbo_argmax]

    # --- phase 3: draw from the selected Gaussian (eval batch still set) ---
    u_final = np.random.default_rng(final_ss).standard_normal((num_draws, N))
    phi, logQ, logP, _ = sample_logp(
        best["x"], best["g"], best["alpha"], best["s_win"], best["z_win"], u_final
    )
    samples = np.asarray(phi)
    logP = np.asarray(logP)
    logQ = np.asarray(logQ)

    pareto_k = None
    if importance_sampling is not None:
        result = psis_fn(
            samples[None],
            logP[None],
            logQ[None],
            num_draws,
            method=importance_sampling,
            random_seed=int(final_ss.generate_state(1)[0]),
        )
        samples = np.asarray(result.samples)
        pareto_k = result.pareto_k

    return StreamingPathfinderResult(
        samples=samples,
        logP=logP,
        logQ=logQ,
        elbo_trace=elbo_trace,
        elbo_argmax=elbo_argmax,
        pareto_k=pareto_k,
        violation_rate=traj.violation_rate,
        n_ls_failures=traj.n_ls_failures,
    )

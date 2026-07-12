# streaming-vi review snapshot

A throwaway public snapshot of two components from a GSoC 2026 PyMC project
(*Streaming Variational Inference for Large Datasets*), extracted from a private
staging repo purely to get an external code review. Not the canonical home — the
data-transport half of the project lives in public draft PRs
[pymc-extras#698](https://github.com/pymc-devs/pymc-extras/pull/698) (DataLoader)
and [pymc-extras#710](https://github.com/pymc-devs/pymc-extras/pull/710) (Trainer).

## What's here

- `pymc_streaming_lab/monitor.py` — `StreamingConvergenceMonitor`, a one-sided
  CUSUM early-stopping callback for `pm.fit` on a noisy per-step negative-ELBO
  (called `callback(approx, losses, i)`; stops via `StopIteration`). Plus a
  `PatienceStopping` baseline.
- `pymc_streaming_lab/stochastic_lbfgs.py` — a minibatch L-BFGS trajectory whose
  curvature pairs are formed on a single batch (Schraudolph 2007).
- `pymc_streaming_lab/streaming_pathfinder.py` — drives the optimizer, selects the
  best iterate by ELBO on a fixed evaluation batch with common random numbers, then
  draws + PSIS. Reuses `make_pathfinder_sample_fn` / `get_neg_logp_dlogp_of_ravel_inputs`
  from `pymc_extras.inference.pathfinder` unchanged.
- `tests/` and `calibration/curves.py` — test coverage and the CUSUM calibration
  generators, included for review context.

## Claims worth scrutinizing

- **CUSUM.** `S_t = max(0, S_{t-1} + (kappa - z_t))`, stop when `S > h`; robust
  scale from successive differences (von Neumann) + winsorized `z`. Calibrated to a
  worst-family false-positive rate of 0.6% at `kappa=0.25, h=20`. Trace the sign of
  `S` under (i) steady improvement, (ii) converged flat noise, (iii) a *sustained
  upward* move in the loss (stream distribution shift) — does each match the
  docstring's stated behavior?
- **Stochastic L-BFGS.** Same-batch pairing is claimed to keep the curvature-
  violation rate at 0% (vs 40–50% cross-batch). What happens on a line-search
  failure, and can `violation_rate` read 0% while the optimizer makes no progress?
- **Streaming Pathfinder.** Is the `(phi, logQ, logP, _)` unpack right for the ELBO
  (`mean(logP - logQ)`, maximized) and PSIS? The model scales with
  `total_size=len(loader)` but the evaluation batch has a different row count than
  the training batches — is the rescaling consistent for selection and for the
  returned draws?

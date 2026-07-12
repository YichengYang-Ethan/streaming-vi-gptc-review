"""Unit and integration tests for the convergence-monitor callbacks."""

import numpy as np
import pytest

from pymc_streaming_lab.monitor import PatienceStopping, StreamingConvergenceMonitor


def run_monitor(monitor, losses):
    """Feed a loss trace step by step; return the stop step or None."""
    losses = np.asarray(losses, dtype=float)
    for i in range(len(losses)):
        try:
            monitor(None, losses[: i + 1], i)
        except StopIteration:
            return i
    return None


def improving_then_plateau(n_improve, n_plateau, step=1.0, noise=0.5, seed=0):
    """Loss trace that decreases by ~step per iteration, then goes flat."""
    rng = np.random.default_rng(seed)
    deltas = np.concatenate(
        [
            rng.normal(step, noise, size=n_improve),
            rng.normal(0.0, noise, size=n_plateau),
        ]
    )
    return 1000.0 - np.cumsum(deltas)


def test_no_trigger_before_min_steps():
    """A flat-from-the-start trace must never fire inside the arming window."""
    rng = np.random.default_rng(1)
    losses = 100.0 + rng.normal(0.0, 0.5, size=2000)
    monitor = StreamingConvergenceMonitor(min_steps=1000)
    stop = run_monitor(monitor, losses)
    assert stop is None or stop >= 1000


def test_triggers_on_plateau_after_arming():
    """Improvement dies at a known step; the monitor fires within a bounded delay."""
    t_star = 1500
    losses = improving_then_plateau(n_improve=t_star, n_plateau=2000)
    monitor = StreamingConvergenceMonitor(min_steps=500)
    stop = run_monitor(monitor, losses)
    assert stop is not None, "monitor never fired on a clear plateau"
    assert t_star <= stop <= t_star + 500


def test_no_trigger_on_steady_improvement():
    """A trace that keeps improving must run the full horizon."""
    losses = improving_then_plateau(n_improve=10_000, n_plateau=0)
    monitor = StreamingConvergenceMonitor(min_steps=500)
    assert run_monitor(monitor, losses) is None


def test_stopiteration_message():
    """The stop reason names the class, the step, and the statistic."""
    losses = improving_then_plateau(n_improve=1000, n_plateau=2000)
    monitor = StreamingConvergenceMonitor(min_steps=200)
    with pytest.raises(StopIteration, match=r"StreamingConvergenceMonitor: converged at step \d+"):
        for i in range(len(losses)):
            monitor(None, losses[: i + 1], i)


def test_nonfinite_losses_skipped_and_counted():
    """NaN/inf losses are ignored without corrupting the statistics."""
    losses = improving_then_plateau(n_improve=800, n_plateau=1200)
    losses[100] = np.nan
    losses[200] = np.inf
    monitor = StreamingConvergenceMonitor(min_steps=300)
    stop = run_monitor(monitor, losses)
    assert monitor.n_nonfinite == 2
    assert stop is not None  # still detects the plateau
    diag = monitor.diagnostics()
    assert np.isfinite(diag["S"]).all()
    assert np.isfinite(diag["z"]).all()


def test_none_losses_raises_typeerror():
    """score=False (losses=None) produces an actionable error, not a silent no-op."""
    monitor = StreamingConvergenceMonitor()
    with pytest.raises(TypeError, match=r"score=True"):
        monitor(None, None, 0)


def test_diagnostics_alignment():
    """Diagnostic trajectories are aligned and one entry per finite step after warm-up."""
    losses = improving_then_plateau(n_improve=500, n_plateau=0)
    monitor = StreamingConvergenceMonitor(min_steps=100)
    run_monitor(monitor, losses)
    diag = monitor.diagnostics()
    lengths = {key: len(values) for key, values in diag.items()}
    assert len(set(lengths.values())) == 1, f"misaligned diagnostics: {lengths}"
    # three priming steps: prev_loss, prev_delta, then the scale estimate
    assert lengths["S"] == len(losses) - 3


def test_sigma_adapts_to_scale_change():
    """A 10x jump in noise scale mid-stream must not fake a convergence signal."""
    rng = np.random.default_rng(3)
    deltas = np.concatenate(
        [
            rng.normal(1.0, 0.2, size=3000),
            rng.normal(10.0, 2.0, size=3000),  # still improving, just rescaled
        ]
    )
    losses = 1000.0 - np.cumsum(deltas)
    monitor = StreamingConvergenceMonitor(min_steps=500)
    assert run_monitor(monitor, losses) is None


def test_sigma_floor_prevents_z_blowup():
    """Exactly-constant losses (MAD -> 0) stay finite and stop cleanly."""
    losses = np.full(3000, 42.0)
    monitor = StreamingConvergenceMonitor(min_steps=100)
    stop = run_monitor(monitor, losses)
    diag = monitor.diagnostics()
    assert np.isfinite(diag["z"]).all()
    # constant loss really is converged; S grows by kappa per step after arming
    assert stop is not None
    expected = 100 + int(monitor.h / monitor.kappa)
    assert abs(stop - expected) <= 3


def test_reset_forgets_state():
    """reset() returns the monitor to its initial condition."""
    losses = improving_then_plateau(n_improve=500, n_plateau=1000)
    monitor = StreamingConvergenceMonitor(min_steps=100)
    run_monitor(monitor, losses)
    monitor.reset()
    assert monitor.n_nonfinite == 0
    assert not monitor._armed
    assert len(monitor.diagnostics()["S"]) == 0


def test_patience_triggers_on_stall():
    """PatienceStopping fires once the smoothed loss stops improving."""
    losses = improving_then_plateau(n_improve=1000, n_plateau=3000)
    stopper = PatienceStopping(patience=400, halflife=50.0, min_steps=100)
    stop = run_monitor(stopper, losses)
    assert stop is not None
    assert 1000 <= stop <= 2500


def test_patience_silent_while_improving():
    """PatienceStopping never fires while the loss keeps decreasing."""
    losses = improving_then_plateau(n_improve=5000, n_plateau=0)
    stopper = PatienceStopping(patience=400, halflife=50.0)
    assert run_monitor(stopper, losses) is None


def test_pm_fit_integration_smoke(fast_compile):
    """End to end inside pm.fit: early stop returns the partial approximation."""
    pm = pytest.importorskip("pymc")

    rng = np.random.default_rng(0)
    data = rng.normal(1.0, 1.0, size=200)
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 1.0)
        pm.Normal("obs", mu, 1.0, observed=data)
        monitor = StreamingConvergenceMonitor(min_steps=200, halflife=50.0, h=10.0)
        approx = pm.fit(
            5000,
            callbacks=[monitor],
            progressbar=False,
            random_seed=0,
            obj_optimizer=pm.adam(learning_rate=0.1),
        )
    # a conjugate normal model plateaus quickly; the monitor must have stopped early
    assert len(approx.hist) < 5000
    assert np.isfinite(approx.hist[-50:]).all()

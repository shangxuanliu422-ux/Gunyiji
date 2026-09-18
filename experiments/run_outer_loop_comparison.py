"""Compare position MPC, feedforward LQR, incremental preview LQT, and feedforward PD.

Run with .venv/Scripts/python.exe experiments/run_outer_loop_comparison.py --no-show
All controllers share the existing 100 Hz plant/EKF/attitude PID/allocation loop.
Incremental LQT is unconstrained finite-horizon tracking, not constrained MPC;
its first acceleration is clipped only after computing the unconstrained optimum.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "outer_loop_comparison"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import matplotlib
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are

from experiments.run_mpc_tracking_cascade_pid import (
    CasadiPositionMPC, OUTER_DT, PositionMPCResult, calculate_metrics,
    run_simulation, validate_time_steps,
)
from experiments.run_mpc_tracking_disturbance import MEASUREMENT_STD

Q = np.diag([45., 45., 70., 10., 10., 14.])
R = np.diag([.18, .18, .22])
RD = np.diag([.55, .55, .70])
QF = np.diag([120., 120., 180., 24., 24., 34.])
LIMIT = np.array([2., 2., 3.])


def model(dt=OUTER_DT):
    a = np.eye(6)
    a[:3, 3:] = dt * np.eye(3)
    b = np.vstack((.5 * dt**2 * np.eye(3), dt * np.eye(3)))
    return a, b


class AnalyticController:
    """Common acceleration bounds and interface; no online nonlinear solver."""
    def finish(self, raw):
        self.last_raw = np.asarray(raw).copy()
        command = np.clip(raw, -LIMIT, LIMIT)
        return PositionMPCResult(command, np.empty((0, 6)), command[None, :],
                                 float("nan"), bool(np.isfinite(command).all()), "analytic")


class FeedforwardPD(AnalyticController):
    def __init__(self, wn=2., zeta=1.):
        self.kp = np.full(3, wn**2)
        self.kv = np.full(3, 2 * zeta * wn)

    def solve(self, state, previous_acceleration, reference):
        error = reference[0, :6] - state
        return self.finish(reference[0, 6:9] + self.kp * error[:3] + self.kv * error[3:])


class FeedforwardLQR(AnalyticController):
    def __init__(self, r_scale=1.):
        a, b = model()
        self.p = solve_discrete_are(a, b, Q, r_scale * R)
        self.k = np.linalg.solve(r_scale * R + b.T @ self.p @ b, b.T @ self.p @ a)
        self.spectral_radius = float(max(abs(np.linalg.eigvals(a - b @ self.k))))

    def solve(self, state, previous_acceleration, reference):
        return self.finish(reference[0, 6:9] - self.k @ (state - reference[0, :6]))


class IncrementalPreviewLQT(AnalyticController):
    """Exact unconstrained tracking of the MPC stage cost using augmented state.

    z=[position, velocity, previous_acceleration], decision d=a-a_previous.
    Stage cost z'Qa z + 2 z'Na d + d'Ra d + linear reference terms.
    terminal='mpc' reproduces existing MPC's terminal cost exactly (when no
    acceleration bounds bind); 'dare' penalizes augmented terminal error with
    Riccati P about [p_ref, v_ref, a_ref]. No exact infinite-reference claim.
    """
    def __init__(self, horizon=60, delta_scale=1., terminal="dare", terminal_scale=1.):
        self.horizon, self.terminal = horizon, terminal
        a, b = model()
        self.a = np.block([[a, b], [np.zeros((3, 6)), np.eye(3)]])
        self.b = np.vstack((b, np.eye(3)))
        qa = np.zeros((9, 9)); qa[:6, :6] = Q; qa[6:, 6:] = R
        na = np.vstack((np.zeros((6, 3)), R))
        ra = R + delta_scale * RD
        self.dare_p = solve_discrete_are(self.a, self.b, qa, ra, s=na)
        self.dare_k = np.linalg.solve(ra + self.b.T @ self.dare_p @ self.b,
                                     self.b.T @ self.dare_p @ self.a + na.T)
        self.spectral_radius = float(max(abs(np.linalg.eigvals(self.a - self.b @ self.dare_k))))
        pf = self.dare_p.copy() if terminal == "dare" else np.pad(QF, ((0, 3), (0, 3)))
        self.pf = terminal_scale * pf
        self.gains, self.inverses, self.transitions = [], [], []
        p = self.pf.copy()
        for _ in range(horizon):
            g = ra + self.b.T @ p @ self.b
            invg = np.linalg.solve(g, np.eye(3))
            k = invg @ (self.b.T @ p @ self.a + na.T)
            p = qa + self.a.T @ p @ self.a - (self.a.T @ p @ self.b + na) @ k
            p = .5 * (p + p.T)
            self.gains.append(k); self.inverses.append(invg)
            self.transitions.append((self.a - self.b @ k).T)
        self.gains.reverse(); self.inverses.reverse(); self.transitions.reverse()

    def solve(self, state, previous_acceleration, reference):
        s = -self.pf @ reference[-1]
        # Reference-dependent affine term. All Riccati matrices are offline.
        for i in range(self.horizon - 1, -1, -1):
            q = -np.concatenate((Q @ reference[i, :6], R @ reference[i, 6:9]))
            r = -R @ reference[i, 6:9]
            if i == 0:
                feedforward = self.inverses[0] @ (self.b.T @ s + r)
            s = q + self.transitions[i] @ s - self.gains[i].T @ r
        z = np.concatenate((state, previous_acceleration))
        delta = -self.gains[0] @ z - feedforward
        return self.finish(previous_acceleration + delta)


class TimedController:
    """Time only solve(), separately from reference/force conversion and plant."""
    def __init__(self, controller):
        self.controller = controller
        self.times, self.commands, self.raw_commands = [], [], []

    def solve(self, *args):
        start = perf_counter()
        result = self.controller.solve(*args)
        self.times.append(perf_counter() - start)
        self.commands.append(result.acceleration.copy())
        self.raw_commands.append(getattr(self.controller, "last_raw", result.acceleration).copy())
        return result


def run_case(name, controller, args, seed, duration=None, offset=None):
    local = argparse.Namespace(duration=duration or args.duration, horizon=args.horizon,
                               inner_dt=.01, seed=seed)
    rng = np.random.default_rng(seed)
    n = round(local.duration / local.inner_dt)
    std = MEASUREMENT_STD * args.noise_scale
    noise = rng.normal(size=(n + 1, 12)) * std
    disturbance = rng.normal(size=(n, 5))
    timed = TimedController(controller)
    start = perf_counter()
    result = run_simulation("cascade_pid", local, noise, disturbance,
                            measurement_std=std, position_controller_override=timed,
                            initial_position_offset=offset)
    elapsed = perf_counter() - start
    metrics = calculate_metrics(result)
    errors = np.linalg.norm(result.states[:, :3] - result.references[:, :3], axis=1)
    commands = np.asarray(timed.commands)
    times = np.asarray(timed.times) * 1000
    metrics.update(name=name, seed=seed, duration=local.duration,
                   rmse_position=float(np.sqrt(np.mean(errors**2))),
                   mean_controller_ms=float(times.mean()), p95_controller_ms=float(np.percentile(times, 95)),
                   max_controller_ms=float(times.max()),
                   warm_mean_controller_ms=float(times[1:].mean()) if len(times)>1 else float(times.mean()),
                   deadline_misses=int(np.sum(times > 100)),
                   acceleration_boundary_fraction=float(np.mean(np.any(abs(commands) >= LIMIT - 1e-5, axis=1))),
                   acceleration_delta_rms=float(np.sqrt(np.mean(np.sum(np.diff(commands, axis=0)**2, axis=1)))),
                   simulation_wall_s=elapsed)
    print(f"{name:20} seed={seed} mean={metrics['mean_position_error']:.5f}m "
          f"max={metrics['max_position_error']:.5f}m control={times.mean():.4f}ms "
          f"wall={elapsed:.1f}s", flush=True)
    return metrics, result, commands, times


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=24.)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[21, 22, 23])
    parser.add_argument("--noise-scale", type=float, default=1.)
    parser.add_argument("--pd-wn", type=float, default=2.)
    parser.add_argument("--pd-zeta", type=float, default=1.)
    parser.add_argument("--lqr-r-scale", type=float, default=1.)
    parser.add_argument("--delta-scale", type=float, default=1.)
    parser.add_argument("--terminal", choices=["dare", "mpc"], default="dare")
    parser.add_argument("--initial-offset", type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument("--sweep", action="store_true", help="Additional 12 s tuning cases, seed 7; does not auto-select test gains")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0 or not np.isclose(args.duration / .1, round(args.duration / .1)):
        parser.error("duration must be a positive multiple of 0.1")
    if args.horizon < 1 or min(args.noise_scale, args.pd_wn, args.pd_zeta, args.lqr_r_scale) <= 0 or args.delta_scale < 0:
        parser.error("invalid horizon, scale or gain")
    validate_time_steps(.1, .01)
    factories = {
        "mpc": lambda: CasadiPositionMPC(.1, args.horizon),
        "lqr": lambda: FeedforwardLQR(args.lqr_r_scale),
        "lqr_delta_preview": lambda: IncrementalPreviewLQT(args.horizon, args.delta_scale, args.terminal),
        "ff_pd": lambda: FeedforwardPD(args.pd_wn, args.pd_zeta),
    }
    rows, payload, first = [], {}, {}
    for seed in args.seeds:
        for name, factory in factories.items():
            start = perf_counter(); controller = factory(); setup = perf_counter() - start
            row, result, command, times = run_case(name, controller, args, seed, offset=np.asarray(args.initial_offset))
            row["setup_ms"] = setup * 1000
            rows.append(row)
            prefix = f"{name}_seed{seed}"
            for field in ("time", "states", "references", "estimated_states", "desired_controls", "disturbances", "solver_success", "allocation_success"):
                payload[f"{prefix}_{field}"] = getattr(result, field)
            payload[f"{prefix}_accelerations"] = command
            payload[f"{prefix}_controller_ms"] = times
            if seed == args.seeds[0]: first[name] = (result, command, times)
            write_csv(OUT / "metrics.csv", rows)
    np.savez_compressed(OUT / "histories.npz", **payload)
    metadata = vars(args).copy()
    metadata.update(python=sys.version, numpy=np.__version__, plant_mass_scale=1., plant_inertia_scale=[1.,1.,1.],
                    outer_dt=.1, inner_dt=.01, allocation_command_rate_limits=False,
                    shared_height_integral=True, note="Seedwise identical injected noise; LQR/PD have post-clipping, MPC has predictive bounds. Times exclude initialization; full loop is not a real-time benchmark.")
    lqr, pd = FeedforwardLQR(args.lqr_r_scale), FeedforwardPD(args.pd_wn, args.pd_zeta)
    delta = IncrementalPreviewLQT(args.horizon, args.delta_scale, args.terminal)
    metadata.update(lqr_gain=lqr.k.tolist(), lqr_p=lqr.p.tolist(), lqr_spectral_radius=lqr.spectral_radius,
                    pd_kp=pd.kp.tolist(), pd_kv=pd.kv.tolist(), delta_dare_p=delta.dare_p.tolist(),
                    delta_dare_gain=delta.dare_k.tolist(), delta_spectral_radius=delta.spectral_radius)
    (OUT / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, (result, command, times) in first.items():
        err = np.linalg.norm(result.states[:, :3] - result.references[:, :3], axis=1)
        axes[0,0].plot(result.time, err, label=name)
        axes[0,1].plot(result.time, result.states[:,2]-result.references[:,2], label=name)
        axes[1,0].plot(np.arange(len(command))*.1, command[:,2], label=name)
        axes[1,1].plot(np.arange(len(times))*.1, times, label=name)
    for ax, title in zip(axes.flat, ["Position error (m)", "NED z error (m)", "Command az (m/s2)", "Outer controller call (ms)"]):
        ax.set_title(title); ax.set_xlabel("Time (s)"); ax.grid(True); ax.legend(fontsize=8)
    axes[1,1].set_yscale("log")
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=160)
    if args.sweep:
        sweep = []
        candidates = [(f"pd_wn_{w:g}", lambda w=w: FeedforwardPD(w,1.)) for w in (1.,2.,3.)]
        candidates += [(f"pd_zeta_{z:g}", lambda z=z: FeedforwardPD(2.,z)) for z in (.7,1.3)]
        candidates += [(f"lqr_r_{r:g}", lambda r=r: FeedforwardLQR(r)) for r in (1.,10.,100.)]
        candidates += [(f"delta_{d:g}", lambda d=d: IncrementalPreviewLQT(args.horizon,d,"dare")) for d in (.1,1.,10.)]
        candidates += [(f"terminal_mpc_{s:g}", lambda s=s: IncrementalPreviewLQT(args.horizon,1.,"mpc",s)) for s in (.1,1.,10.)]
        for name, factory in candidates:
            row, _, _, _ = run_case(name, factory(), args, 7, duration=12., offset=np.array([.5,-.3,.2]))
            sweep.append(row); write_csv(OUT / "tuning_sweep.csv", sweep)
    print(f"Saved results: {OUT}", flush=True)
    if args.no_show: plt.close(fig)
    else: plt.show()


if __name__ == "__main__":
    main()

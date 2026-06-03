"""Phase 0 --- the gate.

Attribute the ~0.525 s/solve baseline across its components so Phases 1-2 are prioritized
by evidence, not guesswork. Everything is JIT-compiled and ``block_until_ready``-guarded
(eager evaluation of the order-5 Lie-derivative Jacobians is ~6500x slower and would be
meaningless to time). Reports:

  * component micro-benchmarks: forward rollout, objective, gradient (reverse pass),
    constraint value, constraint Jacobian, and a Hessian-vector product;
  * full ``minimize`` wall time + iteration count over realistic snapshots;
  * the per-iteration overhead factor (solve_time / nit vs. the summed component cost),
    which quantifies the scipy<->JAX boundary + multiple-evals-per-iteration tax;
  * a recompilation count (JAX compile-log counter) to confirm the steady loop reuses one
    compiled artifact.

Real-time target: 10 Hz guidance on CPU => <= 100 ms/solve (~5x over baseline).

Run:  source env.sh && uv run python benchmarks/profile_solve.py
"""

import argparse
import csv
import logging
from pathlib import Path
import statistics
import time

import jax
import numpy as np

import rt_oac
from rt_oac.scenario import DATA_DIR, build_scenario

TARGET_MS = 100.0  # 10 Hz guidance budget on CPU
RESULTS = Path(__file__).resolve().parents[1] / "results"


class _CompileCounter(logging.Handler):
    """Counts JAX XLA compilation events (requires jax_log_compiles=True)."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        msg = record.getMessage()
        if "Compiling" in msg or "Finished XLA compilation" in msg:
            self.count += 1


def _bench(fn, *args, reps=50, warmup=3):
    """Median/p95 wall time (ms) of a jitted fn, block_until_ready-guarded."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - t) * 1e3)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (len(samples) - 1))],
        "min_ms": samples[0],
    }


def _load_states():
    """Realistic operating states from the companion closed-loop run, if available."""
    companion = rt_oac.COMPANION_SRC.parent / "data" / "optimization_results_stlog.npz"
    for path in (DATA_DIR / "optimization_results_stlog.npz", companion):
        if path.exists():
            with np.load(path) as d:
                if "states" in d.files:
                    return np.asarray(d["states"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--order", type=int, default=None, help="STLOG order (default cfg=5)"
    )
    ap.add_argument("--manifold", action="store_true", help="tangent-space 9x9 Gramian")
    ap.add_argument("--snapshots", type=int, default=12)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    sc = build_scenario(order=args.order, use_manifold=args.manifold)
    cost, ctrl = sc.cost, sc.controller
    dt = sc.stlog_dt
    states = _load_states()
    snaps = sc.snapshots(args.snapshots, states=states)
    x0, u0 = snaps[0]
    np.asarray(sc.follower_indices)

    print(
        f"# RT-OAC Phase 0 profile  (order={args.order or sc.cfg['stlog']['order']}, "
        f"manifold={args.manifold}, window={sc.window})"
    )
    print(f"# target: <= {TARGET_MS:.0f} ms/solve (10 Hz CPU)\n")

    # ---- component micro-benchmarks (jitted) --------------------------------------
    rollout = jax.jit(cost.eval_integrator)
    objective = jax.jit(lambda x, u, t: cost(x, u, t).objective)
    value_and_grad = jax.jit(
        jax.value_and_grad(lambda u, x, t: cost(x, u, t).objective)
    )
    grad_only = jax.jit(jax.grad(lambda u, x, t: cost(x, u, t).objective))
    constraint = jax.jit(lambda x, u: cost_constraint(cost, x, u))
    constraint_jac = jax.jit(jax.jacobian(lambda u, x: cost_constraint(cost, x, u)))

    def hvp(u, x, t, v):
        return jax.jvp(lambda uu: grad_only(uu, x, t), (u,), (v,))[1]

    hvp_j = jax.jit(hvp)
    v = np.ones_like(u0)

    comps = {
        "forward_rollout": _bench(rollout, x0, u0, reps=args.reps),
        "objective": _bench(objective, x0, u0, dt, reps=args.reps),
        "value_and_grad": _bench(value_and_grad, u0, x0, dt, reps=args.reps),
        "gradient_only": _bench(grad_only, u0, x0, dt, reps=args.reps),
        "constraint": _bench(constraint, x0, u0, reps=args.reps),
        "constraint_jac": _bench(constraint_jac, u0, x0, reps=args.reps),
        "hessian_vec_prod": _bench(hvp_j, u0, x0, dt, v, reps=args.reps),
    }
    print(f"{'component':<20}{'median(ms)':>12}{'p95(ms)':>10}{'min(ms)':>10}")
    for name, m in comps.items():
        print(
            f"{name:<20}{m['median_ms']:>12.3f}{m['p95_ms']:>10.3f}{m['min_ms']:>10.3f}"
        )

    # one trust-constr iteration calls (at least) objective + gradient + constraint +
    # constraint_jac; estimate the irreducible per-iteration JAX compute.
    per_iter_jax = (
        comps["objective"]["median_ms"]
        + comps["gradient_only"]["median_ms"]
        + comps["constraint"]["median_ms"]
        + comps["constraint_jac"]["median_ms"]
    )
    print(f"\nsum(obj+grad+con+conjac) per iter (JAX only): {per_iter_jax:.3f} ms")

    # ---- full solve timing + recompile probe --------------------------------------
    jax.config.update("jax_log_compiles", True)
    counter = _CompileCounter()
    jlog = logging.getLogger("jax")
    jlog.addHandler(counter)
    prev_level = jlog.level
    jlog.setLevel(logging.DEBUG)

    # compile pass (first solve)
    t = time.perf_counter()
    soln = ctrl.minimize(x0, u0, dt, sc.follower_indices)
    compile_solve_s = time.perf_counter() - t
    compiles_first = counter.count

    solve_ms, nits = [], []
    counter.count = 0
    for xi, ui in snaps:
        t = time.perf_counter()
        soln = ctrl.minimize(xi, ui, dt, sc.follower_indices)
        solve_ms.append((time.perf_counter() - t) * 1e3)
        nits.append(int(soln.nit))
    compiles_steady = counter.count
    jlog.removeHandler(counter)
    jlog.setLevel(prev_level)
    jax.config.update("jax_log_compiles", False)

    solve_ms.sort()
    med = statistics.median(solve_ms)
    p95 = solve_ms[int(0.95 * (len(solve_ms) - 1))]
    med_nit = statistics.median(nits)
    print(
        f"\nfull solve (steady, n={len(snaps)}): median {med:.1f} ms, "
        f"p95 {p95:.1f} ms, median nit {med_nit:.0f}"
    )
    print(
        f"first solve incl. compile: {compile_solve_s:.2f} s "
        f"({compiles_first} compiles); steady-loop new compiles: {compiles_steady}"
    )
    overhead = med / (per_iter_jax * max(med_nit, 1))
    print(
        f"per-iter overhead factor (solve/nit vs JAX-compute): {overhead:.1f}x "
        f"=> scipy bookkeeping + boundary + evals/iter"
    )
    print(f"\nspeedup needed for {TARGET_MS:.0f} ms target: {med / TARGET_MS:.1f}x")

    # ---- persist ------------------------------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "phase0_attribution.csv"
    with out.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["component", "median_ms", "p95_ms", "min_ms"])
        for name, m in comps.items():
            w.writerow([name, m["median_ms"], m["p95_ms"], m["min_ms"]])
        w.writerow(["full_solve_median", med, p95, ""])
        w.writerow(["median_nit", med_nit, "", ""])
        w.writerow(["compiles_first_solve", compiles_first, "", ""])
        w.writerow(["compiles_steady_loop", compiles_steady, "", ""])
    print(f"\nwrote {out}")


def cost_constraint(cost, x, u):
    """The companion distance-squared path constraint, in cost terms."""
    xs, _ = cost.eval_integrator(x, u)
    return (xs[:, 0:3] ** 2).sum(axis=1)


if __name__ == "__main__":
    main()

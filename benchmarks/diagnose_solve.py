"""One-off diagnostic: why is a steady-state solve slow, and does it recompile?

Resolves the 10 s/solve vs companion 0.4 s discrepancy by reporting, in a single warm
process: scipy's function/gradient/constraint eval counts per solve, the recompilation
count during steady solves, and the Gramian eigenvalue scale at the operating point.
"""

import logging
import statistics
import time

import jax
import jax.numpy as jnp

import rt_oac  # noqa: F401
from rt_oac.scenario import build_scenario


class _CompileCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        m = record.getMessage()
        if "Compiling" in m or "Finished XLA compilation" in m:
            self.count += 1


def main():
    sc = build_scenario()
    dt = sc.stlog_dt
    x0 = sc.x0
    u0 = sc.reference_guess(0)

    # Gramian scale sanity at the operating point.
    g = jax.jit(lambda x, u, t: sc.cost.eval_gramian(x, u, t))(x0, u0[0], dt)
    eig = jnp.linalg.eigvalsh(0.5 * (g + g.T))
    print(
        f"sliced Gramian shape {g.shape}, eigs min {float(eig.min()):.3e} "
        f"max {float(eig.max()):.3e}"
    )

    # Warmup solve (compiles; populates persistent cache).
    t = time.perf_counter()
    soln = sc.controller.minimize(x0, u0, dt, sc.follower_indices)
    print(f"warmup solve: {time.perf_counter() - t:.1f} s  nit={soln.nit}")

    # Steady solves with recompile counter.
    counter = _CompileCounter()
    jlog = logging.getLogger("jax")
    jax.config.update("jax_log_compiles", True)
    jlog.addHandler(counter)
    jlog.setLevel(logging.DEBUG)

    times = []
    for _ in range(4):
        counter.count = 0
        t = time.perf_counter()
        soln = sc.controller.minimize(x0, u0, dt, sc.follower_indices)
        dt_s = time.perf_counter() - t
        times.append(dt_s)
        print(
            f"solve {dt_s:6.2f} s  nit={soln.nit:>3}  "
            f"nfev={soln.get('nfev', '?')}  njev={soln.get('njev', '?')}  "
            f"constr_nfev={soln.get('constr_nfev', '?')}  "
            f"new_compiles={counter.count}"
        )
    jlog.removeHandler(counter)
    jax.config.update("jax_log_compiles", False)

    # Standalone gradient timing for comparison (what one njev *should* cost).
    grad = jax.jit(jax.grad(lambda u, x, t: sc.cost(x, u, t).objective))
    jax.block_until_ready(grad(u0, x0, dt))
    g_times = []
    for _ in range(20):
        t = time.perf_counter()
        jax.block_until_ready(grad(u0, x0, dt))
        g_times.append(time.perf_counter() - t)
    g_ms = statistics.median(g_times) * 1e3
    print(f"\nstandalone gradient: {g_ms:.2f} ms")
    print(
        f"median solve: {statistics.median(times):.2f} s; "
        f"implied evals @ grad-cost: {statistics.median(times) / (g_ms / 1e3):.0f}"
    )


if __name__ == "__main__":
    main()

"""Is log-det's *evaluation* more expensive than soft-min's, or is the speedup elsewhere?

Times the bare objective value+gradient (one jitted ``value_and_grad`` of the cost wrt the
input sequence) for both surrogates on an identical (x, u). Both reduce to the same per-node
eigendecomposition, so if the per-call cost is ~equal then the solve-time gap seen in
objective_solver_profile.py is NOT the metric -- it is the solver doing more work when log-det
pins the follower to the active max-distance constraint (vs soft-min orbiting the band interior).

  uv run python benchmarks/objective_eval_microbench.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.scenario import build_scenario

N = 500


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


X0 = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])


def main():
    print(f"{'objective':10s} | {'value+grad ms/call':>20s}")
    print("-" * 36)
    for name, m in (
        ("logdet", metrics.neg_logdet),
        ("softmin", metrics.neg_softmin_eig),
    ):
        sc = build_scenario(gramian_metric=m)
        x = jnp.asarray(X0)
        u0 = jnp.asarray(sc.reference_guess(0))
        vg = jax.jit(jax.value_and_grad(lambda u: sc.cost(x, u, sc.stlog_dt).objective))
        v, g = vg(u0)
        jax.block_until_ready((v, g))  # warm up / compile
        t = time.perf_counter()
        for _ in range(N):
            v, g = vg(u0)
        jax.block_until_ready((v, g))
        ms = (time.perf_counter() - t) / N * 1e3
        print(f"{name:10s} | {ms:20.4f}")


if __name__ == "__main__":
    main()

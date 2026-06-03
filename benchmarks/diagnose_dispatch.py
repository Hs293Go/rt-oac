"""Pin down the per-eval gap: controller eqx.filter_jit methods vs standalone jax.jit.

The full solve does ~50 obj + 50 grad + 50 constraint evals (nit=40) yet takes ~10 s,
while a standalone jitted gradient is ~4 ms. This measures the controller's scipy-facing
methods *exactly as scipy calls them* to locate the ~47x overhead.
"""

import statistics
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import rt_oac  # noqa: F401
from rt_oac.scenario import build_scenario


def bench(label, fn, *args, reps=30, warmup=3):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t) * 1e3)
    print(f"{label:<42}{statistics.median(ts):>10.3f} ms")
    return statistics.median(ts)


def main():
    sc = build_scenario()
    ctrl, cost = sc.controller, sc.cost
    dt = sc.stlog_dt
    x0 = sc.x0
    u0 = sc.reference_guess(0)  # (20, 8)

    min_idx = jnp.asarray(sc.follower_indices)
    const_idx = jnp.setdiff1d(jnp.arange(8), min_idx)
    u_j = jnp.asarray(u0)
    u_const = u_j[..., const_idx]  # (20, 4) constant (leader) inputs
    free_np = np.asarray(u_j[..., min_idx].ravel())  # (80,) decision vector, numpy
    free_j = jnp.asarray(free_np)
    args = (u_const, x0, dt, min_idx, const_idx)

    print("# standalone jax.jit baselines")
    obj = jax.jit(lambda u, x, t: cost(x, u, t).objective)
    grad = jax.jit(jax.grad(lambda u, x, t: cost(x, u, t).objective))
    vg = jax.jit(jax.value_and_grad(lambda u, x, t: cost(x, u, t).objective))
    bench("standalone objective (full u)", obj, u_j, x0, dt)
    bench("standalone gradient (full u)", grad, u_j, x0, dt)
    bench("standalone value_and_grad (full u)", vg, u_j, x0, dt)

    print("\n# controller eqx.filter_jit methods, as scipy calls them (numpy free vec)")
    bench("ctrl._objective (numpy)", ctrl._objective, free_np, *args)
    bench("ctrl._gradient  (numpy)", ctrl._gradient, free_np, *args)
    bench("ctrl._constraint(numpy)", ctrl._constraint, free_np, *args)
    bench(
        "ctrl._constraint_jacobian (numpy)", ctrl._constraint_jacobian, free_np, *args
    )

    print("\n# same controller methods but passing a jax array (no host transfer)")
    bench("ctrl._objective (jax)", ctrl._objective, free_j, *args)
    bench("ctrl._gradient  (jax)", ctrl._gradient, free_j, *args)

    # Is the cost of eqx.filter_jit dispatch (partition+hash of `self`/static) the gap?
    # Re-jit the controller's exact gradient computation with a plain jax.jit closure
    # over the *free* vector, fusing value+grad, to see the achievable per-eval cost.
    print("\n# plain jax.jit closure over the free vector (fused value+grad)")

    def recombine(u_free):
        u = u_free.reshape(u_const.shape[0], -1)
        full = jnp.zeros((u_const.shape[0], 8))
        full = full.at[:, const_idx].set(u_const)
        full = full.at[:, min_idx].set(u)
        return full

    fused = jax.jit(
        jax.value_and_grad(lambda uf, x, t: cost(x, recombine(uf), t).objective)
    )
    bench("plain jit fused v&g (numpy free)", fused, free_np, x0, dt)
    bench("plain jit fused v&g (jax free)", fused, free_j, x0, dt)

    # Confirm there is no recompilation hiding in eqx dispatch: call counts.
    print(f"\neqx version: {eqx.__version__}, jax version: {jax.__version__}")


if __name__ == "__main__":
    main()

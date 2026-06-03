"""Attribute one solve: time spent *inside* our callables vs. inside scipy trust-constr.

Replicates the controller's minimize() problem setup but wraps the objective / gradient /
constraint / constraint-Jacobian with cumulative timers, so we can subtract callable time
from total minimize wall time and see exactly how much is scipy's internal overhead.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

import rt_oac  # noqa: F401
from rt_oac.scenario import build_scenario


class Timer:
    def __init__(self, fn, name):
        self.fn, self.name, self.t, self.n = fn, name, 0.0, 0

    def __call__(self, *a, **k):
        t0 = time.perf_counter()
        out = np.asarray(jax.block_until_ready(self.fn(*a, **k)))
        self.t += time.perf_counter() - t0
        self.n += 1
        return out


def main():
    sc = build_scenario()
    ctrl = sc.controller
    dt, x0 = sc.stlog_dt, sc.x0
    u0 = sc.reference_guess(0)

    min_idx = jnp.asarray(sc.follower_indices)
    const_idx = jnp.setdiff1d(jnp.arange(8), min_idx)
    u_j = jnp.asarray(u0)
    u_const = u_j[..., const_idx]
    free = np.asarray(u_j[..., min_idx].ravel())
    args = (u_const, x0, dt, min_idx, const_idx)

    # bounds (free components only), matching minimize()
    lb = np.tile(sc.cfg["optim"]["lb"], (sc.window, 2))
    ub = np.tile(sc.cfg["optim"]["ub"], (sc.window, 2))
    bounds = optimize.Bounds(
        np.asarray(lb)[..., np.asarray(sc.follower_indices)].ravel(),
        np.asarray(ub)[..., np.asarray(sc.follower_indices)].ravel(),
    )

    f = Timer(lambda u: ctrl._objective(u, *args), "objective")
    g = Timer(lambda u: ctrl._gradient(u, *args), "gradient")
    c = Timer(lambda u: ctrl._constraint(u, *args), "constraint")
    cj = Timer(lambda u: ctrl._constraint_jacobian(u, *args), "constraint_jac")

    lo = sc.cfg["opc"]["min_inter_vehicle_distance"] ** 2
    hi = sc.cfg["opc"]["max_inter_vehicle_distance"] ** 2
    constr = optimize.NonlinearConstraint(
        c, np.full(sc.window, lo), np.full(sc.window, hi), jac=cj
    )

    # warmup (compile, populate cache) -- not timed for attribution
    optimize.minimize(
        f,
        free,
        jac=g,
        method="trust-constr",
        bounds=bounds,
        constraints=[constr],
        options={"maxiter": 3, "verbose": 0},
    )
    for t in (f, g, c, cj):
        t.t = t.n = 0

    t0 = time.perf_counter()
    soln = optimize.minimize(
        f,
        free,
        jac=g,
        method="trust-constr",
        bounds=bounds,
        constraints=[constr],
        options={"maxiter": 40, "verbose": 0},
    )
    total = time.perf_counter() - t0

    in_callables = f.t + g.t + c.t + cj.t
    print(f"total minimize wall:        {total:8.3f} s   (nit={soln.nit})")
    print(f"  objective:       {f.t:7.3f} s  ({f.n} calls)")
    print(f"  gradient:        {g.t:7.3f} s  ({g.n} calls)")
    print(f"  constraint:      {c.t:7.3f} s  ({c.n} calls)")
    print(f"  constraint_jac:  {cj.t:7.3f} s  ({cj.n} calls)")
    print(
        f"  sum in callables:{in_callables:7.3f} s  ({100 * in_callables / total:.1f}%)"
    )
    print(
        f"  scipy internal:  {total - in_callables:7.3f} s  "
        f"({100 * (total - in_callables) / total:.1f}%)"
    )


if __name__ == "__main__":
    main()

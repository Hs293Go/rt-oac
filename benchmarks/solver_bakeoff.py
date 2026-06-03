"""Solver bake-off on the identical OAC problem.

Phase 0 showed ~94% of the 10 s solve is scipy trust-constr's internal overhead, not the
observability computation (0.68 s for 40 iters). So the lever is the solver. Compare:
  * trust-constr (baseline)
  * trust-constr + analytic Hessian-vector product (jax HVP) for the objective
  * SLSQP (much leaner per-iteration Python; two-sided distance -> two one-sided ineqs)

Reports wall time, iterations, final objective, max constraint violation, and time spent
inside our JAX callables vs. the solver's internals.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

import rt_oac  # noqa: F401
from rt_oac.scenario import build_scenario


class Timer:
    def __init__(self, fn):
        self.fn, self.t, self.n = fn, 0.0, 0

    def __call__(self, *a):
        t0 = time.perf_counter()
        out = np.asarray(jax.block_until_ready(self.fn(*a)))
        self.t += time.perf_counter() - t0
        self.n += 1
        return out


def main():
    sc = build_scenario()
    ctrl, cost = sc.controller, sc.cost
    dt, x0 = sc.stlog_dt, sc.x0
    u0 = sc.reference_guess(0)
    fi = np.asarray(sc.follower_indices)

    min_idx = jnp.asarray(sc.follower_indices)
    const_idx = jnp.setdiff1d(jnp.arange(8), min_idx)
    u_j = jnp.asarray(u0)
    u_const = u_j[..., const_idx]
    free0 = np.asarray(u_j[..., min_idx].ravel())
    args = (u_const, x0, dt, min_idx, const_idx)

    lb = np.tile(sc.cfg["optim"]["lb"], (sc.window, 2))[..., fi].ravel()
    ub = np.tile(sc.cfg["optim"]["ub"], (sc.window, 2))[..., fi].ravel()
    bounds = optimize.Bounds(lb, ub)
    lo = sc.cfg["opc"]["min_inter_vehicle_distance"] ** 2
    hi = sc.cfg["opc"]["max_inter_vehicle_distance"] ** 2

    # Analytic objective Hessian-vector product over the free vector (for trust-constr).
    def recombine(uf):
        full = jnp.zeros((sc.window, 8)).at[:, const_idx].set(u_const)
        return full.at[:, min_idx].set(uf.reshape(sc.window, -1))

    obj_free = jax.jit(lambda uf: cost(x0, recombine(uf), dt).objective)
    grad_free = jax.jit(jax.grad(lambda uf: cost(x0, recombine(uf), dt).objective))
    hvp_free = jax.jit(lambda uf, p: jax.jvp(grad_free, (uf,), (p,))[1])

    def true_objective(uf):
        return float(obj_free(np.asarray(uf)))

    def max_viol(uf):
        d2 = np.asarray(jax.block_until_ready(ctrl._constraint(np.asarray(uf), *args)))
        return float(np.maximum(0, np.maximum(lo - d2, d2 - hi)).max())

    def run(label, fn):
        # warmup (compile / cache load) at low iters, then timed run
        try:
            fn(maxiter=2)
        except Exception as e:
            print(f"{label}: warmup failed: {e}")
            return
        t0 = time.perf_counter()
        soln, f, g, c, cj = fn(maxiter=40)
        wall = time.perf_counter() - t0
        in_call = f.t + g.t + c.t + cj.t
        print(f"\n## {label}")
        print(
            f"  wall {wall:7.3f} s | nit {int(getattr(soln, 'nit', -1)):>3} | "
            f"true_obj {true_objective(soln.x):.5g} | maxviol {max_viol(soln.x):.2e}"
        )
        print(
            f"  in-callables {in_call:.3f} s ({100 * in_call / wall:.1f}%) | "
            f"solver-internal {wall - in_call:.3f} s ({100 * (wall - in_call) / wall:.1f}%)"
        )
        print(f"  calls: obj {f.n} grad {g.n} con {c.n} conjac {cj.n}")

    # --- trust-constr (baseline) ----------------------------------------------------
    def trust_constr(maxiter):
        f = Timer(lambda u: ctrl._objective(u, *args))
        g = Timer(lambda u: ctrl._gradient(u, *args))
        c = Timer(lambda u: ctrl._constraint(u, *args))
        cj = Timer(lambda u: ctrl._constraint_jacobian(u, *args))
        con = optimize.NonlinearConstraint(
            c, np.full(sc.window, lo), np.full(sc.window, hi), jac=cj
        )
        soln = optimize.minimize(
            f,
            free0,
            jac=g,
            method="trust-constr",
            bounds=bounds,
            constraints=[con],
            options={"maxiter": maxiter, "verbose": 0},
        )
        return soln, f, g, c, cj

    # --- trust-constr + analytic objective HVP --------------------------------------
    def trust_constr_hvp(maxiter):
        f = Timer(lambda u: ctrl._objective(u, *args))
        g = Timer(lambda u: ctrl._gradient(u, *args))
        c = Timer(lambda u: ctrl._constraint(u, *args))
        cj = Timer(lambda u: ctrl._constraint_jacobian(u, *args))
        con = optimize.NonlinearConstraint(
            c, np.full(sc.window, lo), np.full(sc.window, hi), jac=cj
        )
        soln = optimize.minimize(
            f,
            free0,
            jac=g,
            hessp=lambda u, p: np.asarray(hvp_free(u, p)),
            method="trust-constr",
            bounds=bounds,
            constraints=[con],
            options={"maxiter": maxiter, "verbose": 0},
        )
        return soln, f, g, c, cj

    # --- SLSQP (two one-sided inequalities) -----------------------------------------
    def slsqp(maxiter):
        f = Timer(lambda u: ctrl._objective(u, *args))
        g = Timer(lambda u: ctrl._gradient(u, *args))
        c = Timer(lambda u: ctrl._constraint(u, *args))
        cj = Timer(lambda u: ctrl._constraint_jacobian(u, *args))
        cons = [
            {"type": "ineq", "fun": lambda u: c(u) - lo, "jac": lambda u: cj(u)},
            {"type": "ineq", "fun": lambda u: hi - c(u), "jac": lambda u: -cj(u)},
        ]
        soln = optimize.minimize(
            f,
            free0,
            jac=g,
            method="SLSQP",
            bounds=list(zip(lb, ub, strict=True)),
            constraints=cons,
            options={"maxiter": maxiter},
        )
        return soln, f, g, c, cj

    run("trust-constr (baseline)", trust_constr)
    run("trust-constr + HVP", trust_constr_hvp)
    run("SLSQP", slsqp)


if __name__ == "__main__":
    main()

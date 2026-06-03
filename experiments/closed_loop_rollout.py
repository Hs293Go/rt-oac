"""True-state receding-horizon rollout: does warm-start make the OAC solve real-time?

No EKF yet -- this isolates the optimization/real-time question. Starting from a non-MUC
relative state, at each step we solve the OAC problem, apply the first control, and advance
the true relative dynamics. Compares cold-start (reseed from reference every step) vs
warm-start (shift the previous solution). Reports per-step iteration count, wall time, and
the trajectory's accumulated observability (#observable directions of sum_k W_k).

Real-time target: <= 100 ms/solve (10 Hz).
"""

import argparse

from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.scenario import build_rt_controller, build_scenario


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def run(ctrl, sc, x_init, steps, warm, sim, score):
    x = np.asarray(x_init)
    prev_u = None
    nits, walls, nobs = [], [], []
    for step in range(steps):
        ref = sc.reference_guess(step)
        if warm and prev_u is not None:
            from rt_oac.warmstart import warm_guess

            guess = warm_guess(ref, prev_u)
        else:
            guess = ref
        res = ctrl.solve(x, guess)
        prev_u = res.u
        nits.append(res.nit)
        walls.append(res.wall_time)
        nobs.append(score(x, res.u))
        # apply first control, advance true relative dynamics, renormalize quaternion
        x_next, _ = sim(jnp.asarray(x), jnp.asarray(res.u[0]))
        x = np.array(x_next)  # writable copy (jax arrays are read-only)
        x[3:7] /= np.linalg.norm(x[3:7])
    return np.array(nits), np.array(walls), np.array(nobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--maxiter", type=int, default=80)
    args = ap.parse_args()

    sc = build_scenario(gramian_metric=metrics.neg_logdet)
    ctrl = build_rt_controller(sc, method="SLSQP", maxiter=args.maxiter)
    dt_int = sc.cfg["sim"]["integrator_dt"]
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.EULER, stepsize=dt_int)
    )

    # accumulated-observability score (#directions of sum_k W_k above rel threshold)
    gram = jax.jit(
        lambda x, u: sc.cost(
            x, jnp.asarray(u), sc.stlog_dt, return_gramians=True
        ).gramians
    )

    def score(x, u):
        g = np.asarray(gram(jnp.asarray(x), u))
        acc = 0.5 * (g + np.swapaxes(g, -1, -2)).sum(0)
        e = np.linalg.eigvalsh(acc)
        return int((e > 1e-6 * e[-1]).sum())

    x_init = np.concatenate([[2.0, 0, 0], quat_z(20), [0, 1.0, 0]])  # non-MUC

    for warm in (False, True):
        label = "warm-start" if warm else "cold-start"
        nits, walls, nobs = run(ctrl, sc, x_init, args.steps, warm, sim, score)
        # skip step 0 (may include residual compile) for steady stats
        steady = slice(1, None)
        print(f"\n## {label}")
        print(
            f"  nit:   median {np.median(nits[steady]):.0f}  mean {nits[steady].mean():.1f}  "
            f"range [{nits[steady].min()}, {nits[steady].max()}]"
        )
        print(
            f"  wall:  median {np.median(walls[steady]) * 1e3:.0f} ms  "
            f"p95 {np.percentile(walls[steady], 95) * 1e3:.0f} ms"
        )
        print(
            f"  #observable dirs over run: median {np.median(nobs):.0f} (of 6), "
            f"min {nobs.min()}, final {nobs[-1]}"
        )
        print(f"  per-step nit: {nits.tolist()}")


if __name__ == "__main__":
    main()

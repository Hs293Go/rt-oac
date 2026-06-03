"""Method x Problem matrix: do our innovations help across difficulty levels?

Rows (method):  OG   = trust-constr + min-eig (run to convergence)
                Ours = SLSQP + log-det + early-stop (maxiter 6)
Cols (problem): planar no-heading (harder, drone-like: drop the compass)
                planar OG-simple  (heading observed, easy)
                drone             (order-5 range-only relative pose, non-MUC)

Per cell: tick wall (ms) | iterations | achieved observability = min-eig of the accumulated
Gramian sum_k W_k (objective-agnostic, so min-eig and log-det runs share one ruler).
"""

import time

from example_lib.models import leader_follower_robots as plmdl
import jax.numpy as jnp
import numpy as np
from observability_aware_control import observability_cost

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.controller import RTController
from rt_oac.scenario import build_rt_controller, build_scenario


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def planar_obs_heading(x, u=None):
    return plmdl.observation(x, 0)  # [leader_pos(2), headings(2), range(1)]


def planar_obs_noheading(x, u=None):
    f = plmdl.observation(x, 0)
    return jnp.concatenate([f[:2], f[4:]])  # drop headings -> [leader_pos(2), range(1)]


PLANAR_X0 = np.array([0.5, 0.0, 0.0, 0.0, 5.0, 0.0])
PLANAR_DT = 0.2
DRONE_X0 = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])  # non-MUC


def planar_builder(obs_fn, var):
    def build(metric, method, maxiter):
        cost = observability_cost.ObservabilityCost(
            plmdl.dynamics,
            obs_fn,
            0.1,
            gramian_kw={"order": 1, "var": var},
            gramian_metric=metric,
            observed_indices=(),
        )
        ctrl = RTController(
            cost,
            stlog_dt=PLANAR_DT,
            lb=np.array([0.0, -2.0]),
            ub=np.array([4.0, 2.0]),
            n_inputs=4,
            follower_indices=(2, 3),
            method=method,
            maxiter=maxiter,
        )
        return ctrl, PLANAR_X0, np.tile([1.0, 0.0, 1.0, 0.0], (5, 1)), PLANAR_DT, cost

    return build


def drone_build(metric, method, maxiter):
    sc = build_scenario(gramian_metric=metric)
    ctrl = build_rt_controller(sc, method=method, maxiter=maxiter)
    return ctrl, DRONE_X0, sc.reference_guess(0), sc.stlog_dt, sc.cost


def observability(cost, x0, dt, u_full):
    g = np.asarray(
        cost(jnp.asarray(x0), jnp.asarray(u_full), dt, return_gramians=True).gramians
    )
    acc = 0.5 * (g + np.swapaxes(g, -1, -2)).sum(0)
    eig = np.linalg.eigvalsh(acc)
    return float(eig[0]), int((eig > 1e-6 * eig[-1]).sum()), len(eig)


def main():
    columns = [
        (
            "planar ours (no-heading)",
            planar_builder(planar_obs_noheading, np.r_[np.full(2, 1e-1), 1e-2]),
        ),
        (
            "planar OG (simple)",
            planar_builder(
                planar_obs_heading, np.r_[np.full(2, 1e-1), np.full(2, 1e-2), 1e-2]
            ),
        ),
        ("drone (range-only, o5)", drone_build),
    ]
    methods = [
        ("OG  trust-constr+min-eig", metrics.reciprocal_min_eig, "trust-constr", 100),
        ("Ours SLSQP+logdet stop@6", metrics.neg_logdet, "SLSQP", 6),
    ]

    print("# Method x Problem.  cell = tick_ms | nit | acc-min-eig (#obs/dim)\n")
    w = 30
    print(f"{'':<28}" + "".join(f"{c[0]:<{w}}" for c in columns))

    # reference (un-optimized) observability per column
    refrow = f"{'reference (un-optimized)':<28}"
    for _, build in columns:
        _, x0, ug, dt, cost = build(metrics.reciprocal_min_eig, "SLSQP", 2)
        me, no, nd = observability(cost, x0, dt, ug)
        refrow += f"{f'{me:.2e} ({no}/{nd})':<{w}}"
    print(refrow + "\n")

    for mlabel, metric, method, maxiter in methods:
        row = f"{mlabel:<28}"
        for _, build in columns:
            ctrl, x0, ug, dt, cost = build(metric, method, maxiter)
            ctrl.solve(x0, ug)  # warmup
            t = time.perf_counter()
            res = ctrl.solve(x0, ug)
            ms = (time.perf_counter() - t) * 1e3
            me, no, nd = observability(cost, x0, dt, res.u)
            row += f"{f'{ms:5.0f}ms {res.nit:>3}it {me:.1e}({no}/{nd})':<{w}}"
        print(row)


if __name__ == "__main__":
    main()

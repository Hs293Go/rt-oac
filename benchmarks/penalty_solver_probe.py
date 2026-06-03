"""Penalty-baked distance band + unconstrained solver vs hard SLSQP constraint.

Investigation (PROGRESS.md future-work): the Phase-0 finding is that constraint *activity*
is what makes the SLSQP solve costlier (active-set QP each iteration). If we bake the
inter-vehicle distance band into the cost as a smooth one-sided penalty, we can drop the
hard nonlinear constraint and solve with a box-bounded *unconstrained* method (L-BFGS-B) --
no active set, fully JIT-able, and a stepping stone to a JAX-native solver.

Uses log-det (which pins the follower to the max-distance bound, so the constraint is ACTIVE
and the penalty is genuinely exercised) on the quadrotor scenario, band [1, 3] m. Compares,
over a perfect-feedback rollout: median solve time, the realized distance band-violation, and
the observability objective. The penalty trades exact feasibility for speed -- the table shows
the trade.

  uv run python benchmarks/penalty_solver_probe.py
"""

import time

from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator
from scipy import optimize

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.controller import RTController
from rt_oac.scenario import build_scenario
from rt_oac.warmstart import warm_guess

STEPS = 50
LO, HI = (
    1.0,
    3.0,
)  # distance band (config default; log-det pegs the max -> active constraint)
W_PEN_GRID = [1.0, 10.0, 100.0]
LEADER, FOLLOWER = (0, 1, 2, 3), (4, 5, 6, 7)


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


X0 = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])


def renorm(x):
    x = np.array(x)
    x[3:7] /= np.linalg.norm(x[3:7])
    return x


def band_violation(dist):
    """Max realized distance-band violation over a closed-loop run [m, in distance]."""
    d = np.asarray(dist)
    return float(np.maximum(0.0, np.maximum(d - HI, LO - d)).max())


def run_hard(ctrl, sc, sim):
    x = X0.copy()
    walls, dist, obj = [], [], []
    prev_u = None
    for i in range(STEPS):
        guess = warm_guess(sc.reference_guess(i), prev_u)
        res = ctrl.solve(x, guess)
        prev_u = res.u
        walls.append(res.wall_time * 1e3)
        dist.append(float(np.linalg.norm(x[0:3])))
        obj.append(res.objective)
        x = renorm(sim(jnp.asarray(x), jnp.asarray(res.u[0]))[0])
    return walls, dist, obj


def run_penalty(vag, obs_jit, sc, sim, w, lb, ub):
    window = sc.window
    leader_idx, follower_idx = jnp.asarray(LEADER), jnp.asarray(FOLLOWER)
    lb_t = np.broadcast_to(lb, (window, len(FOLLOWER))).ravel()
    ub_t = np.broadcast_to(ub, (window, len(FOLLOWER))).ravel()
    box = list(zip(lb_t, ub_t, strict=True))
    x = X0.copy()
    walls, dist, obj = [], [], []
    prev_u = None
    for i in range(STEPS):
        guess = warm_guess(sc.reference_guess(i), prev_u)
        u_const = jnp.asarray(np.asarray(guess)[:, LEADER])
        free0 = np.asarray(np.asarray(guess)[:, FOLLOWER].ravel())
        x_j, w_j = jnp.asarray(x), jnp.asarray(float(w))
        cache: dict = {}

        def fj(free, _xj=x_j, _uc=u_const, _wj=w_j):
            key = free.tobytes()
            if cache.get("k") != key:
                v, g = vag(jnp.asarray(free), _uc, _xj, _wj)
                cache["k"], cache["v"] = key, (float(v), np.asarray(g))
            return cache["v"]

        t = time.perf_counter()
        soln = optimize.minimize(
            lambda f: fj(f)[0],
            free0,
            jac=lambda f: fj(f)[1],
            method="L-BFGS-B",
            bounds=box,
            options={"maxiter": 6},
        )
        walls.append((time.perf_counter() - t) * 1e3)
        full = jnp.zeros((window, 8)).at[:, leader_idx].set(u_const)
        full = full.at[:, follower_idx].set(jnp.asarray(soln.x).reshape(window, -1))
        prev_u = np.asarray(full)
        dist.append(float(np.linalg.norm(x[0:3])))
        obj.append(
            float(obs_jit(jnp.asarray(soln.x), u_const, x_j))
        )  # true (unpenalised) obs
        x = renorm(sim(jnp.asarray(x), jnp.asarray(full[0]))[0])
    return walls, dist, obj


def main():
    sc = build_scenario(gramian_metric=metrics.neg_logdet)
    dt = sc.cfg["sim"]["integrator_dt"]
    window = sc.window
    lb = np.array(sc.cfg["optim"]["lb"])
    ub = np.array(sc.cfg["optim"]["ub"])
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.EULER, stepsize=dt)
    )
    leader_idx, follower_idx = jnp.asarray(LEADER), jnp.asarray(FOLLOWER)

    def recombine(free, u_const):
        full = jnp.zeros((window, 8)).at[:, leader_idx].set(u_const)
        return full.at[:, follower_idx].set(free.reshape(window, -1))

    def pen_obj(free, u_const, x0, w):
        v = sc.cost(x0, recombine(free, u_const), sc.stlog_dt, return_trajectory=True)
        d2 = jnp.sum(v.states[:, 0:3] ** 2, axis=1)
        pen = jnp.sum(jnp.maximum(d2 - HI**2, 0.0) ** 2) + jnp.sum(
            jnp.maximum(LO**2 - d2, 0.0) ** 2
        )
        return v.objective + w * pen

    vag = jax.jit(jax.value_and_grad(pen_obj))
    obs_jit = jax.jit(
        lambda free, u_const, x0: sc.cost(
            x0, recombine(free, u_const), sc.stlog_dt
        ).objective
    )

    ctrl = RTController(
        sc.cost,
        stlog_dt=sc.stlog_dt,
        lb=lb,
        ub=ub,
        n_inputs=8,
        follower_indices=FOLLOWER,
        method="SLSQP",
        maxiter=6,
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=(LO**2, HI**2),
    )

    print(
        f"# Penalty-solver probe (quad, log-det, band [{LO:.0f},{HI:.0f}], steps={STEPS})"
    )
    print(
        f"{'method':<22}{'median ms':>11}{'p95 ms':>9}{'band viol [m]':>15}"
        f"{'dist[min,max]':>16}{'median obj':>12}"
    )
    hw, hd, ho = run_hard(ctrl, sc, sim)
    s = slice(1, None)
    print(
        f"{'hard SLSQP+constraint':<22}{np.median(np.array(hw)[s]):>11.1f}"
        f"{np.percentile(np.array(hw)[s], 95):>9.1f}{band_violation(hd):>15.3f}"
        f"  [{min(hd):.2f},{max(hd):.2f}]{np.median(ho):>12.2f}"
    )
    for w in W_PEN_GRID:
        pw, pd, po = run_penalty(vag, obs_jit, sc, sim, w, lb, ub)
        print(
            f"{f'penalty L-BFGS w={w:g}':<22}{np.median(np.array(pw)[s]):>11.1f}"
            f"{np.percentile(np.array(pw)[s], 95):>9.1f}{band_violation(pd):>15.3f}"
            f"  [{min(pd):.2f},{max(pd):.2f}]{np.median(po):>12.2f}"
        )
    print(
        "\n=> penalty + L-BFGS-B drops the active-set QP; higher w_pen tightens feasibility."
        " The trade is solve time vs band-violation (and a fully JIT-able unconstrained solve)."
    )


if __name__ == "__main__":
    main()

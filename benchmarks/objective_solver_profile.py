"""Why is soft-min faster than log-det? Isolate objective vs band via solver counters.

Runs the quadrotor receding-horizon loop for the 2x2 of {log-det, soft-min} x {[1,3], [1,2]}
from the example's start with warm-starting, and reports per-solve median iterations (nit),
function evaluations (nfev), wall time, and the maxiter-hit rate. log-det's gradient ~ -1/lambda
blows up near small eigenvalues (ill-conditioned) so SLSQP backtracks more / hits the cap;
soft-min's gradient is softmax-weighted (bounded) so line searches accept and it converges
earlier. The counters show which effect dominates and whether the speedup is the objective or
the (also-changed) distance band.

  uv run python benchmarks/objective_solver_profile.py
"""

from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.controller import RTController
from rt_oac.scenario import build_scenario
from rt_oac.warmstart import warm_guess

STEPS = 120
METRICS = {"logdet": metrics.neg_logdet, "softmin": metrics.neg_softmin_eig}
BANDS = [(1.0, 3.0), (1.0, 2.0)]


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


X0 = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])


def renorm(x):
    x = np.array(x)
    x[3:7] /= np.linalg.norm(x[3:7])
    return x


def run(sc, lo, hi):
    ctrl = RTController(
        sc.cost,
        stlog_dt=sc.stlog_dt,
        lb=np.array(sc.cfg["optim"]["lb"]),
        ub=np.array(sc.cfg["optim"]["ub"]),
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=6,
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=(lo**2, hi**2),
    )
    dt = sc.cfg["sim"]["integrator_dt"]
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.EULER, stepsize=dt)
    )
    x = X0.copy()
    rows = {k: [] for k in ("nit", "nfev", "ms", "success", "viol", "dist")}
    prev_u = None
    for i in range(STEPS):
        res = ctrl.solve(x, warm_guess(sc.reference_guess(i), prev_u))
        prev_u = res.u
        rows["nit"].append(res.nit)
        rows["nfev"].append(res.nfev)
        rows["ms"].append(res.wall_time * 1e3)
        rows["success"].append(bool(res.success))
        rows["viol"].append(res.max_constraint_violation)
        rows["dist"].append(float(np.linalg.norm(x[0:3])))
        x = renorm(sim(jnp.asarray(x), jnp.asarray(res.u[0]))[0])
    return {k: np.array(v) for k, v in rows.items()}


def main():
    scenarios = {k: build_scenario(gramian_metric=v) for k, v in METRICS.items()}
    print(
        f"{'objective':9s} {'band':10s} | {'nit':>7s} {'nfev':>7s} {'ms':>7s} "
        f"{'%cap':>6s} {'dist[min,max]':>16s}"
    )
    print("-" * 72)
    for obj, sc in scenarios.items():
        for lo, hi in BANDS:
            r = run(sc, lo, hi)
            s = slice(1, None)  # drop tick-0 compile
            cap = 100.0 * np.mean(r["nit"][s] >= 6)
            print(
                f"{obj:9s} [{lo:.0f},{hi:.0f}]      | "
                f"{np.median(r['nit'][s]):7.1f} {np.median(r['nfev'][s]):7.1f} "
                f"{np.median(r['ms'][s]):7.1f} {cap:5.0f}% "
                f"[{r['dist'].min():6.2f},{r['dist'].max():6.2f}]"
            )
    print(
        "\nnit=major iters (cap 6), nfev=objective evals incl. line search, %cap=fraction"
        " hitting the 6-iter cap. Lower nfev/nit and lower %cap => the objective is easier"
        " for SLSQP (fewer backtracks, earlier convergence)."
    )


if __name__ == "__main__":
    main()

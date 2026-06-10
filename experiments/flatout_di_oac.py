r"""Derivative-bounded range-only OA planner -- TEST of "plan one order higher to fix trackability".

Hypothesis: the rung-3a bridge's ~1 m quad tracking error comes from the velocity-bounded OA plan being
too aggressive (peak ~2.3 g accel, ~4.6 g jerk). So plan ONE integration order higher -- a DOUBLE-
INTEGRATOR flat-output model with ACCELERATION as the bounded input -- to get a smooth/feasible plan +
a clean accel feedforward (rung 3a had to finite-difference it).

VERDICT (8-seed bridge head-to-head, see PROGRESS section 9): the hypothesis FAILS -- this is DOMINATED
and UNNECESSARY:
  1. Accel-bounding per component still allows 1.4 g magnitude and makes the plan JERKIER (the optimizer
     just shifts the aggressiveness to the next derivative; jerk max 88.7 > the velocity-bounded 45.7).
  2. A jerk penalty (--jerk LAMBDA) does not cleanly help -- the within-window penalty does not constrain
     the APPLIED receding-horizon trajectory across solves.
  3. At the SAME quad tracking error (~0.6 m), simply GENTLING the velocity-bounded plan (vscale 0.7)
     beats this: 4.2x localization benefit vs the DI's 2.7x. Scaling the velocity bound preserves the
     observability-effective maneuver GEOMETRY; re-planning in accel space does not.
  4. It is moot anyway: the AGGRESSIVE velocity-bounded plan transfers 6.3x DESPITE ~1 m (max 2.4 m)
     tracking error -- the OA benefit comes from the quad maneuvering vigorously, not from precise
     tracking, so it is robust to tracking error. The "trackability gap" is a formation-keeping cost
     during the maneuver, NOT a blocker on the localization benefit.
Takeaway: the OA-vs-trackability tradeoff is a fundamental Pareto traced by the velocity-bound scale;
don't add integrator order. Kept as the experiment that establishes this.

State x = [leader_pos(3), follower_pos(3), follower_vel(3)] (9); input u = [leader_vel(3) (held cruise),
follower_accel(3)] (6); dynamics xdot = [lvel, fvel, faccel]. RANGE-only obs = [leader_pos, |r|]; psi is
dropped (yaw is inert in range-only). softmin-eig STLOG + bounded standoff + symmetry-break. Dumps the
reference [follower_pos(3), psi=0, follower_vel(3), leader_pos(3)] for experiments/flatout_bridge.py.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/flatout_di_oac.py --dump /tmp/oa_di.npz
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator, observability_cost

from rt_oac import metrics
from rt_oac.controller import RTController

DT, STLOG_DT, WINDOW, STEPS, ORDER = 0.1, 0.2, 5, 120, 2
VAR = np.r_[np.full(3, 1e-1), 1e-2]  # [leader pos(3), range]
X0 = np.array([
    0.0,
    0,
    0,
    0,
    5,
    0,
    1,
    0,
    0,
])  # lpos(0,0,0), fpos(0,5,0), fvel(1,0,0) cruising +x
LEADER_VEL = np.array([1.0, 0.0, 0.0])
ACC_BOUND = 8.0  # m/s^2 follower-acceleration bound (sub-1g; quad lateral limit ~9.8)
DIST_BOUNDS = (4.5, 5.5)


def dynamics(x, u):  # xdot = [lvel, fvel, faccel]
    return jnp.concatenate([u[0:3], x[6:9], u[3:6]])


def observation(x, u=None):
    return jnp.array([x[0], x[1], x[2], jnp.linalg.norm(x[3:6] - x[0:3])])


def interrobot_distance(x0, us, cost):
    xs, _ = cost.eval_integrator(x0, us)
    return jnp.linalg.norm(xs[:, 3:6] - xs[:, 0:3], axis=1)


def build_di_oac(jerk_lambda=0.0):
    base = observability_cost.ObservabilityCost(
        dynamics,
        observation,
        DT,
        gramian_kw={"order": ORDER, "var": VAR},
        gramian_metric=metrics.neg_softmin_eig,
        observed_indices=(),
    )
    cost = base if jerk_lambda <= 0 else _JerkCost(base, jerk_lambda)
    return RTController(
        cost,
        stlog_dt=STLOG_DT,
        lb=-ACC_BOUND * np.ones(3),
        ub=ACC_BOUND * np.ones(3),
        n_inputs=6,
        follower_indices=(3, 4, 5),
        method="SLSQP",
        maxiter=6,
        constraint=interrobot_distance,
        constraint_bounds=DIST_BOUNDS,
        constraint_mode="hard",
    )


class _JerkCost:
    """STLOG observability cost + a jerk penalty lambda * sum |Delta faccel|^2 over the window."""

    def __init__(self, base, lam):
        self._base, self._lam = base, lam

    def eval_integrator(self, x0, us):
        return self._base.eval_integrator(x0, us)

    def eval_gramian(self, x, u, dt, *a):
        return self._base.eval_gramian(x, u, dt, *a)

    def __call__(self, x0, us, dt):
        v = self._base(x0, us, dt)
        fa = jnp.asarray(us)[:, 3:6]
        jerk = jnp.sum((fa[1:] - fa[:-1]) ** 2)
        return v._replace(objective=v.objective + self._lam * jerk)


def generate(ctrl, steps):
    """Plan the OA orbit on truth (perfect-feedback). Returns the bridge reference + accel sequence."""
    sim = jax.jit(integrator.Integrator(dynamics, integrator.Methods.RK4, stepsize=DT))
    x = X0.copy()
    prev, ref, accel = None, [], []
    for _i in range(steps):
        guess = np.tile(np.r_[LEADER_VEL, np.zeros(3)], (WINDOW, 1))
        if prev is not None:
            guess[:, 3:6] = np.vstack([prev[1:, 3:6], prev[-1:, 3:6]])
        else:  # symmetry-break the straight-line (zero-accel) critical point of the OA objective
            kk = np.arange(WINDOW)
            guess[:, 3:6] += 3.0 * np.c_[np.sin(kk), np.cos(kk), np.sin(kk + 2.0)]
        res = ctrl.solve(jnp.asarray(x), jnp.asarray(guess))
        prev = res.u
        fa = np.asarray(res.u[0, 3:6])
        accel.append(fa)
        ref.append(
            np.r_[x[3:6], 0.0, x[6:9], x[0:3]]
        )  # [fpos, psi=0, fvel, lpos] for flatout_bridge
        x = np.array(sim(jnp.asarray(x), jnp.asarray(np.r_[LEADER_VEL, fa]))[0])
    return np.array(ref), np.array(accel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump",
        required=True,
        help="npz: OA reference for experiments/flatout_bridge.py",
    )
    ap.add_argument(
        "--jerk",
        type=float,
        default=0.0,
        help="jerk penalty lambda (0 = accel-bound only)",
    )
    args = ap.parse_args()
    ref, accel = generate(build_di_oac(args.jerk), STEPS)
    np.savez(args.dump, ref=ref, dt=DT)
    acc_mag = np.linalg.norm(accel, axis=1)
    jerk_mag = np.linalg.norm(np.diff(accel, axis=0), axis=1) / DT
    speed = np.linalg.norm(ref[:, 4:7], axis=1)
    print(
        f"dumped DI OA reference {ref.shape} -> {args.dump} (jerk_lambda={args.jerk})"
    )
    print(
        f"  mean speed {speed.mean():.2f} m/s | accel: mean {acc_mag.mean():.2f}, max {acc_mag.max():.2f} "
        f"m/s^2 ({acc_mag.max() / 9.81:.2f} g) | jerk: mean {jerk_mag.mean():.1f}, max {jerk_mag.max():.1f} m/s^3"
    )
    print(
        "  (rung-3a velocity-bounded plan for comparison: accel max ~22.5 m/s^2 = 2.3 g, jerk max ~45.7)"
    )


if __name__ == "__main__":
    main()

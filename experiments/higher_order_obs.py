r"""Does HIGHER-ORDER observability make the structurally-invisible (tangential) coordinate observable?

Hypothesis (to examine, before climbing to O1): the single-range tangential is observable only at high
Lie order via motion (the #14 ridge). So a 2nd-ORDER O0 model (double-integrator: velocity becomes a
STATE, accel the input) + a HIGHER-ORDER STLOG (more Lie derivatives) should make the STLOG *see* the
tangential as observable -- and richer Lie derivatives (the velocity state propagates) should help.

This probe computes the accumulated STLOG over a short TANGENTIALLY-maneuvering window and reports the
tangential follower-position 1-sigma (from the STLOG pseudo-inverse) as a function of STLOG order, for:
  - O0-1st: flat single-integrator [lpos(3), fpos(3)], input = velocity (the current OA model);
  - O0-2nd: flat double-integrator [lpos(3), fpos(3), fvel(3)], input = accel (velocity is a STATE).
Range-only obs = [leader_pos(3), |r|]. If the tangential 1-sigma SHRINKS with order (esp. for O0-2nd),
higher-order observability structurally helps; if it is ~flat, the invisibility is an information limit
(only fixable by more measurements / a 2nd anchor), not an order artifact.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/higher_order_obs.py
"""

import argparse

import jax.numpy as jnp
import numpy as np
from observability_aware_control import observability_cost

DT = 0.1
WINDOW = 5
VAR = np.r_[np.full(3, 1e-1), 1e-2]  # [leader pos(3), range]
LPOS = np.array([0.0, 0.0, 0.0])
FPOS = np.array([
    0.0,
    5.0,
    0.0,
])  # follower 5 m along +y (radial = y; tangentials = x, z)
LVEL = np.array([1.0, 0.0, 0.0])  # leader cruises +x
FREL_TANG = np.array([
    1.0,
    0.0,
    0.0,
])  # follower RELATIVE tangential velocity (along +x): excites x-tangential
RHAT = (FPOS - LPOS) / np.linalg.norm(FPOS - LPOS)
THAT = np.array([
    1.0,
    0.0,
    0.0,
])  # the structurally-weak (tangential) direction we maneuver along + probe


def dyn1(x, u):
    return jnp.asarray(u)  # [lvel(3), fvel(3)]


def obs1(x, u=None):
    return jnp.array([x[0], x[1], x[2], jnp.linalg.norm(x[3:6] - x[0:3])])


def dyn2(x, u):
    return jnp.concatenate([
        u[0:3],
        x[6:9],
        u[3:6],
    ])  # lpos'=lvel, fpos'=fvel, fvel'=faccel


def obs2(x, u=None):
    return jnp.array([x[0], x[1], x[2], jnp.linalg.norm(x[3:6] - x[0:3])])


def tangential_sigma(cost, x0, us, foll_idx, dt):
    """Accumulated STLOG over the window -> tangential follower-position 1-sigma (from pinv)."""
    xs, _ = cost.eval_integrator(jnp.asarray(x0), jnp.asarray(us))
    g = np.array([
        np.asarray(cost.eval_gramian(xs[k], us[k], dt)) for k in range(len(us))
    ])
    acc = 0.5 * (g + np.swapaxes(g, -1, -2)).sum(0)  # accumulated STLOG (symmetric)
    cov = np.linalg.pinv(acc, rcond=1e-12)  # information -> covariance
    fcov = cov[np.ix_(foll_idx, foll_idx)]  # follower-position block
    var_t = float(THAT @ fcov @ THAT)  # variance along the tangential
    var_r = float(
        RHAT @ fcov @ RHAT
    )  # variance along the radial (observed) for reference
    mineig = float(np.linalg.eigvalsh(acc)[0])
    return np.sqrt(max(var_t, 0.0)), np.sqrt(max(var_r, 0.0)), mineig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dt",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 1.0],
        help="STLOG dt(s) to sweep (operating ~0.1-0.2; large = where the short-time approx breaks)",
    )
    args = ap.parse_args()
    x0_1 = np.r_[LPOS, FPOS]
    us_1 = np.tile(
        np.r_[LVEL, LVEL + FREL_TANG], (WINDOW, 1)
    )  # follower moves tangentially
    x0_2 = np.r_[
        LPOS, FPOS, LVEL + FREL_TANG
    ]  # velocity is a STATE (set to the tangential cruise)
    us_2 = np.tile(
        np.r_[LVEL, np.zeros(3)], (WINDOW, 1)
    )  # constant velocity (accel input = 0)

    print(
        "=== Higher-order observability probe: tangential follower-position 1-sigma vs STLOG order ==="
    )
    print(
        "(accumulated STLOG over a 5-step tangential maneuver; lower tangential 1sig = more observable)\n"
    )
    for dt in args.dt:
        print(f"--- STLOG dt = {dt} ---")
        print(
            f"{'model':>10} {'order':>6} {'tang 1sig':>11} {'radial 1sig':>12} {'tang/radial':>12} {'min-eig':>11}"
        )
        for name, dyn, obs, x0, us, fidx in [
            ("O0-1st", dyn1, obs1, x0_1, us_1, [3, 4, 5]),
            ("O0-2nd", dyn2, obs2, x0_2, us_2, [3, 4, 5]),
        ]:
            for order in (1, 2, 3, 4, 5):
                cost = observability_cost.ObservabilityCost(
                    dyn,
                    obs,
                    dt,
                    gramian_kw={"order": order, "var": VAR},
                    gramian_metric=lambda g: g,
                    observed_indices=(),
                )
                st, sr, me = tangential_sigma(cost, x0, us, fidx, dt)
                print(
                    f"{name:>10} {order:>6} {st:>11.3f} {sr:>12.4f} {st / (sr + 1e-12):>12.1f} {me:>11.2e}"
                )
            print()
    print(
        "READ: if tang 1sig drops sharply with order (esp. O0-2nd), higher-order observability "
        "structurally reveals the invisible coordinate. If ~flat / still >> radial, the tangential is "
        "information-limited (needs a 2nd anchor), not order-limited -- and a 1st-order EKF can't exploit "
        "whatever the high-order STLOG sees anyway (cf. the UKF findings)."
    )


if __name__ == "__main__":
    main()

r"""(E1-lean, O1) CLOSED, the robust way: plan FLAT, score the O1 STLOG, on the clins lean-INS loop.

Assembles the robust corner identified in §6.10/6.11: the O1 *objective* (full-quad relative-pose STLOG via
the differentiable flatness map, `o1_flat`) driving the VALIDATED control+estimation architecture (clins
flat-plan + geometric tracker + lean translation-INS + 2-range fusion, 95% at O0). We wrap the O1-on-flat
objective in a controller with the `solve(x_flat, guess)` interface `clins_closed_loop.run` expects
(projected gradient on the follower's flat velocities + a formation-distance penalty + velocity bounds), so
the lean INS, tracker, and 2-range fusion are reused unchanged. Compares O1-on-flat vs the O0 baseline.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/o1_flat_closed.py --seeds 8
"""

import argparse
from collections import namedtuple
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import observability_cost

from rt_oac import metrics
from rt_oac.controller import RTController

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import clins_closed_loop as clins
import o1_flat

D12 = np.array([
    0.0,
    10.0,
    0.0,
])  # 2nd leader (lateral, the clins 95% placement) -- ABSOLUTE offset
o1_flat.LEADER_V = jnp.asarray(
    clins.LEADER_VEL
)  # the O1 STLOG must see the leader's cruise velocity
o1_flat.D12_ABS = jnp.asarray(
    D12
)  # ... and the 2nd-leader offset (matches the INS's l2_off)

WINDOW, DT = clins.WINDOW, clins.DT
FB_LB, FB_UB = clins.FB_LB[:3], clins.FB_UB[:3]  # follower velocity bounds
BAND = (4.5, 5.5)  # formation-distance band (squared penalty outside)
DIST_PEN, SMOOTH_PEN, N_GRAD, STEP = 30.0, 8.0, 8, 0.02
_Q_L = jnp.array([0.0, 0.0, 0.0, 1.0])
Res = namedtuple("Res", ["u", "wall_time"])


def _dist_penalty(v_w, p_f0, p_l0):
    t = jnp.arange(v_w.shape[0])[:, None] * DT
    p_f = p_f0 + jnp.cumsum(v_w, axis=0) * DT
    p_l = p_l0 + o1_flat.LEADER_V * t
    d = jnp.linalg.norm(p_f - p_l, axis=1)
    return jnp.sum(jax.nn.relu(d - BAND[1]) ** 2 + jax.nn.relu(BAND[0] - d) ** 2)


# --- the two objectives, scored on the SAME flat-velocity plan + SAME optimizer (disentangle obj vs solver) ---
def _o1_obj(v_w, p_f0, p_l0):  # full-quad relative-pose STLOG via the flatness map
    return o1_flat.o1_on_flat(v_w, p_f0, _Q_L, p_l0)


_o0 = observability_cost.ObservabilityCost(
    clins.flat_dyn,
    clins.flat_obs,
    clins.DT,
    gramian_kw={"order": clins.ORDER, "var": clins.VAR},
    gramian_metric=metrics.neg_softmin_eig,
    observed_indices=(),
)


def _o0_obj(
    v_w, p_f0, p_l0
):  # the O0 baseline objective (flat model xdot=u, range; matches build_oac)
    x0 = jnp.concatenate([
        p_l0,
        jnp.zeros(1),
        p_f0,
        jnp.zeros(1),
    ])  # 8-state flat [leader1, follower]
    foll = v_w[:WINDOW]
    us = jnp.concatenate(
        [
            jnp.broadcast_to(o1_flat.LEADER_V, (WINDOW, 3)),
            jnp.zeros((WINDOW, 1)),
            foll,
            jnp.zeros((WINDOW, 1)),
        ],
        axis=1,
    )
    return _o0(x0, us, o1_flat.STLOG_DT).objective


def _make_pen(obj_fn):
    def pen(v_w, p_f0, p_l0):
        # observability + hold the formation + keep the plan smooth (bounded accel -> feasible flatness map,
        # no jagged commands for the tracker to chase)
        smooth = jnp.sum(jnp.diff(v_w, axis=0) ** 2)
        return (
            obj_fn(v_w, p_f0, p_l0)
            + DIST_PEN * _dist_penalty(v_w, p_f0, p_l0)
            + SMOOTH_PEN * smooth
        )

    return jax.jit(jax.grad(pen))


class FlatPGController:
    """RTController-compatible (solve(x_flat, guess) -> Res): optimizes the follower FLAT velocities by
    projected gradient on a chosen observability objective (O1 via the flatness map, or O0 baseline) +
    formation-hold + smoothness. Same optimizer for both -> isolates objective from solver."""

    def __init__(self, obj_fn):
        self._grad = _make_pen(obj_fn)
        self.prev = None

    def solve(self, x0_flat, guess):
        x0, g = np.asarray(x0_flat), np.asarray(guess)
        p_l0, p_f0 = x0[0:3], x0[4:7]
        if self.prev is not None:
            v_w = self.prev
        else:  # symmetry-break the cruise critical point of the observability objective
            v_w = np.tile(clins.LEADER_VEL, (WINDOW + 2, 1))
            kk = np.arange(WINDOW + 2)
            v_w += 0.2 * np.c_[np.sin(kk), np.cos(kk), np.sin(kk + 2.0)]
        for _ in range(N_GRAD):
            gr = np.asarray(
                self._grad(jnp.asarray(v_w), jnp.asarray(p_f0), jnp.asarray(p_l0))
            )
            if not np.all(
                np.isfinite(gr)
            ):  # degenerate-STLOG NaN: skip this step, keep last feasible plan
                break
            v_w = np.clip(v_w - STEP * gr, FB_LB, FB_UB)
        self.prev = np.vstack([v_w[1:], v_w[-1:]])  # warm-start shift
        u = g.copy()
        u[:, 4:7] = v_w[:WINDOW]
        return Res(u=jnp.asarray(u), wall_time=0.0)


# --- O1 objective on the SAME robust solver as O0 (SLSQP + hard distance constraint) -- the fair test ---
_ObjResult = namedtuple("_ObjResult", ["objective"])


class O1FlatCost:
    """ObservabilityCost-compatible (``cost(x0, us, dt).objective`` + ``eval_integrator``): the full-quad O1
    STLOG via the flatness map, scored on a flat-output velocity plan. Drop-in for ``build_oac``'s flat O0
    cost, so the O1 objective runs through the identical RTController (SLSQP, maxiter 6, hard distance
    constraint) -- the same solver as O0, the only thing that differs is the objective."""

    def __init__(self):
        # delegate the rollout (for the interrobot-distance constraint) to the flat O0 model (xdot = u)
        self._flat = observability_cost.ObservabilityCost(
            clins.flat_dyn,
            clins.flat_obs,
            clins.DT,
            gramian_kw={"order": clins.ORDER, "var": clins.VAR},
            gramian_metric=metrics.neg_softmin_eig,
            observed_indices=(),
        )

    def eval_integrator(self, x0, us):
        return self._flat.eval_integrator(x0, us)

    def __call__(self, x0, us, dt):
        p_l0, p_f0 = x0[0:3], x0[4:7]
        v_foll = us[:, 4:7]  # (window, 3) decision velocities
        v_w = jnp.concatenate(
            [v_foll, v_foll[-1:], v_foll[-1:]], axis=0
        )  # pad for the jerk finite-diff
        return _ObjResult(objective=o1_flat.o1_on_flat(v_w, p_f0, _Q_L, p_l0))


def build_o1_rt(maxiter=6):
    return RTController(
        O1FlatCost(),
        stlog_dt=o1_flat.STLOG_DT,
        lb=clins.FB_LB,  # full [vx,vy,vz,yaw-rate] -- matches the 4 follower indices
        ub=clins.FB_UB,
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=maxiter,
        constraint=clins.interrobot_distance,
        constraint_bounds=clins.DIST_BOUNDS,
        constraint_mode="hard",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument(
        "--only", choices=["o1rt", "o1flat", "o0pg", "o0", "all"], default="all"
    )
    ap.add_argument("--o1-maxiter", type=int, default=6)
    args = ap.parse_args()
    print(
        "=== (E1-lean, O1) CLOSED: plan-flat / score-O1 on the clins lean-INS + tracker + 2-range loop ==="
    )
    print(
        f"{args.seeds} seeds, 2 leaders (lateral D12={D12.tolist()}); bounded = recovery<1.5 m AND "
        f"formation<15 m. Compares O1-on-flat objective vs the O0 baseline.\n"
    )
    print(
        f"{'config':>26} {'rec_med':>8} {'rec_p90':>8} {'NEES':>7} {'%bnd':>6} {'distMed':>8}"
    )
    cfgs = []
    if args.only in {"o1rt", "all"}:
        cfgs.append((
            f"O1-on-flat (SLSQP m{args.o1_maxiter})",
            build_o1_rt(args.o1_maxiter),
        ))
    if args.only in {"o1flat", "all"}:
        cfgs.append(("O1-on-flat (PG)", FlatPGController(_o1_obj)))
    if args.only in {"o0pg", "all"}:
        cfgs.append(("O0-on-flat (PG, same solver)", FlatPGController(_o0_obj)))
    if args.only in {"o0", "all"}:
        cfgs.append(("O0 baseline (SLSQP)", clins.build_oac()))
    for label, ctrl in cfgs:
        rows = []
        for s in range(args.seeds):
            errs, nees, dists, *_ = clins.run(
                "oac", ctrl, s, driven="imu", l2_off=D12, oac2=False
            )
            bounded = errs[-1] < 1.5 and np.max(dists) < 15.0
            rows.append((errs[-1], np.median(nees), bounded, np.median(dists)))
        r = np.array(rows)
        print(
            f"{label:>26} {np.median(r[:, 0]):>8.2f} {np.percentile(r[:, 0], 90):>8.2f} "
            f"{np.median(r[:, 1]):>7.1f} {100 * np.mean(r[:, 2]):>5.0f}% {np.median(r[:, 3]):>8.2f}"
        )
    print(
        "\nVERDICT: does O1-on-flat (full-quad STLOG via the flatness map, on the robust lean-INS+tracker "
        "loop) match/beat the O0 baseline -- i.e. does the O1 objective add value at the robust operating "
        "point, or is O0 already sufficient with the lean estimator?"
    )


if __name__ == "__main__":
    main()

r"""Ladder rung 3a (flat-output OA): plan OA in the quadrotor FLAT-OUTPUT space [x, y, z, psi].

Premise (differential flatness, Mellinger-Kumar): a quad's flat outputs are sigma = [x, y, z, psi];
any smooth sigma(t) is dynamically feasible, the full state/inputs being algebraic in sigma and its
derivatives. So the principled OA-planning model is the FLAT-OUTPUT kinematics -- the smallest state
that is both the quad's genuine planning space and guaranteed quad-realizable (the attitude dynamics
are the subservient tracker's job, not the OA problem's). This is rung 2 (3D point-mass) + a yaw state.

World-frame velocity (yaw-decoupled, faithful to a quad) so dynamics(x,u)=u for x=[x,y,z,psi] per agent.
Range => the range does NOT depend on psi, so psi is OA-inert (a free flat output along for the ride) --
correct for the RANGE-ONLY JGCD paper, where yaw is a symmetry of the objective. (A yaw-sensitive
bearing/FOV sensor would make psi an active lever, but bearing is a CONFOUND vs the range-only paper and
is OFF the table.) The yaws are MEASURED (IMU), so psi is bounded/known; the follower POSITION is
observable only via range (radial range-observed, 2 tangential motion-observable) -- identical OA
structure to rung 2, now embedded in the flat-output state.

Carries the rung-2 recipe (softmin-eig + bounded distance + symmetry-break). Also dumps an OA flat-
output reference trajectory (--dump) for the quad-tracking bridge (experiments/flatout_bridge.py).

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/flatout_oac_validate.py --seeds 20
"""

import argparse

from example_lib.misc import simple_ekf
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator, observability_cost

from rt_oac import metrics
from rt_oac.controller import RTController

DT, STLOG_DT, WINDOW, STEPS = 0.1, 0.2, 5, 120
ORDER = 2
# state per agent = [x, y, z, psi]; 2 agents stacked -> 8. obs = [leader_pos(3), psi_l, psi_f, range].
VAR = np.r_[np.full(3, 1e-1), 1e-2, 1e-2, 1e-2]
INPUT_VAR = np.tile([1e-2, 1e-2, 1e-2, 1e-3], 2)  # [vx,vy,vz, yaw-rate] x 2 agents
X0 = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    5.0,
    0.0,
    0.0,
])  # leader [0,0,0,psi0], follower [0,5,0,psi0]
LEADER_U = np.array([1.0, 0.0, 0.0, 0.0])  # leader cruises +x, no yaw rate
DIST_BOUNDS = (0.2, 7.0)
P0 = np.diag([
    1e-3,
    1e-3,
    1e-3,
    1e-3,
    2.0,
    2.0,
    2.0,
    1e-2,
])  # large only on follower position
FB_LB = np.array([-1.0, -2.0, -2.0, -2.0])  # follower [vx,vy,vz, yaw-rate] bounds
FB_UB = np.array([3.0, 2.0, 2.0, 2.0])
METRIC = metrics.neg_softmin_eig


def dynamics(x, u):
    return jnp.asarray(
        u
    )  # flat-output kinematics: xdot = u (world-frame vel + yaw rate)


def observation(x, u=None):
    lp, psi_l, fp, psi_f = x[0:3], x[3], x[4:7], x[7]
    rng = jnp.linalg.norm(fp - lp)
    return jnp.array([lp[0], lp[1], lp[2], psi_l, psi_f, rng])


def interrobot_distance(x0, us, cost):
    xs, _ = cost.eval_integrator(x0, us)
    return jnp.linalg.norm(xs[:, 4:7] - xs[:, 0:3], axis=1)


_r = X0[4:7] - X0[0:3]
RHAT = _r / np.linalg.norm(_r)
THAT_H = np.cross(RHAT, np.array([0.0, 0.0, 1.0]))
THAT_H /= np.linalg.norm(THAT_H)
THAT_V = np.cross(RHAT, THAT_H)
THAT_V /= np.linalg.norm(THAT_V)


def _cost():
    return observability_cost.ObservabilityCost(
        dynamics,
        observation,
        DT,
        gramian_kw={"order": ORDER, "var": VAR},
        gramian_metric=METRIC,
        observed_indices=(),
    )


def build_oac(dist_bounds=DIST_BOUNDS):
    return RTController(
        _cost(),
        stlog_dt=STLOG_DT,
        lb=FB_LB,
        ub=FB_UB,
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=6,
        constraint=interrobot_distance,
        constraint_bounds=dist_bounds,
        constraint_mode="hard",
    )


def run_once(mode, ctrl, seed, offset, *, dump=False):
    rng = np.random.default_rng(seed)
    sim = jax.jit(integrator.Integrator(dynamics, integrator.Methods.RK4, stepsize=DT))
    ekf = simple_ekf.SimpleEKF(
        lambda x, u, dt: x + dt * dynamics(x, u),
        observation,
        in_cov=np.diag(INPUT_VAR),
        obs_cov=np.diag(VAR),
    )
    x_true = X0.copy()
    P = P0.copy()
    x_hat = (
        x_true + np.r_[0, 0, 0, 0, offset, 0.0]
    )  # follower-position offset (indices 4,5,6)
    prev_u, errs, neess, dists, ref = None, [], [], [], []
    for _i in range(STEPS):
        if mode == "noac":
            u_foll = LEADER_U.copy()  # drive straight alongside, no yaw rate
        else:
            guess = np.tile(np.r_[LEADER_U, LEADER_U], (WINDOW, 1))
            if prev_u is not None:
                guess[:, 4:8] = np.vstack([prev_u[1:, 4:8], prev_u[-1:, 4:8]])
            else:  # break the log-det/observability critical point (rung-2 lesson)
                kk = np.arange(WINDOW)
                guess[:, 4:7] += 0.3 * np.c_[np.sin(kk), np.cos(kk), np.sin(kk + 2.0)]
            res = ctrl.solve(x_hat, guess)
            prev_u = res.u
            u_foll = np.asarray(res.u[0, 4:8])
        u = np.r_[LEADER_U, u_foll]
        u_applied = u + np.r_[np.zeros(4), rng.normal(0, np.sqrt(INPUT_VAR[4:]))]
        x_hat, P = ekf.predict(
            jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u_applied), DT
        )
        x_true = np.array(sim(jnp.asarray(x_true), jnp.asarray(u_applied))[0])
        y = np.array(observation(jnp.asarray(x_true))) + rng.normal(0, np.sqrt(VAR))
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)
        ef = x_hat[4:7] - x_true[4:7]
        errs.append(float(np.linalg.norm(ef)))
        dists.append(float(np.linalg.norm(x_true[4:7] - x_true[0:3])))
        Pf = 0.5 * (P[4:7, 4:7] + P[4:7, 4:7].T)
        neess.append(float(ef @ np.linalg.solve(Pf + 1e-9 * np.eye(3), ef)))
        if dump:  # the OA flat-output reference: follower [x,y,z,psi] + world-frame velocity command
            ref.append(
                np.r_[x_true[4:8], u_applied[4:7], x_true[0:3]]
            )  # +leader pos for the range
    errs = np.array(errs)
    out = (
        errs[-1],
        float(np.sqrt((errs**2).mean())),
        float(np.median(neess[20:])),
        bool(errs[-1] < 5),
        float(np.mean(dists)),
    )
    return (out, np.array(ref)) if dump else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--mag", type=float, default=2.26)
    ap.add_argument(
        "--dump",
        help="npz path: OA flat-output reference traj (seed 0, tang_horiz) for the bridge",
    )
    ap.add_argument(
        "--vscale",
        type=float,
        default=1.0,
        help="scale the follower velocity/yaw-rate bounds (<1 = gentler, more trackable plan)",
    )
    args = ap.parse_args()
    global FB_LB, FB_UB
    FB_LB, FB_UB = FB_LB * args.vscale, FB_UB * args.vscale

    dirs = {
        "radial": args.mag * RHAT,
        "tang_horiz": args.mag * THAT_H,
        "tang_vert": args.mag * THAT_V,
    }
    ctrls = {
        "noac": None,
        "oac": build_oac(),
        "oac_tight": build_oac(dist_bounds=(4.5, 5.5)),
    }

    if args.dump:
        (_, ref) = run_once(
            "oac_tight", ctrls["oac_tight"], 0, args.mag * THAT_H, dump=True
        )
        np.savez(args.dump, ref=ref, dt=DT)
        print(f"dumped OA flat-output reference {ref.shape} -> {args.dump}")
        return

    print(
        f"flat-output [x,y,z,psi] OA (rung 3a): {args.seeds} seeds, |err0|={args.mag:.2f} m, "
        f"order {ORDER} (NEES dof=3 follower-position; stable=final<5 m)\n"
    )
    for dname, off in dirs.items():
        print(f"--- initial follower-position error: {dname} ---")
        print(
            f"{'mode':>9} {'final_med':>10} {'final_p90':>10} {'NEES_med':>9} {'%stable':>8} {'meandist':>9}"
        )
        base = None
        for mode, ctrl in ctrls.items():
            R = np.array([run_once(mode, ctrl, s, off) for s in range(args.seeds)])
            fe = R[:, 0]
            if mode == "noac":
                base = np.median(fe)
            tag = "" if mode == "noac" else f"  ({base / np.median(fe):.1f}x)"
            print(
                f"{mode:>9} {np.median(fe):>10.3f} {np.percentile(fe, 90):>10.3f} "
                f"{np.median(R[:, 2]):>9.1f} {100 * np.mean(R[:, 3]):>7.0f}% {np.median(R[:, 4]):>9.2f}{tag}"
            )
        print()


if __name__ == "__main__":
    main()

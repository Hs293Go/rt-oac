"""Can we speed the leader up to cover more ground without breaking the orbit?

The companion/paper leader is a min-snap cruise of 10 m over 120 s (peak ~0.16 m/s), so over
the example's 15 s window it moves ~0.16 m -- visually frozen. The ``inter_quadrotor_pose``
relative dynamics are Galilean-invariant: a common constant translation velocity cancels in
the relative state, so replacing the leader with a faster *level* constant-velocity cruise
should leave the follower's orbit, observability, and solve time unchanged and only stretch
the absolute scene into a helix. This script verifies that empirically.

Runs the soft-min [1,2] quadrotor rollout (the chosen headline config) against the min-snap
baseline and against synthesized level cruises at several speeds, and checks that the relative
orbit metrics (revs, range, observability, ms, max constraint violation) are invariant while
the ground covered scales with speed. Saves results/leader_speed_sweep.png.

  uv run python benchmarks/leader_speed_sweep.py
"""

import dataclasses
import pathlib

from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from observability_aware_control import integrator

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.controller import RTController
from rt_oac.scenario import build_scenario
from rt_oac.warmstart import warm_guess

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
STEPS = 300
MIN_DIST, MAX_DIST = 1.0, 2.0


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


X0 = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])


def renorm(x):
    x = np.array(x)
    x[3:7] /= np.linalg.norm(x[3:7])
    return x


def const_vel_leader(v, n, dt, z0=10.0):
    """A level constant-velocity cruise along +x: identity attitude, hover thrust."""
    t = np.arange(n) * dt
    xl = np.zeros((n, 10))
    xl[:, 0] = v * t
    xl[:, 2] = z0
    xl[:, 6] = 1.0  # quat w (xyzw identity)
    xl[:, 7] = v  # vx
    ul = np.tile([9.81, 0.0, 0.0, 0.0], (n, 1))
    return xl, ul


def run(sc, grams, label):
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
        constraint_bounds=(MIN_DIST**2, MAX_DIST**2),
    )
    dt = sc.cfg["sim"]["integrator_dt"]
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.EULER, stepsize=dt)
    )
    x = X0.copy()
    rel, foll, dist, walls, ndirs, viols = [], [], [], [], [], []
    prev_u = None
    for i in range(STEPS):
        res = ctrl.solve(x, warm_guess(sc.reference_guess(i), prev_u))
        prev_u = res.u
        walls.append(res.wall_time * 1e3)
        viols.append(res.max_constraint_violation)
        rel.append(x[0:3].copy())
        dist.append(float(np.linalg.norm(x[0:3])))
        x_abs = np.asarray(mdl.to_absolute_state(sc.x_leader[i], jnp.asarray(x)))
        foll.append(x_abs[10:13])
        if i % 20 == 0:
            g = np.asarray(grams(jnp.asarray(x), res.u))
            eig = np.linalg.eigvalsh(0.5 * (g + np.swapaxes(g, -1, -2)).sum(0))
            ndirs.append(int((eig > 1e-6 * eig[-1]).sum()))
        x = renorm(sim(jnp.asarray(x), jnp.asarray(res.u[0]))[0])
    rel = np.array(rel)
    az = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    lead = np.asarray(sc.x_leader[:STEPS, 0:3])
    return {
        "label": label,
        "rel": rel,
        "foll": np.array(foll),
        "lead": lead,
        "dist": np.array(dist),
        "revs": float((az[-1] - az[0]) / (2 * np.pi)),
        "ndir": int(np.median(ndirs)),
        "ms": float(np.median(np.array(walls)[1:])),
        "viol": float(np.max(viols)),
        "ground": float(np.linalg.norm(lead[-1] - lead[0])),
    }


def main():
    sc = build_scenario(gramian_metric=metrics.neg_softmin_eig)
    grams = jax.jit(
        lambda x, u: sc.cost(
            x, jnp.asarray(u), sc.stlog_dt, return_gramians=True
        ).gramians
    )
    dt = sc.cfg["sim"]["integrator_dt"]
    n = STEPS + sc.window + 2

    runs = []
    # baseline: the original min-snap leader from the npz
    runs.append(run(sc, grams, "min-snap (OG)"))
    # synthesized level constant-velocity cruises
    for v in (1.0, 2.0, 3.0):
        xl, ul = const_vel_leader(v, n, dt)
        sc = dataclasses.replace(sc, x_leader=xl, u_leader=ul)
        runs.append(run(sc, grams, f"level cruise {v:.0f} m/s"))

    print(
        f"{'leader':18s} | {'revs':>5s} {'dist[min,max]':>15s} {'obs':>5s} {'ms':>6s} "
        f"{'maxviol':>9s} {'ground/15s':>11s}"
    )
    print("-" * 78)
    for r in runs:
        print(
            f"{r['label']:18s} | {r['revs']:+5.1f} "
            f"[{r['dist'].min():5.2f},{r['dist'].max():5.2f}] {r['ndir']:>3d}/6 "
            f"{r['ms']:6.0f} {r['viol']:9.2e} {r['ground']:9.2f} m"
        )
    print(
        "\nIf the relative orbit (revs, dist, obs, ms, viol) is ~constant while ground/15s"
        " scales with speed, speeding up the leader is free (Galilean invariance)."
    )

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        "Leader speed sweep -- soft-min [1,2] orbit is invariant; only the world-frame "
        "helix stretches",
        fontsize=14,
    )
    tt = np.arange(STEPS)
    for i, r in enumerate(runs):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.plot(r["lead"][:, 0], r["lead"][:, 1], r["lead"][:, 2], color="black", lw=2)
        ax.scatter(
            r["foll"][:, 0], r["foll"][:, 1], r["foll"][:, 2], c=tt, cmap="viridis", s=6
        )
        ax.set(
            title=f"{r['label']}\nground {r['ground']:.1f} m, {r['revs']:+.1f} rev, obs {r['ndir']}/6",
            xlabel="x [m]",
            ylabel="y [m]",
            zlabel="z [m]",
        )
    # invariance panel: relative-frame XY orbits overlaid (should coincide)
    ax = fig.add_subplot(2, 3, 5)
    for r in runs:
        ax.plot(r["rel"][:, 0], r["rel"][:, 1], lw=1.0, alpha=0.7, label=r["label"])
    ax.scatter([0], [0], marker="*", c="black", s=120)
    ax.set(
        title="Relative-frame orbits overlaid (invariance check)",
        xlabel="x [m]",
        ylabel="y [m]",
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    # distance overlaid
    ax = fig.add_subplot(2, 3, 6)
    for r in runs:
        ax.plot(tt * dt, r["dist"], lw=1.0, alpha=0.7, label=r["label"])
    ax.axhline(MIN_DIST, color="0.5", ls="--", lw=1)
    ax.axhline(MAX_DIST, color="0.5", ls="--", lw=1)
    ax.set(title="Inter-drone distance overlaid", xlabel="t [s]", ylabel="m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS / "leader_speed_sweep.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

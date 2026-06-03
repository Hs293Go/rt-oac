"""Explore drone observability orbits: long horizon + varied config/constraints.

Perfect-feedback receding-horizon planning (log-det + SLSQP@6), run for 30 s+, sweeping the
initial configuration and the inter-drone distance bounds to see whether clearer orbits
(many revolutions) or clean side-to-side swinging emerge. Tighter max-distance => smaller
radius => faster angular sweep for the same speed. Saves a figure per config and prints a
comparison. Left in the working tree on purpose.

Run the built-in sweep:  uv run python experiments/drone_orbit_explore.py
Single config:           ... --x0 close --min-dist 0.5 --max-dist 1.2 --steps 700 --obj logdet
"""

import argparse
import pathlib
import time

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
METRICS = {"logdet": metrics.neg_logdet, "softmin": metrics.neg_softmin_eig}


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


X0S = {
    "tangential": np.concatenate([[2.0, 0, 0], quat_z(20), [0, 1.0, 0]]),
    "close": np.concatenate([[1.0, 0, 0], quat_z(20), [0, 1.0, 0]]),
    "fast": np.concatenate([[2.0, 0, 0], quat_z(30), [0, 2.0, 0]]),
    "offset": np.concatenate([[1.5, 1.5, 0.5], quat_z(25), [0.3, 1.0, -0.2]]),
    "vertical": np.concatenate([[2.0, 0, 0], quat_z(20), [0, 0.5, 1.0]]),
}


def renorm(x):
    x = np.array(x)
    x[3:7] /= np.linalg.norm(x[3:7])
    return x


def run(x0_name, min_d, max_d, steps, obj):
    sc = build_scenario(gramian_metric=METRICS[obj])
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
        constraint_bounds=(min_d**2, max_d**2),
    )
    dt = sc.cfg["sim"]["integrator_dt"]
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.EULER, stepsize=dt)
    )
    x = X0S[x0_name].copy()
    rel, walls, dist = [], [], []
    prev_u = None
    for i in range(steps):
        guess = warm_guess(sc.reference_guess(i), prev_u)
        t = time.perf_counter()
        res = ctrl.solve(x, guess)
        walls.append((time.perf_counter() - t) * 1e3)
        prev_u = res.u
        rel.append(x[0:3].copy())
        dist.append(float(np.linalg.norm(x[0:3])))
        x = renorm(sim(jnp.asarray(x), jnp.asarray(res.u[0]))[0])
    rel = np.array(rel)
    az = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    daz = np.diff(az)
    sweep_deg = float(np.degrees(az[-1] - az[0]))
    reversals = int(np.sum(np.diff(np.sign(daz + 1e-12)) != 0))
    return {
        "rel": rel,
        "dist": np.array(dist),
        "wall": np.array(walls),
        "sweep_deg": sweep_deg,
        "revs": sweep_deg / 360.0,
        "reversals": reversals,
        "z_range": float(rel[:, 2].max() - rel[:, 2].min()),
        "dt": dt,
        "label": f"{x0_name}_r{min_d}-{max_d}_{obj}",
    }


def plot(r, steps):
    rel, dist = r["rel"], r["dist"]
    tt = np.arange(len(rel)) * r["dt"]
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    s = ax.scatter(rel[:, 0], rel[:, 1], rel[:, 2], c=tt, cmap="viridis", s=8)
    ax.plot(rel[:, 0], rel[:, 1], rel[:, 2], color="0.6", lw=0.6)
    ax.scatter([0], [0], [0], marker="*", c="black", s=160)
    ax.scatter(*rel[0], marker="o", edgecolor="k", facecolor="none", s=70)
    fig.colorbar(s, ax=ax, pad=0.1, label="t [s]")
    ax.set(
        title=f"relative trajectory\n{r['revs']:+.2f} rev, {r['reversals']} reversals",
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(
        tt, np.degrees(np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))), color="tab:purple"
    )
    ax2.set(title="bearing (azimuth) vs time", xlabel="t [s]", ylabel="deg")
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(tt, dist, color="tab:blue")
    ax3.set(
        title=f"distance (range [{dist.min():.2f},{dist.max():.2f}])",
        xlabel="t [s]",
        ylabel="m",
    )
    ax3.grid(alpha=0.3)
    fig.suptitle(
        f"{r['label']}  |  median plan {np.median(r['wall']):.0f} ms", fontsize=12
    )
    fig.tight_layout()
    out = RESULTS / f"orbit_{r['label']}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x0", choices=list(X0S))
    ap.add_argument("--min-dist", type=float, default=0.5)
    ap.add_argument("--max-dist", type=float, default=1.2)
    ap.add_argument("--steps", type=int, default=700)  # 700*0.05 = 35 s
    ap.add_argument("--obj", choices=list(METRICS), default="logdet")
    args = ap.parse_args()

    if args.x0:
        sweep = [(args.x0, args.min_dist, args.max_dist, args.obj)]
    else:  # built-in priority sweep, all ~35 s
        sweep = [
            ("close", 0.5, 1.2, "logdet"),  # tight radius -> clear orbit?
            ("tangential", 1.0, 2.0, "logdet"),  # medium radius
            ("fast", 0.8, 1.5, "logdet"),  # faster follower
            ("close", 0.5, 1.2, "softmin"),  # softmin objective
        ]
    print(
        f"# Drone orbit exploration  (steps={args.steps}, ~{args.steps * 0.05:.0f} s)\n"
    )
    print(
        f"{'config':<34}{'revs':>8}{'reversals':>11}{'z-range':>9}{'dist[min,max]':>16}{'plan ms':>9}"
    )
    for x0_name, mn, mx, obj in sweep:
        r = run(x0_name, mn, mx, args.steps, obj)
        out = plot(r, args.steps)
        print(
            f"{r['label']:<34}{r['revs']:>+8.2f}{r['reversals']:>11}{r['z_range']:>9.2f}"
            f"{f'[{r["dist"].min():.2f},{r["dist"].max():.2f}]':>16}{np.median(r['wall']):>9.0f}  -> {out.name}"
        )


if __name__ == "__main__":
    main()

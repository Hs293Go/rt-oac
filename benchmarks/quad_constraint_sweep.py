"""Pick the prettiest inter-vehicle distance band for the quadrotor headline example.

Holds the example's exact start and horizon fixed and varies only (objective, distance band),
so the result directly answers "which constraint values make the example's orbit prettiest?"
Scores each run on revolutions, azimuth reversals (0 = a clean planar ring), radial tightness
(std/mean of the range -- low = constant radius), and planarity (smallest/largest PCA singular
value -- low = flat). Saves a montage to results/quad_constraint_sweep.png.

  uv run python benchmarks/quad_constraint_sweep.py
"""

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

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
STEPS = 300  # matches the example (15 s)
METRICS = {"logdet": metrics.neg_logdet, "softmin": metrics.neg_softmin_eig}


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


X0 = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])  # example's start


def renorm(x):
    x = np.array(x)
    x[3:7] /= np.linalg.norm(x[3:7])
    return x


# (objective, min_d, max_d); first is the current example baseline.
CONFIGS = [
    ("logdet", 1.0, 3.0),
    ("softmin", 0.5, 1.2),
    ("softmin", 1.2, 1.8),
    ("softmin", 1.8, 2.4),
    ("softmin", 2.4, 3.0),
    ("softmin", 1.0, 2.0),
]


def run(scenarios, grams, obj, lo, hi):
    sc = scenarios[obj]
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
    rel, walls, dist, ndirs = [], [], [], []
    prev_u = None
    for i in range(STEPS):
        guess = sc.reference_guess(i)
        if prev_u is not None:
            guess = np.array(guess)
            guess[:, 4:8] = np.vstack([prev_u[1:, 4:8], prev_u[-1:, 4:8]])
        res = ctrl.solve(x, guess)
        walls.append(res.wall_time * 1e3)
        prev_u = res.u
        rel.append(x[0:3].copy())
        dist.append(float(np.linalg.norm(x[0:3])))
        if (
            i % 20 == 0
        ):  # sample observability sparsely (full Gramian eval is not cheap)
            g = np.asarray(grams[obj](jnp.asarray(x), res.u))
            eig = np.linalg.eigvalsh(0.5 * (g + np.swapaxes(g, -1, -2)).sum(0))
            ndirs.append(int((eig > 1e-6 * eig[-1]).sum()))
        x = renorm(sim(jnp.asarray(x), jnp.asarray(res.u[0]))[0])

    rel = np.array(rel)
    dist = np.array(dist)
    az = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    daz = np.diff(az)
    reversals = int(np.sum(np.diff(np.sign(daz + 1e-12)) != 0))
    # planarity: PCA of the centred steady-state cloud (drop the fly-in transient)
    pts = rel[STEPS // 5 :] - rel[STEPS // 5 :].mean(0)
    sv = np.linalg.svd(pts, compute_uv=False)
    planarity = float(sv[2] / sv[0]) if sv[0] > 0 else 0.0
    steady = slice(STEPS // 5, None)
    return {
        "obj": obj,
        "lo": lo,
        "hi": hi,
        "rel": rel,
        "dist": dist,
        "revs": float((az[-1] - az[0]) / (2 * np.pi)),
        "reversals": reversals,
        "rad_cv": float(dist[steady].std() / max(dist[steady].mean(), 1e-9)),
        "planarity": planarity,
        "ndir": int(np.median(ndirs)),
        "ms": float(np.median(np.array(walls)[1:])),
        "label": f"{obj} [{lo:.1f},{hi:.1f}]",
    }


def main():
    scenarios = {k: build_scenario(gramian_metric=v) for k, v in METRICS.items()}
    grams = {
        k: jax.jit(
            lambda x, u, sc=sc: sc.cost(
                x, jnp.asarray(u), sc.stlog_dt, return_gramians=True
            ).gramians
        )
        for k, sc in scenarios.items()
    }

    results = []
    for obj, lo, hi in CONFIGS:
        r = run(scenarios, grams, obj, lo, hi)
        results.append(r)
        print(
            f"{r['label']:18s} | revs {r['revs']:+5.1f} | reversals {r['reversals']:4d}"
            f" | rad CV {r['rad_cv']:.3f} | planarity {r['planarity']:.3f}"
            f" | obs {r['ndir']}/6 | {r['ms']:.0f} ms | range [{r['dist'].min():.2f},{r['dist'].max():.2f}]"
        )

    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5.2 * cols, 4.6 * rows))
    tt = np.arange(STEPS)
    for i, r in enumerate(results):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        rel = r["rel"]
        ax.scatter(rel[:, 0], rel[:, 1], rel[:, 2], c=tt, cmap="viridis", s=6)
        ax.plot(rel[:, 0], rel[:, 1], rel[:, 2], color="0.6", lw=0.5)
        ax.scatter([0], [0], [0], marker="*", c="black", s=140)
        ax.scatter(*rel[0], marker="o", edgecolor="k", facecolor="none", s=60)
        ax.set_box_aspect([1, 1, 1])
        c = rel.mean(0)
        rad = float(np.abs(rel - c).max()) or 1.0
        ax.set_xlim(c[0] - rad, c[0] + rad)
        ax.set_ylim(c[1] - rad, c[1] + rad)
        ax.set_zlim(c[2] - rad, c[2] + rad)
        ax.set(
            title=(
                f"{r['label']}\n{r['revs']:+.1f} rev, {r['reversals']} rev'ls, "
                f"radCV {r['rad_cv']:.2f}, flat {r['planarity']:.2f}, obs {r['ndir']}/6"
            ),
            xlabel="x",
            ylabel="y",
            zlabel="z",
        )
    fig.suptitle(
        "Quadrotor orbit vs (objective, distance band) -- same start, 15 s, SLSQP@6",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS / "quad_constraint_sweep.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

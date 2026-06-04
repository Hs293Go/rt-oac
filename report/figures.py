r"""Render the report's figures from ``report/data/*.npz`` (+ the ``*.json`` summaries).

* ``quad_orbit.pdf``       -- the quadrotor open-loop observability orbit, world frame +
  leader-relative frame (the real-time-solve showcase);
* ``planar_spatial.pdf``   -- planar XY trajectories (leader, follower true/est, no-OAC) with
  $3\sigma$ ellipses, across log-det / balanced x hard / soft;
* ``planar_timeseries.pdf``-- follower-position error, observability margin, and solve time
  across the planar configurations;
* ``quad_trichotomy.pdf``  -- carried-estimation NEES + error for the three estimator setups
  (re-anchor / carry-plan-on-truth / closed loop): the future-work diagnostic.

Run after the sweep + trichotomy, from the repo root:

    JAX_PLATFORMS=cpu uv run python report/figures.py
"""

import json
import pathlib

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures"

mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load(tag):
    """Return ``(arrays, summary)`` for a tag, or ``(None, None)`` if it was not produced."""
    npz, js = DATA / f"{tag}.npz", DATA / f"{tag}.json"
    if not npz.exists():
        print(f"  (skip {tag}: no data)")
        return None, None
    arr = dict(np.load(npz))
    summ = json.loads(js.read_text()) if js.exists() else {}
    return arr, summ


def equal_3d(ax, pts):
    """Equal per-axis scale with limits set to the data extents (round orbits stay round)."""
    mins, maxs = pts.min(0), pts.max(0)
    ax.set_box_aspect(np.maximum(maxs - mins, 1e-9))
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.tick_params(pad=-2, labelsize=6)


def cov_ellipse(center, P2, nsig=3.0, n=40):
    """``(n, 2)`` points on the ``nsig``-sigma ellipse of the 2x2 covariance ``P2``."""
    vals, vecs = np.linalg.eigh(0.5 * (P2 + P2.T))
    vals = np.clip(vals, 0.0, None)
    th = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(th), np.sin(th)])
    return (vecs @ (nsig * np.sqrt(vals)[:, None] * circ)).T + np.asarray(center)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------- quadrotor open-loop orbit
def quad_orbit():
    """The open-loop observability orbit: world frame (moving leader) + leader-relative frame."""
    arr, _ = load("quad_open")
    if arr is None:
        return
    lead, foll, rel, tt = arr["lead"], arr["foll"], arr["rel"], arr["t"]
    fig = plt.figure(figsize=(7.0, 3.6))
    for j, (pts, leader, title) in enumerate([
        (foll, lead, "World frame (moving leader)"),
        (rel, None, "Leader-relative frame"),
    ]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        if leader is not None:
            ax.plot(
                leader[:, 0],
                leader[:, 1],
                leader[:, 2],
                color="black",
                lw=1.6,
                label="leader",
            )
            ref = np.vstack([pts, leader])
        else:
            ax.scatter([0], [0], [0], marker="*", c="black", s=140, label="leader")
            ref = np.vstack([pts, np.zeros(3)])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="0.7", lw=0.4)
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=tt, cmap="viridis", s=5)
        ax.scatter(*pts[0], c="tab:green", s=28, marker="o")
        ax.scatter(*pts[-1], c="tab:red", s=32, marker="X")
        equal_3d(ax, ref)
        ax.set(title=title, xlabel="$x$ [m]", ylabel="$y$ [m]", zlabel="$z$ [m]")
        ax.view_init(elev=24, azim=-60)
        ax.legend(loc="upper left", fontsize=7)
    cb = fig.colorbar(
        sc, ax=fig.axes, orientation="horizontal", fraction=0.035, pad=0.04, shrink=0.4
    )
    cb.set_label("$t$ [s]")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.12, wspace=0.04)
    save(fig, "quad_orbit.pdf")


# ----------------------------------------------------------------------------- planar spatial
def planar_spatial():
    """XY trajectories with 3sigma ellipses across the four planar configurations."""
    panels = [
        ("planar_logdet", "log-det, hard"),
        ("planar_logdet_soft", "log-det, soft"),
        ("planar_hybrid", "balanced, hard"),
        ("planar_hybrid_soft", "balanced, soft"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    for ax, (tag, title) in zip(axes.ravel(), panels, strict=True):
        arr, _ = load(tag)
        if arr is None:
            ax.set_title(f"{title} (no data)")
            continue
        lead, ftrue, fest = arr["oac_lead"], arr["oac_ftrue"], arr["oac_fest"]
        noac = arr["noac_ftrue"]
        ax.plot(lead[:, 0], lead[:, 1], color="black", lw=1.6, label="leader")
        ax.plot(
            ftrue[:, 0],
            ftrue[:, 1],
            color="tab:red",
            lw=1.3,
            label="follower true (OAC)",
        )
        ax.plot(
            fest[:, 0],
            fest[:, 1],
            color="tab:orange",
            lw=1.0,
            ls="--",
            label="follower est. (OAC)",
        )
        ax.plot(
            noac[:, 0], noac[:, 1], color="tab:gray", lw=1.1, ls=":", label="no-OAC"
        )
        n = len(fest)
        for k in range(0, n, max(1, n // 7)):
            ell = cov_ellipse(fest[k], arr["oac_Pf"][k])
            ax.plot(ell[:, 0], ell[:, 1], color="tab:orange", lw=0.5, alpha=0.5)
        ax.scatter(*ftrue[0], c="tab:green", s=22, marker="s", zorder=5)
        ax.scatter(*ftrue[-1], c="tab:red", s=26, marker="X", zorder=5)
        ax.set_title(title)
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_aspect("equal")
    axes[0, 0].legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save(fig, "planar_spatial.pdf")


# -------------------------------------------------------------------------- planar timeseries
def planar_timeseries():
    """Follower-position error, observability margin, and solve time across configs."""
    configs = [
        ("planar_logdet", "log-det, hard", "tab:blue"),
        ("planar_logdet_soft", "log-det, soft", "tab:cyan"),
        ("planar_hybrid", "balanced, hard", "tab:green"),
        ("planar_hybrid_soft", "balanced, soft", "tab:olive"),
    ]
    loaded = [((tag, lab, col), *load(tag)) for tag, lab, col in configs]
    loaded = [(m, a) for m, a, _ in loaded if a is not None]
    if not loaded:
        print("  (skip planar_timeseries: no data)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    (_, ref) = loaded[0]
    axes[0].plot(
        ref["t"], ref["noac_err"], color="tab:gray", lw=1.4, ls=":", label="no-OAC"
    )
    for (_, lab, col), arr in loaded:
        axes[0].plot(arr["t"], arr["oac_err"], color=col, lw=1.2, label=lab)
        axes[1].plot(arr["t"], arr["oac_mineig"], color=col, lw=1.2, label=lab)
        axes[2].plot(arr["t"][1:], arr["oac_walls"][1:], color=col, lw=1.0, label=lab)
    axes[0].set(title="Follower-position error", xlabel="$t$ [s]", ylabel="[m]")
    axes[0].legend(loc="upper right")
    axes[1].set(
        title="Observability margin $\\lambda_{\\min}$",
        xlabel="$t$ [s]",
        ylabel="$\\lambda_{\\min}$",
    )
    axes[2].set(title="Per-tick solve time", xlabel="$t$ [s]", ylabel="[ms]")
    fig.tight_layout()
    save(fig, "planar_timeseries.pdf")


# --------------------------------------------------------------- carried-estimation trichotomy
def quad_trichotomy():
    """NEES + follower-position error for the three estimator setups (seed 0)."""
    setups = [
        ("quad_estimation_reanchor", "re-anchor (paper)", "tab:green"),
        ("quad_estimation_planontruth", "carry, plan on truth", "tab:orange"),
        ("quad_estimation", "carry + closed loop (#8)", "tab:red"),
    ]
    loaded = [((tag, lab, col), *load(tag)) for tag, lab, col in setups]
    loaded = [(m, a) for m, a, _ in loaded if a is not None]
    if not loaded:
        print("  (skip quad_trichotomy: no data)")
        return
    chi = stats.chi2.ppf([0.025, 0.975], 9)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for (_, lab, col), arr in loaded:
        axes[0].plot(
            arr["t"], np.maximum(arr["nees"], 1e-3), color=col, lw=1.3, label=lab
        )
        axes[1].plot(
            arr["t"], np.maximum(arr["err"], 1e-3), color=col, lw=1.3, label=lab
        )
    axes[0].axhspan(chi[0], chi[1], color="0.85", alpha=0.7, label="$\\chi^2_9$ 95%")
    axes[0].axhline(9, color="k", ls=":", lw=0.8)
    axes[0].set(title="Estimator NEES", xlabel="$t$ [s]", ylabel="NEES", yscale="log")
    axes[0].legend(loc="upper left")
    axes[1].set(
        title="Follower-position error", xlabel="$t$ [s]", ylabel="[m]", yscale="log"
    )
    fig.tight_layout()
    save(fig, "quad_trichotomy.pdf")


def main():
    quad_orbit()
    planar_spatial()
    planar_timeseries()
    quad_trichotomy()


if __name__ == "__main__":
    main()

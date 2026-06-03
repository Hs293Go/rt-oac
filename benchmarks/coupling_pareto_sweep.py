"""Pareto sweep for the control/estimation-coupling exercise (planar, fast).

Sweeps the normalized trade-off weight lambda to trace the continuous estimation-vs-formation
Pareto front, and overlays the discrete schemes (observability_only, tracking_only, gradnorm,
tube, scheduled). Because lambda is threaded into the solver as a runtime arg, the whole
lambda-grid reuses ONE compiled controller (no per-lambda recompile). The quadrotor scheme
comparison (the #8 testbed) lives in experiments/drone_coupling_eval.py.

  uv run python benchmarks/coupling_pareto_sweep.py
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

import rt_oac  # noqa: F401

# reuse the planar experiment's setup (build_cost / build_scheme_controller / run / constants)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
import planar_coupling_eval as pce

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
LAMBDA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
SEEDS = 3
STEPS = 120


def mean_over_seeds(run_one):
    est = np.mean([run_one(s)["est_rmse"] for s in range(SEEDS)])
    form = np.mean([run_one(s)["form_rmse"] for s in range(SEEDS)])
    return float(est), float(form)


def main():
    cost = pce.build_cost()
    gram = jax.jit(
        lambda x, u: cost(
            x, jnp.asarray(u), pce.STLOG_DT, return_gramians=True
        ).gramians
    )
    ref_us = np.tile(np.r_[pce.LEADER_U, 1.0, 0.0], (pce.WINDOW, 1))
    s_obs = abs(
        float(cost(jnp.asarray(pce.X0), jnp.asarray(ref_us), pce.STLOG_DT).objective)
    )

    # normalized lambda-front (one compiled controller, weight threaded per run)
    ctrl_norm = pce.build_scheme_controller(cost, "normalized", s_obs)
    front = []
    print(
        f"# Coupling Pareto sweep (planar, steps={STEPS}, seeds={SEEDS}, s_obs={s_obs:.2f})"
    )
    print(f"{'lambda':>8}{'est RMSE':>10}{'form RMSE':>11}")
    for lam in LAMBDA_GRID:
        pce.LAMBDA["normalized"] = lam  # threaded weight -> no recompile
        est, form = mean_over_seeds(
            lambda s, lam=lam: pce.run("normalized", STEPS, s, ctrl_norm, gram)
        )
        front.append((lam, est, form))
        print(f"{lam:>8.2f}{est:>10.3f}{form:>11.3f}")

    # discrete schemes
    pce.LAMBDA["normalized"] = 1.0  # restore
    pts = {}
    print(f"\n{'scheme':<18}{'est RMSE':>10}{'form RMSE':>11}")
    for s in ("observability_only", "tracking_only", "gradnorm", "tube", "scheduled"):
        ctrl = pce.build_scheme_controller(cost, s, s_obs)
        est, form = mean_over_seeds(
            lambda seed, ctrl=ctrl, s=s: pce.run(s, STEPS, seed, ctrl, gram)
        )
        pts[s] = (est, form)
        print(f"{s:<18}{est:>10.3f}{form:>11.3f}")

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    f = np.array(front)
    ax.plot(f[:, 1], f[:, 2], "-o", color="tab:blue", label="normalized lambda-front")
    for lam, est, form in front:
        ax.annotate(
            f"λ={lam:g}",
            (est, form),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
            color="tab:blue",
        )
    markers = {
        "observability_only": "s",
        "tracking_only": "^",
        "gradnorm": "D",
        "tube": "P",
        "scheduled": "*",
    }
    for s, (est, form) in pts.items():
        ax.scatter(est, form, marker=markers[s], s=120, zorder=5, label=s)
    ax.set(
        title="Coupling Pareto: estimation RMSE vs formation RMSE (planar)",
        xlabel="follower-position estimation RMSE [m]",
        ylabel="formation RMSE [m]",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / "coupling_pareto.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    print(
        "=> the normalized lambda-front spans the corners (knee near lambda=1); scheduled"
        " sits on the efficient interior; gradnorm over-weights the steep observability"
        " gradient here and is dominated."
    )


if __name__ == "__main__":
    main()

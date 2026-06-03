"""Planar control/estimation-coupling eval: standoff-tracking balanced with observability.

Fast-dev testbed for the coupling exercise (PROGRESS.md finding #8). The planar
leader-follower is the benign case: the follower's position is observable only through the
inter-robot range, so it MUST maneuver to localise, and the estimate-feedback loop is stable
(observability is the *stabiliser* here, unlike the quadrotor). So this experiment validates
the balanced-cost machinery and the Pareto ordering, and shows the covariance-scheduled
"look when lost" policy getting the best of both: maneuver to localise while uncertain, then
settle into formation.

Two competing objectives on the same closed loop (controller acts on the EKF *estimate*):
  * observability  -- maneuver to keep the follower position observable (range-only);
  * standoff       -- keep formation, tracking p_ref = predicted leader position + offset.

Schemes (``--scheme``): observability_only (pure obs), tracking_only (pure formation),
normalized (NormalizedWeightedSum), gradnorm (auto-scaled), tube (soft standoff tube + obs),
scheduled (normalized with the weight scheduled UP on uncertainty -- active perception).

Metrics: follower-position estimation RMSE/final (does it localise?), formation RMSE
(does it stay in formation?), terminal covariance, observability margin, solve time.

  uv run python experiments/planar_coupling_eval.py
  uv run python experiments/planar_coupling_eval.py --scheme scheduled --plot
"""

import argparse
import pathlib

from example_lib.misc import simple_ekf
from example_lib.models import leader_follower_robots as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator, observability_cost

import rt_oac  # noqa: F401
from rt_oac import balance_constraints, dual_control, metrics, tracking_cost
from rt_oac.balanced_cost import make_balanced
from rt_oac.controller import RTController

DT = 0.1
STLOG_DT = 0.2
WINDOW = 5
ORDER = 1
VAR = np.r_[np.full(2, 1e-1), np.full(2, 1e-2), 1e-2]  # [leader pos, headings, range]
INPUT_VAR = np.tile([1e-2, 1e-2], 2)
X0 = np.array([0.5, 0.0, 0.0, 0.0, 5.0, 0.0])  # leader (0.5,0), follower (0,5)
LEADER_U = np.array([1.0, 0.0])
FORMATION_OFFSET = X0[3:5] - X0[0:2]  # nominal follower-minus-leader offset = [-0.5, 5]
POS_IDX = (3, 4)  # follower position in the 6-state vector
POS_SLICE = slice(3, 5)
RHO = 1.0  # standoff tube radius / tracking scale
INIT_FOLL_VAR = 4.0  # large initial follower-position uncertainty
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

SCHEMES = (
    "observability_only",
    "tracking_only",
    "normalized",
    "gradnorm",
    "tube",
    "scheduled",
)
# scheduled weight ramp: half-weight at trace s0; localises (high obs weight) while uncertain
SCHED = {"s0": 0.5, "w_min": 0.05, "w_max": 4.0, "increasing": True}
LAMBDA = {"normalized": 1.0, "gradnorm": 1.0, "tube": 1.0}


def build_cost():
    return observability_cost.ObservabilityCost(
        mdl.dynamics,
        mdl.observation,
        DT,
        gramian_kw={"order": ORDER, "var": VAR},
        gramian_metric=metrics.neg_logdet,
        observed_indices=(),
    )


def leader_pred(leader_state):
    """(window, 2) predicted leader positions over the horizon (rolls LEADER_U)."""
    px, py, th = leader_state
    ks = np.arange(WINDOW)
    v = LEADER_U[0]
    return np.stack(
        [px + ks * DT * v * np.cos(th), py + ks * DT * v * np.sin(th)], axis=1
    )


def p_ref_window(x_hat):
    return leader_pred(x_hat[0:3]) + FORMATION_OFFSET  # (window, 2)


def build_scheme_controller(cost, scheme, s_obs):
    """Builds the balanced cost + controller for ``scheme`` (compiled once, reused)."""
    # 'scheduled' reuses the normalized combiner; only its per-step weight differs.
    combiner_scheme = "normalized" if scheme == "scheduled" else scheme
    if combiner_scheme == "tube":
        track = balance_constraints.soft_tube_cost(
            RHO, position_indices=POS_IDX, w_tube=20.0
        )
        bal = make_balanced(cost, track, scheme="tube")
    else:
        track = tracking_cost.quadratic_tracking_cost(
            position_indices=POS_IDX, w_pos=1.0
        )
        bal = make_balanced(
            cost, track, scheme=combiner_scheme, s_track=RHO**2 * WINDOW, s_obs=s_obs
        )
    ctrl = RTController(
        bal,
        stlog_dt=STLOG_DT,
        lb=np.array([0.0, -2.0]),
        ub=np.array([4.0, 2.0]),
        n_inputs=4,
        follower_indices=(2, 3),
        method="SLSQP",
        maxiter=6,
        constraint=mdl.interrobot_distance,
        constraint_bounds=(0.2, 6.0),
    )
    return ctrl


def scheme_weight(scheme, P):
    if scheme == "scheduled":
        return dual_control.covariance_weight(
            P,
            pos_slice=POS_SLICE,
            s0=SCHED["s0"],
            w_min=SCHED["w_min"],
            w_max=SCHED["w_max"],
            increasing=SCHED["increasing"],
        )
    return LAMBDA.get(scheme, 1.0)


def run(scheme, steps, seed, ctrl, gram):
    rng = np.random.default_rng(seed)
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.RK4, stepsize=DT)
    )
    ekf = simple_ekf.SimpleEKF(
        lambda x, u, dt: x + dt * mdl.dynamics(x, u),
        lambda x: mdl.observation(x, 0),
        in_cov=np.diag(INPUT_VAR),
        obs_cov=np.diag(VAR),
    )
    x_true = X0.copy()
    P = np.diag([1e-3, 1e-3, 1e-3, INIT_FOLL_VAR, INIT_FOLL_VAR, 1e-2])
    x_hat = x_true + rng.multivariate_normal(np.zeros(6), P)

    rec = {k: [] for k in ("est", "form", "Ptr", "mineig", "w", "ms")}
    prev_u = None
    for _ in range(steps):
        p_ref = p_ref_window(x_hat)
        w = scheme_weight(scheme, P)
        guess = np.tile(np.r_[LEADER_U, 1.0, 0.0], (WINDOW, 1))
        if prev_u is not None:
            guess[:, 2:4] = np.vstack([prev_u[1:, 2:4], prev_u[-1:, 2:4]])
        res = ctrl.solve(x_hat, guess, p_ref=p_ref, weight=w)
        prev_u = res.u
        u_foll = res.u[0, 2:4]
        rec["ms"].append(res.wall_time * 1e3)
        rec["w"].append(float(w))
        g = np.asarray(gram(jnp.asarray(x_hat), res.u))
        rec["mineig"].append(
            float(np.linalg.eigvalsh(0.5 * (g + g.swapaxes(-1, -2)).sum(0))[0])
        )

        u = np.r_[LEADER_U, u_foll]
        u_applied = u + np.r_[0, 0, rng.normal(0, np.sqrt(INPUT_VAR[2:]))]
        x_hat, P = ekf.predict(
            jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u_applied), DT
        )
        x_true = np.array(sim(jnp.asarray(x_true), jnp.asarray(u_applied))[0])
        y = np.array(mdl.observation(jnp.asarray(x_true), 0)) + rng.normal(
            0, np.sqrt(VAR)
        )
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)

        rec["est"].append(float(np.linalg.norm(x_hat[3:5] - x_true[3:5])))
        rec["form"].append(
            float(np.linalg.norm(x_true[3:5] - (x_true[0:2] + FORMATION_OFFSET)))
        )
        rec["Ptr"].append(float(np.trace(P[3:5, 3:5])))

    out = {k: np.array(v) for k, v in rec.items()}
    return {
        "scheme": scheme,
        "est_rmse": float(np.sqrt((out["est"] ** 2).mean())),
        "est_final": float(out["est"][-1]),
        "form_rmse": float(np.sqrt((out["form"] ** 2).mean())),
        "term_P": float(out["Ptr"][-1]),
        "mineig": float(np.mean(out["mineig"])),
        "w_med": float(np.median(out["w"])),
        "ms": float(np.median(out["ms"][1:]))
        if len(out["ms"]) > 1
        else float(out["ms"][0]),
        "series": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--scheme", choices=SCHEMES, default=None, help="single scheme + --plot"
    )
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    cost = build_cost()
    gram = jax.jit(
        lambda x, u: cost(x, jnp.asarray(u), STLOG_DT, return_gramians=True).gramians
    )
    ref_us = np.tile(np.r_[LEADER_U, 1.0, 0.0], (WINDOW, 1))
    s_obs = abs(float(cost(jnp.asarray(X0), jnp.asarray(ref_us), STLOG_DT).objective))

    schemes = [args.scheme] if args.scheme else list(SCHEMES)
    ctrls = {s: build_scheme_controller(cost, s, s_obs) for s in schemes}

    print(
        f"# Planar coupling eval (steps={args.steps}, seeds={args.seeds}, s_obs={s_obs:.2f})"
    )
    print(
        f"# standoff offset {FORMATION_OFFSET}, follower position observable only via range\n"
    )
    print(
        f"{'scheme':<18}{'est RMSE':>10}{'est final':>10}{'form RMSE':>11}"
        f"{'term P':>11}{'min-eig':>10}{'w_med':>8}{'ms':>7}"
    )
    agg = {
        s: {
            k: []
            for k in (
                "est_rmse",
                "est_final",
                "form_rmse",
                "term_P",
                "mineig",
                "w_med",
                "ms",
            )
        }
        for s in schemes
    }
    last = {}
    for s in schemes:
        for seed in range(args.seeds):
            r = run(s, args.steps, seed, ctrls[s], gram)
            for k in agg[s]:
                agg[s][k].append(r[k])
            last[s] = r  # keep one for plotting
    for s in schemes:
        a = {k: np.mean(v) for k, v in agg[s].items()}
        print(
            f"{s:<18}{a['est_rmse']:>10.3f}{a['est_final']:>10.3f}{a['form_rmse']:>11.3f}"
            f"{a['term_P']:>11.2e}{a['mineig']:>10.3f}{a['w_med']:>8.2f}{a['ms']:>7.1f}"
        )
    print("\n=> observability_only localises but breaks formation; tracking_only keeps")
    print("   formation but cannot localise; scheduled localises early then re-forms.")

    if args.plot:
        _plot(last, args.steps)


def _plot(last, steps):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    tt = np.arange(steps) * DT
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.5))
    for s, r in last.items():
        a1.plot(tt, r["series"]["est"], lw=1.4, label=s)
        a2.plot(tt, r["series"]["form"], lw=1.4, label=s)
    a1.set(title="Follower-position estimation error", xlabel="t [s]", ylabel="m")
    a2.set(title="Formation (standoff) error", xlabel="t [s]", ylabel="m")
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    # Pareto: localisation vs formation
    for s, r in last.items():
        a3.scatter(r["est_rmse"], r["form_rmse"], s=60)
        a3.annotate(
            s,
            (r["est_rmse"], r["form_rmse"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    a3.set(
        title="Pareto: estimation RMSE vs formation RMSE",
        xlabel="est RMSE [m]",
        ylabel="formation RMSE [m]",
    )
    a3.grid(alpha=0.3)
    fig.suptitle(
        "Planar coupling: standoff-tracking balanced with observability", fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RESULTS / "planar_coupling.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

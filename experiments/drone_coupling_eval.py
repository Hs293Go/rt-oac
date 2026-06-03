"""Quadrotor control/estimation-coupling resolution (PROGRESS.md finding #8).

The decisive testbed. Under estimate feedback the aggressive pure-observability orbit
(soft-min + tight distance band) destabilises the ErrorStateEKF it feeds -- the
follower-position error diverges to 100s-1000s of m (finding #8). Here we add a Euclidean
*standoff* anchor on the relative position (r_lf -> r_standoff) and balance it against
observability, optionally with the trade-off weight scheduled DOWN on the EKF covariance
(dual control: stay near the anchor while the estimator is uncertain, do not out-run it).

Forks ``drone_ekf_eval_esekf.py``: identical ErrorStateEKF, manifold, noise, seed, and
process-noise floor; the controller still acts on the estimate. Schemes (``--scheme``):
observability_only (the diverging baseline), normalized, gradnorm, tube, scheduled.

  uv run python experiments/drone_coupling_eval.py                 # all schemes
  uv run python experiments/drone_coupling_eval.py --scheme scheduled --plot
"""

import argparse
import pathlib

from example_lib import math as elmath
from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator

import rt_oac  # noqa: F401
from rt_oac import balance_constraints, dual_control, metrics, tracking_cost
from rt_oac.balanced_cost import make_balanced
from rt_oac.controller import RTController
from rt_oac.error_state_ekf import ErrorStateEKF
from rt_oac.scenario import build_scenario
from rt_oac.warmstart import warm_guess

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
# Aggressive config that triggers #8 under estimate feedback: soft-min + tight band.
MIN_DIST, MAX_DIST = 0.5, 1.2
# Standoff anchor = a STATIONARY relative-state set-point: relative position r_lf -> standoff
# AND relative velocity v_lf -> 0. Anchoring position alone cannot arrest the initial relative
# velocity, so the pair drifts apart; damping v_lf parks the follower at the standoff.
POS_IDX_TRACK = (
    0,
    1,
    2,
    7,
    8,
    9,
)  # relative position + relative velocity (Euclidean blocks)
STANDOFF_TRACK = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
POS_IDX_TUBE = (0, 1, 2)  # the tube is on position only
STANDOFF_TUBE = np.array([1.0, 0.0, 0.0])
POS_SLICE = slice(
    0, 3
)  # position block in the 9x9 tangent covariance (for the schedule)
RHO = 0.5  # standoff scale / tube radius
SCHEMES = ("observability_only", "normalized", "gradnorm", "tube", "scheduled")
# scheduled: down-weight observability when uncertain (caution); s0 ~ stable pos-trace.
SCHED = {"s0": 0.3, "w_min": 0.05, "w_max": 2.0, "increasing": False}
LAMBDA = {"normalized": 1.0, "gradnorm": 1.0, "tube": 1.0}


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


X0 = np.concatenate([
    [1.0, 0.0, 0.0],
    quat_z(20),
    [0.0, 1.0, 0.0],
])  # "close" non-MUC start


def renorm(x):
    x = np.array(x)
    x[3:7] /= np.linalg.norm(x[3:7])
    return x


def tangent_error(x_hat, x_true):
    diff = x_hat - x_true
    dtheta = np.array(
        elmath.quaternion_log(
            elmath.quaternion_product(
                elmath.quaternion_inverse(jnp.asarray(x_true[3:7])),
                jnp.asarray(x_hat[3:7]),
            )
        )
    )
    return np.concatenate([diff[0:3], dtheta, diff[7:10]])


def build_scheme_controller(sc, scheme, s_obs):
    combiner_scheme = "normalized" if scheme == "scheduled" else scheme
    if combiner_scheme == "tube":
        track = balance_constraints.soft_tube_cost(
            RHO, position_indices=POS_IDX_TUBE, w_tube=20.0
        )
        bal = make_balanced(sc.cost, track, scheme="tube")
    else:
        track = tracking_cost.quadratic_tracking_cost(
            position_indices=POS_IDX_TRACK, w_pos=1.0
        )
        bal = make_balanced(
            sc.cost,
            track,
            scheme=combiner_scheme,
            s_track=RHO**2 * sc.window,
            s_obs=s_obs,
        )
    ctrl = RTController(
        bal,
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


def run(scheme, sc, ctrl, steps, *, offset_scale=1.0, seed=0):
    dt = sc.cfg["sim"]["integrator_dt"]
    range_var = sc.cfg["noise"]["range_var"]
    att_var = sc.cfg["noise"]["att_var"]
    obs_var = np.concatenate([[range_var], np.full(4, att_var)])
    res_var = np.concatenate([[range_var], np.full(3, att_var)])
    input_var = np.tile([0.05, 0.01, 0.01, 0.01], 2)
    proc_var = np.array([0.02, 0.02, 0.02, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05])

    sim = jax.jit(
        integrator.Integrator(
            mdl.dynamics, integrator.Methods.EULER, stepsize=dt, manifold=mdl.MANIFOLD
        )
    )
    ekf = ErrorStateEKF(
        mdl.dynamics,
        lambda x: mdl.observation(x),
        mdl.MANIFOLD,
        in_cov=np.diag(input_var),
        obs_cov=np.diag(res_var),
        proc_cov=np.diag(proc_var),
        method=integrator.Methods.EULER,
    )
    gram = jax.jit(
        lambda x, u: sc.cost(
            x, jnp.asarray(u), sc.stlog_dt, return_gramians=True
        ).gramians
    )

    rng = np.random.default_rng(seed)
    x_true = X0.copy()
    P = np.diag([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 0.4, 0.4, 0.4])
    delta0 = offset_scale * np.array([0.9, 0.9, 0.4, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0])
    x_hat = np.array(mdl.MANIFOLD.boxplus(jnp.asarray(x_true), jnp.asarray(delta0)))
    standoff = STANDOFF_TUBE if scheme == "tube" else STANDOFF_TRACK
    p_ref = np.tile(standoff, (sc.window, 1))

    rec = {k: [] for k in ("rel", "err", "sig", "dist", "nees", "qnorm", "mineig", "w")}
    prev_u = None
    for i in range(steps):
        w = scheme_weight(scheme, P)
        guess = warm_guess(sc.reference_guess(i), prev_u)
        res = ctrl.solve(x_hat, guess, p_ref=p_ref, weight=w)
        prev_u = res.u
        u = res.u[0].copy()
        u[4:] += rng.normal(0, np.sqrt(input_var[4:]))

        g = np.asarray(gram(jnp.asarray(x_hat), res.u))
        rec["mineig"].append(
            float(np.linalg.eigvalsh(0.5 * (g + g.swapaxes(-1, -2)).sum(0))[0])
        )
        rec["w"].append(float(w))

        x_hat, P = ekf.predict(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u), dt)
        x_hat, P = np.array(x_hat), np.array(P)
        x_true = renorm(sim(jnp.asarray(x_true), jnp.asarray(u))[0])
        y = np.array(mdl.observation(jnp.asarray(x_true))) + rng.normal(
            0, np.sqrt(obs_var)
        )
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)

        err_t = tangent_error(x_hat, x_true)
        nees = float(err_t @ np.linalg.solve(P + 1e-12 * np.eye(9), err_t))
        rec["rel"].append(x_true[0:3].copy())
        rec["err"].append(float(np.linalg.norm(x_hat[0:3] - x_true[0:3])))
        rec["sig"].append(float(3 * np.sqrt(max(np.trace(P[0:3, 0:3]) / 3, 0))))
        rec["dist"].append(float(np.linalg.norm(x_true[0:3])))
        rec["nees"].append(nees)
        rec["qnorm"].append(float(np.linalg.norm(x_hat[3:7])))

    out = {k: np.array(v) for k, v in rec.items()}
    warmup = min(20, steps // 4)
    return {
        "scheme": scheme,
        "err_max": float(out["err"].max()),
        "err_final": float(out["err"][-1]),
        "err_rmse": float(np.sqrt((out["err"] ** 2).mean())),
        "term_P": float(np.trace(P[0:3, 0:3])),
        "ss_nees": float(np.median(out["nees"][warmup:])),
        "in3s": float(np.mean(out["err"] <= out["sig"])),
        "dist_max": float(out["dist"].max()),
        "mineig": float(np.median(out["mineig"])),
        "w_med": float(np.median(out["w"])),
        "qdev": float(np.max(np.abs(out["qnorm"] - 1.0))),
        "diverged": bool(out["err"].max() > 5.0 or out["dist"].max() > 5.0),
        "series": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--scheme", choices=SCHEMES, default=None)
    ap.add_argument("--offset-scale", type=float, default=1.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    sc = build_scenario(gramian_metric=metrics.neg_softmin_eig)
    ref_us = sc.reference_guess(0)
    s_obs = abs(
        float(sc.cost(jnp.asarray(X0), jnp.asarray(ref_us), sc.stlog_dt).objective)
    )
    schemes = [args.scheme] if args.scheme else list(SCHEMES)

    dof = 9
    band = f"[{MIN_DIST}, {MAX_DIST}]"
    print(
        f"# Quadrotor coupling eval (soft-min, band {band}, steps={args.steps}, "
        f"offset x{args.offset_scale}, s_obs={s_obs:.2f})"
    )
    print(
        f"# standoff pos={STANDOFF_TUBE} + v_lf->0; chi2 95% single-sample E[NEES]={dof}\n"
    )
    print(
        f"{'scheme':<18}{'err max':>9}{'err final':>10}{'term P':>10}{'ss NEES':>9}"
        f"{'<=3sig':>8}{'dist max':>9}{'w_med':>7}{'diverged':>9}"
    )
    last = {}
    for s in schemes:
        ctrl = build_scheme_controller(sc, s, s_obs)
        r = run(s, sc, ctrl, args.steps, offset_scale=args.offset_scale)
        last[s] = r
        print(
            f"{s:<18}{r['err_max']:>9.2f}{r['err_final']:>10.2f}{r['term_P']:>10.2e}"
            f"{r['ss_nees']:>9.1f}{r['in3s']:>8.2f}{r['dist_max']:>9.2f}{r['w_med']:>7.2f}"
            f"{('YES' if r['diverged'] else 'no'):>9}"
        )
    print("\n=> observability_only reproduces #8 (diverges); the standoff anchor")
    print(
        "   (normalized/gradnorm/tube/scheduled) keeps the estimate-feedback loop bounded."
    )

    if args.plot:
        _plot(last, args.steps, sc)


def _plot(last, steps, sc):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    dt = sc.cfg["sim"]["integrator_dt"]
    tt = np.arange(steps) * dt
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.5))
    for s, r in last.items():
        a1.semilogy(tt, np.maximum(r["series"]["err"], 1e-3), lw=1.4, label=s)
        a2.plot(tt, r["series"]["dist"], lw=1.2, label=s)
    a1.set(title="Follower rel-pos estimation error (log)", xlabel="t [s]", ylabel="m")
    a1.axhline(5.0, color="k", ls=":", lw=1, label="divergence (5 m)")
    a2.axhline(MIN_DIST, color="0.5", ls="--", lw=1)
    a2.axhline(MAX_DIST, color="0.5", ls="--", lw=1, label="band")
    a2.set(title="Inter-drone distance", xlabel="t [s]", ylabel="m")
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for s, r in last.items():
        a3.scatter(r["mineig"], min(r["err_rmse"], 1e3), s=60)
        a3.annotate(
            s,
            (r["mineig"], min(r["err_rmse"], 1e3)),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    a3.set(
        title="Pareto: observability vs est-RMSE (log)",
        xlabel="median min-eig",
        ylabel="est RMSE [m]",
    )
    a3.set_yscale("log")
    a3.grid(alpha=0.3)
    fig.suptitle(
        "Quadrotor coupling: standoff anchor resolves the #8 divergence", fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RESULTS / "drone_coupling.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

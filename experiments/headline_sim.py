"""Headline simulation: real-time observability-aware control, planar leader-follower.

The "best frontier method so far": per-node log-det objective + lightweight SLSQP solver
with early termination (maxiter=6, ~real-time), driving a follower whose position is
observable only through the inter-vehicle range. A real EKF carries its own estimate (the
controller sees only the estimate; ground truth evolves separately), so the animation shows
the estimate's 3-sigma covariance ellipse *shrink* as the follower orbits and information
accumulates -- the whole story in one view.

Visualization is rerun, mirroring the companion repo's experiments. Live viewer:
    uv run python experiments/headline_sim.py --spawn
Headless recording (default; open the .rrd later with `rerun`):
    uv run python experiments/headline_sim.py            # -> results/headline_sim.rrd
"""

import argparse
import functools
import pathlib

from example_lib.misc import simple_ekf
from example_lib.models import leader_follower_robots as mdl
from example_lib.visualization import visualization as viz
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator, observability_cost
import rerun as rr
import rerun.blueprint as rrb

import rt_oac  # noqa: F401  (bootstraps companion src + x64)
from rt_oac import metrics
from rt_oac.controller import RTController

DT = 0.1
STLOG_DT = 0.2
WINDOW = 5
ORDER = 1
VAR = np.r_[np.full(2, 1e-1), np.full(2, 1e-2), 1e-2]  # [leader pos, headings, range]
INPUT_VAR = np.tile([1e-2, 1e-2], 2)
X0 = np.array([0.5, 0.0, 0.0, 0.0, 5.0, 0.0])  # leader (0.5,0), follower (0,5)
LEADER_U = np.array([1.0, 0.0])  # straight at speed 1

PureYaw = functools.partial(rr.RotationAxisAngle, axis=[0, 0, 1])


def build_controller(maxiter):
    cost = observability_cost.ObservabilityCost(
        mdl.dynamics,
        mdl.observation,
        DT,
        gramian_kw={"order": ORDER, "var": VAR},
        gramian_metric=metrics.neg_logdet,
        observed_indices=(),
    )
    ctrl = RTController(
        cost,
        stlog_dt=STLOG_DT,
        lb=np.array([0.0, -2.0]),
        ub=np.array([4.0, 2.0]),
        n_inputs=4,
        follower_indices=(2, 3),
        method="SLSQP",
        maxiter=maxiter,
        constraint=mdl.interrobot_distance,
        constraint_bounds=(0.2, 6.0),
    )
    return ctrl, cost


def cov_ellipse(center2, P2, nsig=3.0, n=48):
    """3-sigma covariance ellipse as (n,3) points (z=0)."""
    vals, vecs = np.linalg.eigh(0.5 * (P2 + P2.T))
    vals = np.clip(vals, 0.0, None)
    t = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])  # (2,n)
    pts = (vecs @ (nsig * np.sqrt(vals)[:, None] * circ)).T + np.asarray(center2)
    return np.column_stack([pts, np.zeros(n)])


def style_series():
    rr.log(
        "/graphs/error/follower",
        rr.SeriesLines(names="follower pos error [m]", colors=[230, 40, 40], widths=2),
        static=True,
    )
    rr.log(
        "/graphs/error/three_sigma",
        rr.SeriesLines(names="EKF 3-sigma [m]", colors=[150, 150, 150], widths=2),
        static=True,
    )
    rr.log(
        "/graphs/observability/min_eig",
        rr.SeriesLines(names="accumulated min-eig", colors=[40, 120, 230], widths=2),
        static=True,
    )
    rr.log(
        "/graphs/solve/ms",
        rr.SeriesLines(names="solve time [ms]", colors=[40, 180, 80], widths=2),
        static=True,
    )


def send_blueprint():
    rr.send_blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/sim", name="Leader-follower under OAC"),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/graphs/error",
                    name="Follower-position estimation error & 3-sigma",
                ),
                rrb.TimeSeriesView(
                    origin="/graphs/observability",
                    name="Accumulated observability (min-eig)",
                ),
                rrb.TimeSeriesView(origin="/graphs/solve", name="Solve time [ms]"),
                row_shares=[1, 1, 1],
            ),
            column_shares=[1.4, 1.0],
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--maxiter", type=int, default=6, help="SLSQP cap (frontier: 6)")
    ap.add_argument("--spawn", action="store_true", help="launch the live rerun viewer")
    ap.add_argument("--save", type=str, default=None, help="record to this .rrd path")
    ap.add_argument(
        "--gif", action="store_true", help="also render a portable .gif + preview .png"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rr.init("rt_oac_headline", spawn=args.spawn)
    if not args.spawn:
        out = args.save or str(
            pathlib.Path(__file__).resolve().parents[1] / "results" / "headline_sim.rrd"
        )
        rr.save(out)
    style_series()
    send_blueprint()

    rng = np.random.default_rng(args.seed)
    ctrl, cost = build_controller(args.maxiter)
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.RK4, stepsize=DT)
    )
    ekf = simple_ekf.SimpleEKF(
        lambda x, u, dt: x + dt * mdl.dynamics(x, u),
        lambda x: mdl.observation(x, 0),
        in_cov=np.diag(INPUT_VAR),
        obs_cov=np.diag(VAR),
    )
    gram = jax.jit(
        lambda x, u: cost(x, jnp.asarray(u), STLOG_DT, return_gramians=True).gramians
    )

    def acc_min_eig(x, u):
        g = np.asarray(gram(jnp.asarray(x), u))
        acc = 0.5 * (g + np.swapaxes(g, -1, -2)).sum(0)
        return float(np.linalg.eigvalsh(acc)[0])

    leader = viz.PoseReferenceFrame("/sim/leader")
    leader_tr = viz.PositionTrace("/sim/leader", max_length=200)
    foll_true = viz.PoseReferenceFrame("/sim/follower_true")
    foll_true_tr = viz.PositionTrace("/sim/follower_true", max_length=200)
    foll_est = viz.PoseReferenceFrame("/sim/follower_est")

    x_true = X0.copy()
    P = np.diag([1e-3, 1e-3, 1e-3, 2.0, 2.0, 1e-2])  # follower position poorly known
    # deterministic large initial estimate offset so the convergence is visible: the
    # estimate must be driven down through observability, which requires the follower to move
    x_hat = x_true + np.array([
        0.0,
        0.0,
        0.0,
        1.6,
        1.6,
        0.0,
    ])  # ~2.3 m follower-position error

    prev_u = None
    follower_guess = np.array([1.0, 0.0])
    errs, walls = [], []
    rec = {
        "lead": [],
        "ftrue": [],
        "fest": [],
        "Pf": [],
        "err": [],
        "sig": [],
        "obs": [],
    }
    t = 0.0
    for _ in range(args.steps):
        # --- frontier OAC solve on the ESTIMATE ---
        guess = np.tile(np.r_[LEADER_U, follower_guess], (WINDOW, 1))
        if prev_u is not None:
            guess[:, 2:4] = np.vstack([prev_u[1:, 2:4], prev_u[-1:, 2:4]])
        res = ctrl.solve(x_hat, guess)
        prev_u = res.u
        u = np.r_[LEADER_U, res.u[0, 2:4]]
        u_applied = u + np.r_[0, 0, rng.normal(0, np.sqrt(INPUT_VAR[2:]))]

        # --- EKF predict / true step / measure / update ---
        x_hat, P = ekf.predict(
            jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u_applied), DT
        )
        x_next, _ = sim(jnp.asarray(x_true), jnp.asarray(u_applied))
        x_true = np.array(x_next)
        y = np.array(mdl.observation(jnp.asarray(x_true), 0)) + rng.normal(
            0, np.sqrt(VAR)
        )
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)

        foll_err = float(np.linalg.norm(x_hat[3:5] - x_true[3:5]))
        errs.append(foll_err)
        walls.append(res.wall_time * 1e3)
        sig = 3.0 * np.sqrt(max(np.trace(P[3:5, 3:5]) / 2.0, 0.0))
        obs = acc_min_eig(x_hat, res.u)
        t += DT

        if args.gif:
            rec["lead"].append(x_true[0:2].copy())
            rec["ftrue"].append(x_true[3:5].copy())
            rec["fest"].append(x_hat[3:5].copy())
            rec["Pf"].append(P[3:5, 3:5].copy())
            rec["err"].append(foll_err)
            rec["sig"].append(sig)
            rec["obs"].append(obs)

        # --- rerun logging ---
        rr.set_time("/time", duration=t)
        leader.set_pose(np.append(x_true[0:2], 0.0), PureYaw(angle=x_true[2]))
        leader_tr.add_position(np.append(x_true[0:2], 0.0))
        foll_true.set_pose(np.append(x_true[3:5], 0.0), PureYaw(angle=x_true[5]))
        foll_true_tr.add_position(np.append(x_true[3:5], 0.0))
        foll_est.set_pose(np.append(x_hat[3:5], 0.0), PureYaw(angle=x_hat[5]))
        rr.log(
            "/sim/follower_est/cov3sigma",
            rr.LineStrips3D(
                cov_ellipse(x_hat[3:5], P[3:5, 3:5])[None], colors=[230, 160, 40]
            ),
        )
        rr.log(
            "/sim/range",
            rr.LineStrips3D(
                np.array([np.append(x_true[0:2], 0.0), np.append(x_true[3:5], 0.0)])[
                    None
                ],
                colors=[120, 120, 120],
            ),
        )
        rr.log("/graphs/error/follower", rr.Scalars(foll_err))
        rr.log("/graphs/error/three_sigma", rr.Scalars(sig))
        rr.log("/graphs/observability/min_eig", rr.Scalars(obs))
        rr.log("/graphs/solve/ms", rr.Scalars(res.wall_time * 1e3))

    print(
        f"steps={args.steps} maxiter={args.maxiter} | follower-pos error {errs[0]:.2f} -> {errs[-1]:.2f} m"
        f" | median solve {np.median(walls):.1f} ms | terminal cov(foll) {np.trace(P[3:5, 3:5]):.2e}"
    )
    if not args.spawn:
        print(f"recording written; open with:  rerun {out}")
    if args.gif:
        render_gif(rec)


def render_gif(rec):
    """Portable matplotlib animation (gif) + a representative still (png) of the run."""
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt

    lead = np.array(rec["lead"])
    ftrue = np.array(rec["ftrue"])
    fest = np.array(rec["fest"])
    Pf = np.array(rec["Pf"])
    err = np.array(rec["err"])
    sig = np.array(rec["sig"])
    obs = np.array(rec["obs"])
    n = len(lead)
    tt = np.arange(n) * DT
    res_dir = pathlib.Path(__file__).resolve().parents[1] / "results"

    fig, (axw, axe) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    allx = np.r_[lead[:, 0], ftrue[:, 0], fest[:, 0]]
    ally = np.r_[lead[:, 1], ftrue[:, 1], fest[:, 1]]
    axw.set(
        xlim=(allx.min() - 1, allx.max() + 1),
        ylim=(ally.min() - 1, ally.max() + 1),
        xlabel="x [m]",
        ylabel="y [m]",
        title="Leader-follower under OAC (true + EKF estimate)",
    )
    axw.set_aspect("equal")
    axw.grid(alpha=0.3)
    (lead_ln,) = axw.plot([], [], "-", color="black", lw=2, label="leader")
    (ft_ln,) = axw.plot([], [], "-", color="tab:blue", lw=1.5, label="follower (true)")
    (fe_pt,) = axw.plot(
        [], [], "o", color="tab:orange", ms=7, label="follower (EKF est.)"
    )
    (ell_ln,) = axw.plot(
        [], [], "-", color="tab:orange", lw=1, alpha=0.8, label="3σ cov."
    )
    (rng_ln,) = axw.plot([], [], "--", color="0.6", lw=1, label="range meas.")
    lead_dot = axw.scatter([], [], c="black", s=60, zorder=5)
    axw.legend(loc="upper right", fontsize=8)

    axe.set(
        xlim=(0, tt[-1]),
        xlabel="time [s]",
        title="Follower-position error & observability",
    )
    axe.grid(alpha=0.3)
    (err_ln,) = axe.plot([], [], "-", color="tab:red", lw=2, label="EKF error [m]")
    (sig_ln,) = axe.plot([], [], "-", color="0.6", lw=1.5, label="3σ bound [m]")
    ax2 = axe.twinx()
    (obs_ln,) = ax2.plot([], [], "-", color="tab:blue", lw=1.5, label="accum. min-eig")
    axe.set_ylim(0, max(err.max(), sig.max()) * 1.1 + 1e-6)
    ax2.set_ylim(0, obs.max() * 1.1 + 1e-9)
    ax2.set_ylabel("accumulated min-eig", color="tab:blue")
    axe.set_ylabel("error / 3σ [m]")
    axe.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    th = np.linspace(0, 2 * np.pi, 48)
    circ = np.stack([np.cos(th), np.sin(th)])

    def frame(i):
        lead_ln.set_data(lead[: i + 1, 0], lead[: i + 1, 1])
        ft_ln.set_data(ftrue[: i + 1, 0], ftrue[: i + 1, 1])
        fe_pt.set_data([fest[i, 0]], [fest[i, 1]])
        vals, vecs = np.linalg.eigh(0.5 * (Pf[i] + Pf[i].T))
        ell = (vecs @ (3 * np.sqrt(np.clip(vals, 0, None))[:, None] * circ)).T + fest[i]
        ell_ln.set_data(ell[:, 0], ell[:, 1])
        rng_ln.set_data([lead[i, 0], ftrue[i, 0]], [lead[i, 1], ftrue[i, 1]])
        lead_dot.set_offsets([lead[i]])
        err_ln.set_data(tt[: i + 1], err[: i + 1])
        sig_ln.set_data(tt[: i + 1], sig[: i + 1])
        obs_ln.set_data(tt[: i + 1], obs[: i + 1])
        return lead_ln, ft_ln, fe_pt, ell_ln, rng_ln, lead_dot, err_ln, sig_ln, obs_ln

    fig.suptitle(
        "RT-OAC headline: real-time observability-aware control (planar leader-follower)",
        fontsize=13,
    )
    fig.tight_layout()
    frames = range(0, n, max(1, n // 120))  # cap gif length
    anim = FuncAnimation(fig, frame, frames=frames, blit=False)
    gif = res_dir / "headline_sim.gif"
    anim.save(gif, writer=PillowWriter(fps=15), dpi=90)
    frame(n - 1)
    png = res_dir / "headline_sim.png"
    fig.savefig(png, dpi=120)
    print(f"wrote {gif} and {png}")


if __name__ == "__main__":
    main()

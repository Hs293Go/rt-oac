"""Phase B prototype + headroom de-risk: directly minimize the recursive tangential PCRB.

Phase A (PROGRESS #16) found the recursive posterior CRLB on the follower's tangential position is
~3.6 m (3-sigma) on the STLOG-optimal orbit -- comparable to the ~1 m standoff, so the carried
controller never has a tight-enough estimate. The STLOG objective maximizes observability VOLUME,
not this recursive PCRB. Before wiring a new cost into RTController, this script answers the keystone
question: *can any orbit, kept inside the standoff band, push the recursive PCRB meaningfully below
3.6 m?* If yes, reformulation has headroom; if not, the range-only geometry is the limit (redirect).

It is a self-contained differentiable optimizer:
  decision vars  = the follower's input sequence (N x 4: thrust + 3 body rates), leader fixed (hover);
  forward        = integrate the relative dynamics -> trajectory; run the EKF Riccati at truth
                   (F/G/H reused from ErrorStateEKF) -> per-step P; the PCRB is trace(P[pos]);
  objective      = mean steady-state trace(P[pos])  +  w_band * (distance-band violation)^2,
                   so the optimizer cannot cheat by flying the follower out to a huge parallax.
Optimized with JAX value_and_grad + scipy L-BFGS-B. Compares the optimized orbit's PCRB and its
distance profile against the STLOG-orbit baseline.

    JAX_PLATFORMS=cpu uv run python experiments/pcrb_optimize.py [--steps 30] [--wband 50] [--iters 80]
"""

import argparse

from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator
import scipy.optimize as sopt

from rt_oac.error_state_ekf import ErrorStateEKF
from rt_oac.scenario import build_scenario

jax.config.update("jax_enable_x64", True)


def build_pcrb_fn(ekf, dt, in_cov, proc_cov, R4, x0, u_leader, P0, band, w_band):
    """Return a jitted (u_foll -> (pcrb_cost, aux)) over an N-step rollout from x0."""
    n = ekf._tangent_dim
    zero = jnp.zeros(n)
    lo2, hi2 = band[0] ** 2, band[1] ** 2

    def step_jacobians(x, u):
        x_next = ekf._step(x, u, dt)

        def err(d):
            return ekf._boxminus(ekf._step(ekf._manifold.boxplus(x, d), u, dt), x_next)

        def imap(ui):
            return ekf._boxminus(ekf._step(x, ui, dt), x_next)

        return x_next, jax.jacobian(err)(zero), jax.jacobian(imap)(u)

    def meas_jac(x):
        y = ekf._observation(x)
        return -jax.jacobian(lambda d: ekf._residual(ekf._manifold.boxplus(x, d), y))(
            zero
        )

    def rollout(u_foll):
        us = jnp.concatenate([u_leader, u_foll], axis=1)  # (N, 8)

        def scan_step(carry, u):
            x, P = carry
            x_next, F, G = step_jacobians(x, u)
            P = F @ P @ F.T + G @ in_cov @ G.T + proc_cov  # predict at truth
            H = meas_jac(x_next)
            S = H @ P @ H.T + R4
            P -= P @ H.T @ jnp.linalg.solve(S, H @ P)  # update at truth
            P = 0.5 * (P + P.T)
            d2 = jnp.dot(x_next[0:3], x_next[0:3])
            return (x_next, P), (jnp.trace(P[0:3, 0:3]), d2)

        _, (trP_pos, d2) = jax.lax.scan(scan_step, (x0, P0), us)
        return trP_pos, d2

    @jax.jit
    def cost_and_aux(u_foll):
        trP_pos, d2 = rollout(u_foll)
        half = trP_pos.shape[0] // 2
        j_pcrb = jnp.mean(trP_pos[half:])  # steady-state position-uncertainty trace
        viol = jnp.maximum(0.0, d2 - hi2) + jnp.maximum(0.0, lo2 - d2)
        j_band = jnp.mean(viol**2)
        return j_pcrb + w_band * j_band, (j_pcrb, j_band, trP_pos, d2)

    return cost_and_aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--wband", type=float, default=50.0)
    ap.add_argument("--iters", type=int, default=80)
    args = ap.parse_args()

    sc = build_scenario()
    dt = float(sc.cfg["sim"]["integrator_dt"])
    band = (
        0.5,
        1.2,
    )  # the aggressive #8 standoff band (conf/mode/{estimation,hybrid}.yaml)
    range_var = float(sc.cfg["noise"]["range_var"])
    att_var = float(sc.cfg["noise"]["att_var"])
    res_var = np.concatenate([[range_var], np.full(3, att_var)])
    in_cov = jnp.diag(jnp.asarray(np.tile([0.05, 0.01, 0.01, 0.01], 2)))
    proc_cov = jnp.diag(
        jnp.asarray([0.02, 0.02, 0.02, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05])
    )
    R4 = jnp.diag(jnp.asarray(res_var))
    P0 = jnp.diag(jnp.asarray([2.0, 2.0, 2.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0]))

    ekf = ErrorStateEKF(
        mdl.dynamics,
        lambda x: mdl.observation(x),
        mdl.MANIFOLD,
        in_cov=np.eye(8),
        obs_cov=np.diag(res_var),
        method=integrator.Methods.EULER,
    )

    x0 = jnp.asarray(np.concatenate([[1.0, 0.0, 0.0], _quat_z(20), [0.0, 1.0, 0.0]]))
    guess = np.asarray(sc.reference_guess(0))  # (window, 8) leader+follower warm start
    if args.steps > guess.shape[0]:  # extend the warm start by repeating the last input
        pad = np.repeat(guess[-1:], args.steps - guess.shape[0], axis=0)
        guess = np.concatenate([guess, pad], axis=0)
    N = min(args.steps, guess.shape[0])
    guess = guess[:N]
    u_leader = jnp.asarray(guess[:, 0:4])
    u_foll0 = jnp.asarray(guess[:, 4:8])

    cost_and_aux = build_pcrb_fn(
        ekf, dt, in_cov, proc_cov, R4, x0, u_leader, P0, band, args.wband
    )
    vag = jax.jit(jax.value_and_grad(lambda u: cost_and_aux(u)[0]))

    def scipy_obj(flat):
        u = jnp.asarray(flat.reshape(N, 4))
        v, g = vag(u)
        return float(v), np.asarray(g, dtype=np.float64).ravel()

    # baseline (STLOG-orbit warm start) metrics
    _, (j0, b0, _trP0, d20) = cost_and_aux(u_foll0)
    print(f"band {band}, N={N}, w_band={args.wband}")
    print(
        f"baseline (STLOG warm start): PCRB 3sig_pos={_sig(j0):.2f} m  "
        f"dist[{float(jnp.sqrt(d20.min())):.2f},{float(jnp.sqrt(d20.max())):.2f}]  bandviol={float(b0):.2e}"
    )

    res = sopt.minimize(
        scipy_obj,
        np.asarray(u_foll0).ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": args.iters},
    )
    u_opt = jnp.asarray(res.x.reshape(N, 4))
    _, (j1, b1, _trP1, d21) = cost_and_aux(u_opt)
    print(
        f"PCRB-optimized:              PCRB 3sig_pos={_sig(j1):.2f} m  "
        f"dist[{float(jnp.sqrt(d21.min())):.2f},{float(jnp.sqrt(d21.max())):.2f}]  bandviol={float(b1):.2e}"
    )
    print(
        f"\nheadroom: PCRB 3-sigma_pos {_sig(j0):.2f} m -> {_sig(j1):.2f} m  "
        f"(x{_sig(j0) / max(_sig(j1), 1e-9):.2f} tighter)"
    )
    print(
        "  >>1 and band held  => orbit shaping has real headroom; reformulation is worth wiring in."
    )
    print(
        "  ~1               => the range-only geometry is the limit; redirect from objective."
    )


def _quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def _sig(trace_pos):
    return float(3 * np.sqrt(max(float(trace_pos) / 3, 0)))


if __name__ == "__main__":
    main()

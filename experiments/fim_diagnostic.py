"""Phase A: diagnose the STLOG-vs-recursive-convergence mismatch (the #14 smoking gun).

Thesis: the controller maximizes a sum of per-node SHORT-TIME observability Gramians (STLOG),
each of which credits the full Lie-derivative stack up to order r*>=5 -- so each node looks fully
observable. But the recursive EKF/UKF gets only ONE scalar range per step; it must accumulate
tangential information across steps through the state-transition Phi. The honest object is the
trajectory-accumulated discrete observability Gramian (= Fisher information, whose inverse is the
Cramer-Rao bound on the posterior)::

    W_o(T) = sum_{k<T}  Phi(k,0)^T H_k^T R^-1 H_k Phi(k,0),   Phi(k,0) = F_{k-1} ... F_0,

with F_k the 9x9 tangent transition and H_k the 4x9 first-order measurement Jacobian -- both reused
verbatim from the ErrorStateEKF's own linearization. If W_o(T) is full-rank-but-ill-conditioned in
the tangential-position / velocity subspace while every per-node STLOG is healthy, the STLOG metric
over-credits what the filter receives, and no Gaussian filter can localize (P_pos must grow, #14).

Feed it a closed-loop dump that logged the full true relative state + applied input (the example now
dumps ``xrel_full`` / ``u_applied`` when ``report=`` is set), e.g. the truth-planned OA orbit::

    JAX_PLATFORMS=cpu uv run python examples/quadrotor_cooperative_navigation.py \\
        mode=estimation filter=ukf plan_on_truth=true steps=80 seed=0 report=/tmp/derisk
    JAX_PLATFORMS=cpu uv run python experiments/fim_diagnostic.py --npz /tmp/derisk/quad_estimation_planontruth.npz
"""

import argparse
import pathlib

from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator

from rt_oac.error_state_ekf import ErrorStateEKF
from rt_oac.scenario import build_scenario

REPO = pathlib.Path(__file__).resolve().parents[1]
# tangent blocks: position [0:3], attitude [3:6], velocity [6:9]
BLOCKS = {"pos": slice(0, 3), "att": slice(3, 6), "vel": slice(6, 9)}


def block_of(vec):
    """Which tangent block (pos/att/vel) holds most of an eigenvector's energy."""
    mass = {name: float(np.sum(vec[sl] ** 2)) for name, sl in BLOCKS.items()}
    return max(mass, key=mass.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz", default=None, help="trajectory dump (xrel_full + u_applied)"
    )
    ap.add_argument("--order", type=int, default=5)
    args = ap.parse_args()
    npz = args.npz or str(sorted(pathlib.Path("/tmp/derisk").glob("*.npz"))[-1])  # noqa: S108
    d = np.load(npz)
    xs = np.asarray(d["xrel_full"])  # (T, 10) true relative state
    us = np.asarray(d["u_applied"])  # (T, 8) applied input
    sig = np.asarray(d["sig"]) if "sig" in d else None  # actual UKF 3-sigma_pos
    T = xs.shape[0]
    print(f"trajectory: {npz}\n  {T} steps, state {xs.shape}, input {us.shape}")

    sc = build_scenario(order=args.order)
    dt = float(sc.cfg["sim"]["integrator_dt"])  # recursive filter step
    stlog_dt = float(sc.stlog_dt)  # STLOG short window T
    range_var = float(sc.cfg["noise"]["range_var"])
    att_var = float(sc.cfg["noise"]["att_var"])
    res_var = np.concatenate([[range_var], np.full(3, att_var)])  # 4-d residual space
    rinv = np.diag(1.0 / res_var)

    ekf = ErrorStateEKF(
        mdl.dynamics,
        lambda x: mdl.observation(x),
        mdl.MANIFOLD,
        in_cov=np.eye(8),
        obs_cov=np.diag(res_var),
        method=integrator.Methods.EULER,
    )
    n = ekf._tangent_dim
    zero = jnp.zeros(n)

    # process / input noise matching the closed-loop example (so the PCRB sees the real Q)
    input_var = np.tile([0.05, 0.01, 0.01, 0.01], 2)
    proc_var = np.array([0.02, 0.02, 0.02, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05])
    in_cov = np.diag(input_var)
    proc_cov = np.diag(proc_var)

    @jax.jit
    def transition(x, u):
        x_next = ekf._step(x, u, dt)

        def err(delta):
            return ekf._boxminus(
                ekf._step(ekf._manifold.boxplus(x, delta), u, dt), x_next
            )

        def input_map(u_in):
            return ekf._boxminus(ekf._step(x, u_in, dt), x_next)

        return jax.jacobian(err)(zero), jax.jacobian(input_map)(u)

    @jax.jit
    def meas_jac(x):
        y = ekf._observation(
            x
        )  # nu=0 at delta=0 so this is the pure measurement Jacobian

        def res(delta):
            return ekf._residual(ekf._manifold.boxplus(x, delta), y)

        return -jax.jacobian(res)(zero)  # H := d h_tangent / d delta

    # ---- accumulate the trajectory observability Gramian + per-node STLOG -------------
    stlog_dt_arr = jnp.broadcast_to(jnp.asarray(stlog_dt), us.shape[0])
    stlogs = np.asarray(
        jax.vmap(sc.cost.eval_gramian)(jnp.asarray(xs), jnp.asarray(us), stlog_dt_arr)
    )

    phi = np.eye(n)
    W = np.zeros((n, n))
    # PCRB: the recursive posterior covariance (EKF Riccati linearized at TRUTH, real Q/R) --
    # the best ANY recursive estimator can do on this trajectory WITH process noise.
    P = np.diag([2.0, 2.0, 2.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
    R4 = np.diag(res_var)
    lam_min_curve, stlog_min_curve, pcrb_sig_curve = [], [], []
    for k in range(T):
        H = np.asarray(meas_jac(jnp.asarray(xs[k])))  # (4, 9)
        Hphi = H @ phi
        W += Hphi.T @ rinv @ Hphi  # Phi(k,0)^T H^T R^-1 H Phi(k,0)
        lam_min_curve.append(float(np.linalg.eigvalsh(W)[0]))
        stlog_min_curve.append(float(np.linalg.eigvalsh(stlogs[k])[0]))
        F, G = (
            np.asarray(a) for a in transition(jnp.asarray(xs[k]), jnp.asarray(us[k]))
        )
        # PCRB Riccati: predict then update, all Jacobians at truth (no estimate error).
        P = F @ P @ F.T + G @ in_cov @ G.T + proc_cov
        S = H @ P @ H.T + R4
        P -= P @ H.T @ np.linalg.solve(S, H @ P)
        P = 0.5 * (P + P.T)
        pcrb_sig_curve.append(float(3 * np.sqrt(max(np.trace(P[0:3, 0:3]) / 3, 0))))
        phi = F @ phi

    # ---- report -----------------------------------------------------------------------
    evals, evecs = np.linalg.eigh(W)  # ascending
    print(
        f"\nsim dt={dt}, STLOG window={stlog_dt}, R=diag(range {range_var}, att {att_var})"
    )
    print(
        "\n=== per-node STLOG (the OPTIMIZER's metric): min-eigenvalue across nodes ==="
    )
    sm = np.array(stlog_min_curve)
    print(
        f"  min-eig(STLOG_k): min {sm.min():.2e}  median {np.median(sm):.2e}  max {sm.max():.2e}"
    )
    print(
        "  -> FLOORED by the T^(2r*+1) short-time scaling: the worst direction is numerically"
    )
    print(
        "     singular per-node, so the log-det/softmin objective optimizes VOLUME (the easy"
    )
    print("     attitude/radial directions), not the starved tangential one.")

    print(
        "\n=== accumulated trajectory Gramian W_o(T) (what the FILTER actually gets) ==="
    )
    print("  eigenvalues (ascending) and the tangent block each eigenvector lives in:")
    for i in range(n):
        print(f"    lam[{i}] = {evals[i]:.3e}   block={block_of(evecs[:, i])}")
    print(f"  condition number {evals[-1] / max(evals[0], 1e-300):.2e}")
    smallest_blocks = [block_of(evecs[:, i]) for i in range(3)]
    print(f"  -> the 3 smallest eigen-directions are in: {smallest_blocks}")

    print("\n=== information ACCUMULATION: does lam_min(W_o(t)) grow or saturate? ===")
    lm = np.array(lam_min_curve)
    for frac in (0.25, 0.5, 0.75, 1.0):
        t = int(frac * T) - 1
        print(
            f"  t={t + 1:3d}/{T}: lam_min(W_o)={lm[t]:.3e}   STLOG_min(node)={sm[t]:.3e}"
        )
    growth = lm[-1] / max(lm[T // 4], 1e-300)
    print(
        f"  lam_min growth over last 3/4 of the run: x{growth:.2f}  (>>1 = accumulating; ~1 = starved)"
    )

    # batch CRLB (x_0, no process noise) vs PCRB (recursive, real Q) vs the actual filter
    crlb_pos = np.linalg.inv(W)[0:3, 0:3]
    crlb_sig = float(3 * np.sqrt(max(np.trace(crlb_pos) / 3, 0)))
    pc = np.array(pcrb_sig_curve)
    print(
        "\n=== the decisive comparison: batch CRLB vs recursive PCRB vs actual filter ==="
    )
    print(
        f"  batch CRLB 3-sigma_pos (estimate x_0, NO process noise) = {crlb_sig:.2f} m"
    )
    print(
        f"  recursive PCRB 3-sigma_pos (Riccati at TRUTH, real Q):   start {pc[0]:.2f} -> "
        f"end {pc[-1]:.2f} m  (steady {np.median(pc[len(pc) // 2 :]):.2f})"
    )
    if sig is not None:
        print(
            f"  actual UKF 3-sigma_pos (mean not pinned):                start {sig[0]:.2f} -> "
            f"end {sig[-1]:.2f} m"
        )
    print(
        "  Reading: PCRB is the floor for ANY recursive estimator on THIS trajectory with process"
    )
    print(
        "  noise. PCRB small => trajectory IS recursively informative => the carried failure is"
    )
    print(
        "  CONTROL/mean-extraction, not objective. PCRB large => trajectory cannot support"
    )
    print("  recursive localization => reformulating the OA objective is the lever.")


if __name__ == "__main__":
    main()

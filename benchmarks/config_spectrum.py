"""Eigenvalue-spectrum analysis: per-node vs accumulated observability, MUC vs non-MUC.

Single-node STLOG min-eig ~ T^(2r*+1) (≈ T^11 at order 5, T=0.2) sits at the numerical
floor for every config. Two things to disentangle:
  * The paper metric sums PER-NODE min-eig: sum_k lambda_min(W_k). Degenerate if each short
    window is individually rank-deficient.
  * The ACCUMULATED trajectory observability lambda_min(sum_k W_k) -- information adds across
    the horizon and may be healthy even when per-node is ~0.
Reports both, plus the full spectrum, for the co-hover MUC vs candidate non-MUC configs,
rolling out the 20-node trajectory from each initial state with the reference input guess.
"""

import jax
import numpy as np

import rt_oac  # noqa: F401
from rt_oac.scenario import build_scenario


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def state(r, q, v):
    return np.concatenate([np.asarray(r, float), q, np.asarray(v, float)])


def main():
    sc = build_scenario()
    dt = sc.stlog_dt
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    u0 = sc.reference_guess(0)  # (20,8) leader ref + follower hover

    # roll out trajectory from x0 with the reference inputs, gramians at each node
    gramians_sliced = jax.jit(
        lambda x: sc.cost(x, u0, dt, return_gramians=True).gramians
    )

    configs = {
        "MUC co-hover": state([0, -1, 1], ident, [0, 0, 0]),
        "tangential vel + 20deg att": state([2, 0, 0], quat_z(20), [0, 1, 0]),
        "thesis-rec": state([5, 0, 0], quat_z(15), [0, 1, 0.2]),
        "rich offset+tangential+tilt": state([1, 2, 0.5], quat_z(25), [0.3, 1.0, -0.2]),
    }

    for name, x in configs.items():
        g = np.asarray(gramians_sliced(x))  # (20, 6, 6)
        g = 0.5 * (g + np.swapaxes(g, -1, -2))
        per_node_mineig = np.linalg.eigvalsh(g)[:, 0]  # (20,)
        acc = g.sum(axis=0)  # accumulated info over the horizon
        acc_eig = np.linalg.eigvalsh(acc)
        print(f"\n## {name}")
        print(
            f"  per-node min-eig: sum={per_node_mineig.sum():.3e} "
            f"max={per_node_mineig.max():.3e}  (paper metric components)"
        )
        print(
            f"  ACCUMULATED sum_k W_k spectrum: "
            f"min={acc_eig[0]:.3e} ... max={acc_eig[-1]:.3e}"
        )
        print(
            f"    full spectrum: {np.array2string(acc_eig, precision=2, suppress_small=False)}"
        )
        print(
            f"    cond(accumulated) = {acc_eig[-1] / acc_eig[0]:.2e}"
            if acc_eig[0] > 0
            else f"    accumulated min-eig <= 0 ({acc_eig[0]:.2e})"
        )


if __name__ == "__main__":
    main()

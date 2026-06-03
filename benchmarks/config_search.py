"""Find a non-pathological initial configuration (well-conditioned observability Gramian).

The thesis identifies 4 manifestly unobservable configurations (MUCs) for the range+attitude
relative quadrotor: (1) zero relative velocity / co-hover, (2) radial co-motion v || r,
(3) perfect orbit (r x v planar = 0), (4) identical attitudes q_fl = identity. The default
scenario (x0 = [0,-1,1, 0,0,0,1, 0,0,0], hover) hits MUCs 1 and 4. This sweeps candidate
initial relative states that avoid all four and reports the sliced (position+velocity) and
full ambient Gramian eigenvalue scale, to pick a non-degenerate config for the real-time
study.

State layout: [r_lf(0:3), q_fl(3:7) xyzw, v_lf(7:10)] in the follower body frame.
"""

import jax
import numpy as np

import rt_oac  # noqa: F401
from rt_oac.scenario import build_scenario


def quat_z(deg):
    """Xyzw quaternion for a yaw rotation of ``deg`` degrees."""
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def state(r, q, v):
    return np.concatenate([
        np.asarray(r, float),
        np.asarray(q, float),
        np.asarray(v, float),
    ])


def main():
    sc = build_scenario()  # ambient 10x10, sliced to observed_components [0,1,2,7,8,9]
    dt = sc.stlog_dt
    hover = np.array([9.81, 0, 0, 0, 9.81, 0, 0, 0])  # both robots hover

    # sliced Gramian (6x6 pos+vel) and full ambient 10x10
    g_sliced = jax.jit(lambda x, u: sc.cost.eval_gramian(x, u, dt))
    g_full = jax.jit(lambda x, u: sc.cost._stlog(x, u, dt))  # unsliced 10x10

    ident = np.array([0.0, 0.0, 0.0, 1.0])

    candidates = {
        "MUC co-hover (baseline x0)": (state([0, -1, 1], ident, [0, 0, 0]), hover),
        "MUC1 zero-vel, tilted att": (state([2, 0, 0], quat_z(20), [0, 0, 0]), hover),
        "tangential vel only": (state([2, 0, 0], ident, [0, 1, 0]), hover),
        "tangential vel + 20deg att": (state([2, 0, 0], quat_z(20), [0, 1, 0]), hover),
        "thesis-rec (r=5x, v=[0,1,.2], 15deg)": (
            state([5, 0, 0], quat_z(15), [0, 1, 0.2]),
            hover,
        ),
        "rich: offset+tangential+tilt": (
            state([1, 2, 0.5], quat_z(25), [0.3, 1.0, -0.2]),
            hover,
        ),
        "radial vel (MUC2-ish) v||r": (state([2, 0, 0], quat_z(20), [1, 0, 0]), hover),
        "follower banking input": (
            state([2, 0, 0], quat_z(20), [0, 1, 0]),
            np.array([9.81, 0, 0, 0, 10.5, 0.5, 0.3, 0]),
        ),
    }

    print(f"{'config':<42}{'min-eig(6x6)':>14}{'cond(6x6)':>12}{'min-eig(10x10)':>16}")
    for name, (x, u) in candidates.items():
        gs = np.asarray(g_sliced(x, u))
        gs = 0.5 * (gs + gs.T)
        es = np.linalg.eigvalsh(gs)
        gf = np.asarray(g_full(x, u))
        gf = 0.5 * (gf + gf.T)
        ef = np.linalg.eigvalsh(gf)
        # full 10x10 has the quaternion gauge null -> smallest is ~0 by construction;
        # report the smallest NONZERO-ish (2nd smallest) for the full matrix.
        cond_s = es[-1] / es[0] if es[0] > 0 else np.inf
        print(f"{name:<42}{es[0]:>14.3e}{cond_s:>12.2e}{ef[0]:>16.3e}")


if __name__ == "__main__":
    main()

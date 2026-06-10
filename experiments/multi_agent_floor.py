r"""Measurement-multiplicity floor check: does adding agents (range channels) make a TIGHT carried
loop reachable, or is the single-range floor irreducible?

The Phase-2 marginality (25% bounded) is the #14 ridge: with range to ONE anchor (the leader), the
follower's two TANGENTIAL directions are observable only via MOTION (weak) -- the recursive filter
collapses/diverges on that thin curved ridge. Measurement multiplicity (a follower ranging to N-1
other agents) makes the tangential observable INSTANTANEOUSLY (trilateration), removing the
motion-dependence and the ridge. This quantifies the position CRLB floor for N agents:

  static FIM  F = sum_i u_i u_i^T / sigma_r^2     (u_i = LOS unit vector to anchor i)

so rank(F) = # of independent LOS directions: 1 range -> rank 1 (radial only, 2 tangential null);
2 ranges -> rank 2 (a plane, 1 normal null); 3 non-coplanar ranges -> rank 3 (FULL, instantaneous).
Reports the worst-direction 1-sigma floor for N=2,3,4 agents, in a good formation and a DEGENERATE
(coplanar) one (the geometry the closed-loop controller could steer into). sigma_r=0.1 m, standoff 5 m.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/multi_agent_floor.py
"""

import numpy as np

SIGMA_R = 0.1  # range-measurement noise std (m), matching the OAC obs VAR range = 1e-2
STANDOFF = 5.0
FOLLOWER = np.array([
    0.0,
    5.0,
    0.0,
])  # the agent we localize (relative to the leader at origin)

# Anchor sets: the leader (origin) + extra agents the follower also ranges to. N agents total = leader
# + follower + (N-2) extra anchors, so the follower has N-1 range channels.
GOOD = {  # a well-spread (non-coplanar) formation
    2: [np.array([0.0, 0.0, 0.0])],  # leader only -> 1 range
    3: [
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0]),
    ],  # + agent in-plane -> 2 ranges
    4: [
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 4.0]),
    ],  # +out-of-plane
}
DEGENERATE = {  # all anchors collinear with the leader along x (coplanar/degenerate)
    2: [np.array([0.0, 0.0, 0.0])],
    3: [np.array([0.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])],
    4: [
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
        np.array([6.0, 0.0, 0.0]),
    ],
}


def fim_pos(p, anchors):
    f = np.zeros((3, 3))
    for a in anchors:
        d = p - a
        u = d / np.linalg.norm(d)
        f += np.outer(u, u) / SIGMA_R**2
    return f


def worst_floor(f):
    """rank, worst-direction 1-sigma floor (inf if rank-deficient), condition number."""
    vals = np.linalg.eigvalsh(f)
    rank = int(np.sum(vals > 1e-6 * vals.max()))
    lam_min = vals[0]
    floor = np.inf if lam_min <= 1e-6 * vals.max() else 1.0 / np.sqrt(lam_min)
    cond = vals.max() / max(lam_min, 1e-12)
    return rank, floor, cond


def recursive_floor(anchors_rel, traj, q_pos=0.05):
    """Recursive position CRLB (Riccati at truth) along the follower trajectory, anchors riding the
    formation. q_pos = per-step position process-noise std (the motion that lets a rank-deficient
    static geometry resolve the null direction over time). Returns steady-state worst 1-sigma floor."""
    p_cov = np.eye(3) * 9.0
    qmat = q_pos**2 * np.eye(3)
    for p in traj:
        p_cov += qmat  # predict (random-walk position)
        info = np.linalg.inv(p_cov)
        for a in anchors_rel:
            d = p - a
            u = d / np.linalg.norm(d)
            info += np.outer(u, u) / SIGMA_R**2  # range update
        p_cov = np.linalg.inv(info)
    return float(np.sqrt(np.linalg.eigvalsh(p_cov)[-1]))  # worst-direction 1-sigma


def main():
    print("=== Measurement-multiplicity floor check (sigma_r=0.1 m, standoff 5 m) ===")
    print(
        "Follower localized via range to the leader + (N-2) extra agents = N-1 range channels.\n"
    )
    print("STATIC instantaneous floor (no motion) -- the #14-ridge test:")
    print(
        f"{'N agents':>9} {'channels':>9} {'rank/3':>7} {'worst 1sig':>11} {'cond':>9}  geometry"
    )
    for name, sets in [
        ("good (non-coplanar)", GOOD),
        ("degenerate (coplanar)", DEGENERATE),
    ]:
        for n in (2, 3, 4):
            rank, floor, cond = worst_floor(fim_pos(FOLLOWER, sets[n]))
            fs = "inf (motion-only)" if np.isinf(floor) else f"{floor:.3f} m"
            print(f"{n:>9} {n - 1:>9} {rank:>5}/3 {fs:>11} {cond:>9.1f}  {name}")
        print()

    # RECURSIVE floor along an OA-like orbit (so the rank-deficient cases get motion to resolve the null)
    th = np.linspace(0, 2 * np.pi, 120)
    traj = (
        FOLLOWER + np.c_[1.5 * np.sin(2 * th), 0.8 * np.sin(th), 0.5 * np.sin(3 * th)]
    )  # maneuvering
    print(
        "RECURSIVE floor along a maneuvering orbit (motion resolves null dirs), q_pos=0.05 m/step:"
    )
    print(
        f"{'N agents':>9} {'worst 1sig':>11}  (vs 5 m standoff, vs 2.26 m init error)"
    )
    for n in (2, 3, 4):
        f = recursive_floor(GOOD[n], traj)
        print(f"{n:>9} {f:>9.3f} m")
    print(
        "\nDECISION: if N>=3 drops the worst (tangential) floor far below the standoff INSTANTANEOUSLY "
        "(static, not motion-reliant), a TIGHT carried loop is reachable -- the #14 ridge is removed by "
        "geometry, not by a fancier filter. If it stays inf/large until 4 agents or only resolves via "
        "motion, multiplicity helps less than hoped. Degenerate (coplanar) rows show the geometry the "
        "closed-loop controller must be kept out of."
    )


if __name__ == "__main__":
    main()

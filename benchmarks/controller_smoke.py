"""Smoke-test RTController: log-det + SLSQP, cold-start vs warm-start, at a non-MUC config.

Validates the controller artifact end to end and demonstrates the warm-start iteration cut
that targets the 100 ms real-time budget.
"""

import numpy as np

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.scenario import build_rt_controller, build_scenario
from rt_oac.warmstart import warm_guess


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def main():
    sc = build_scenario(gramian_metric=metrics.neg_logdet)
    ctrl = build_rt_controller(sc, method="SLSQP", maxiter=80)

    x0 = np.concatenate([[2.0, 0, 0], quat_z(20), [0, 1.0, 0]])  # non-MUC
    ref = sc.reference_guess(0)

    # cold start (includes compile)
    r0 = ctrl.solve(x0, ref)
    print(
        f"cold start (incl compile): obj={r0.objective:.4g} nit={r0.nit} "
        f"nfev={r0.nfev} viol={r0.max_constraint_violation:.2e} wall={r0.wall_time:.2f}s "
        f"success={r0.success}"
    )

    # cold start again (steady, compiled)
    r1 = ctrl.solve(x0, ref)
    print(
        f"cold start (steady):       obj={r1.objective:.4g} nit={r1.nit} "
        f"wall={r1.wall_time:.3f}s"
    )

    # warm start: seed the next solve from the shifted previous solution
    guess = warm_guess(sc.reference_guess(1), r1.u)
    r2 = ctrl.solve(x0, guess)
    print(
        f"warm start (shifted prev): obj={r2.objective:.4g} nit={r2.nit} "
        f"wall={r2.wall_time:.3f}s  -> iters {r1.nit}->{r2.nit}"
    )

    # a short receding-horizon warm-started chain to see steady iteration count
    prev = r1.u
    nits, walls = [], []
    for step in range(2, 8):
        guess = warm_guess(sc.reference_guess(step), prev)
        r = ctrl.solve(x0, guess)
        prev = r.u
        nits.append(r.nit)
        walls.append(r.wall_time)
    print(f"warm-started chain nits={nits} walls(ms)={[round(w * 1e3) for w in walls]}")


if __name__ == "__main__":
    main()

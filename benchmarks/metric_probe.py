"""Linchpin test: can a smooth surrogate give usable gradients and climb observability?

For baseline (reciprocal min-eig) and two surrogates (neg_logdet, neg_softmin_eig):
  * gradient norm at the reference guess (baseline expected ~0 -> stuck);
  * SLSQP solve: wall, iters, and the TRUE observability achieved (sum & min of the
    per-node minimum eigenvalue), scored on the real Gramian, not the surrogate.

If a surrogate lifts the true min-eigenvalue from ~0 quickly while the baseline cannot,
the real-time path runs through reframing the objective.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.scenario import build_scenario


def main():
    base = build_scenario()  # baseline cost, used to score TRUE observability
    dt, x0 = base.stlog_dt, base.x0
    u0 = base.reference_guess(0)
    window = base.window

    min_idx = jnp.asarray(base.follower_indices)
    const_idx = jnp.setdiff1d(jnp.arange(8), min_idx)
    u_const = jnp.asarray(u0)[..., const_idx]
    free0 = np.asarray(jnp.asarray(u0)[..., min_idx].ravel())

    lb = np.tile(base.cfg["optim"]["lb"], (window, 2))[
        ..., np.asarray(base.follower_indices)
    ].ravel()
    ub = np.tile(base.cfg["optim"]["ub"], (window, 2))[
        ..., np.asarray(base.follower_indices)
    ].ravel()
    bnds = list(zip(lb, ub, strict=True))
    lo = base.cfg["opc"]["min_inter_vehicle_distance"] ** 2
    hi = base.cfg["opc"]["max_inter_vehicle_distance"] ** 2

    def recombine(uf):
        full = jnp.zeros((window, 8)).at[:, const_idx].set(u_const)
        return full.at[:, min_idx].set(uf.reshape(window, -1))

    # TRUE observability of a free vector: per-node min eigenvalue via the baseline cost.
    true_gramians = jax.jit(
        lambda uf: base.cost(x0, recombine(uf), dt, return_gramians=True).gramians
    )

    def true_stats(uf):
        g = np.asarray(true_gramians(np.asarray(uf)))
        g = 0.5 * (g + np.swapaxes(g, -1, -2))
        eig = np.linalg.eigvalsh(g)  # (N, m)
        mineig = eig[:, 0]
        return mineig.sum(), mineig.min(), mineig.max()

    # constraint (metric-independent): squared inter-vehicle distance per node
    cdist = jax.jit(
        lambda uf: (base.cost.eval_integrator(x0, recombine(uf))[0][:, 0:3] ** 2).sum(1)
    )
    cdist_j = jax.jit(
        jax.jacobian(
            lambda uf: (
                base.cost.eval_integrator(x0, recombine(uf))[0][:, 0:3] ** 2
            ).sum(1)
        )
    )
    cons = [
        {
            "type": "ineq",
            "fun": lambda u: np.asarray(cdist(u)) - lo,
            "jac": lambda u: np.asarray(cdist_j(u)),
        },
        {
            "type": "ineq",
            "fun": lambda u: hi - np.asarray(cdist(u)),
            "jac": lambda u: -np.asarray(cdist_j(u)),
        },
    ]

    s0_sum, s0_min, s0_max = true_stats(free0)
    print(
        f"reference guess TRUE observability: sum(min_eig)={s0_sum:.4e} "
        f"min={s0_min:.4e} max(min_eig over nodes)={s0_max:.4e}\n"
    )

    configs = [
        ("baseline reciprocal_min_eig", metrics.reciprocal_min_eig),
        ("neg_logdet", metrics.neg_logdet),
        ("neg_softmin_eig", metrics.neg_softmin_eig),
    ]
    for name, metric in configs:
        sc = build_scenario(gramian_metric=metric)
        obj = jax.jit(lambda uf, m=sc.cost: m(x0, recombine(uf), dt).objective)
        grad = jax.jit(
            jax.grad(lambda uf, m=sc.cost: m(x0, recombine(uf), dt).objective)
        )
        gnorm = float(jnp.linalg.norm(jax.block_until_ready(grad(free0))))

        f = lambda u: float(obj(np.asarray(u)))  # noqa: E731
        g = lambda u: np.asarray(grad(np.asarray(u)))  # noqa: E731
        # warmup compile
        optimize.minimize(
            f,
            free0,
            jac=g,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": 2},
        )
        t = time.perf_counter()
        soln = optimize.minimize(
            f,
            free0,
            jac=g,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": 60},
        )
        wall = time.perf_counter() - t
        ssum, smin, smax = true_stats(soln.x)
        print(f"## {name}")
        print(f"  grad-norm @ ref: {gnorm:.4e}")
        print(f"  SLSQP: wall {wall:.3f} s | nit {soln.nit} | success {soln.success}")
        print(
            f"  TRUE observability after: sum(min_eig)={ssum:.4e} "
            f"(ref {s0_sum:.4e}, x{ssum / max(s0_sum, 1e-30):.1f}) "
            f"min={smin:.4e} max={smax:.4e}\n"
        )


if __name__ == "__main__":
    main()

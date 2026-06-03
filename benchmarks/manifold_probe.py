"""Is the principled tangent-space (9x9) manifold Gramian a non-degenerate objective?

The sliced ambient (r,v) 6x6 objective is structurally near-singular (min-eig ~ 1e-9 even
after optimization). This tests the standard observability objective instead: the full
tangent-space Gramian (quaternion gauge removed analytically), no slicing. Reports the
eigenvalue scale at the reference and what log-det optimization achieves.
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
    # Full tangent-space Gramian: manifold on, no observed-index slicing -> 9x9.
    base = build_scenario(use_manifold=True, observed_indices=())
    dt, x0 = base.stlog_dt, base.x0
    u0 = base.reference_guess(0)
    window = base.window

    min_idx = jnp.asarray(base.follower_indices)
    const_idx = jnp.setdiff1d(jnp.arange(8), min_idx)
    u_const = jnp.asarray(u0)[..., const_idx]
    free0 = np.asarray(jnp.asarray(u0)[..., min_idx].ravel())
    fi = np.asarray(base.follower_indices)

    lb = np.tile(base.cfg["optim"]["lb"], (window, 2))[..., fi].ravel()
    ub = np.tile(base.cfg["optim"]["ub"], (window, 2))[..., fi].ravel()
    bnds = list(zip(lb, ub, strict=True))
    lo = base.cfg["opc"]["min_inter_vehicle_distance"] ** 2
    hi = base.cfg["opc"]["max_inter_vehicle_distance"] ** 2

    def recombine(uf):
        full = jnp.zeros((window, 8)).at[:, const_idx].set(u_const)
        return full.at[:, min_idx].set(uf.reshape(window, -1))

    true_gramians = jax.jit(
        lambda uf: base.cost(x0, recombine(uf), dt, return_gramians=True).gramians
    )

    def eig_stats(uf):
        g = np.asarray(true_gramians(np.asarray(uf)))
        g = 0.5 * (g + np.swapaxes(g, -1, -2))
        eig = np.linalg.eigvalsh(g)  # (N, tangent_dim)
        return eig[:, 0], eig[:, -1]  # per-node min, max

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

    g0 = np.asarray(true_gramians(free0))
    print(f"manifold tangent Gramian shape per node: {g0.shape[1:]}")
    mn, mx = eig_stats(free0)
    print(
        f"reference: min-eig per node in [{mn.min():.3e}, {mn.max():.3e}], "
        f"max-eig per node max {mx.max():.3e}, sum(min-eig)={mn.sum():.4e}\n"
    )

    for name, metric in [
        ("baseline reciprocal_min_eig", metrics.reciprocal_min_eig),
        ("neg_logdet", metrics.neg_logdet),
    ]:
        sc = build_scenario(
            use_manifold=True, observed_indices=(), gramian_metric=metric
        )
        obj = jax.jit(lambda uf, m=sc.cost: m(x0, recombine(uf), dt).objective)
        grad = jax.jit(
            jax.grad(lambda uf, m=sc.cost: m(x0, recombine(uf), dt).objective)
        )
        gnorm = float(jnp.linalg.norm(jax.block_until_ready(grad(free0))))
        f = lambda u: float(obj(np.asarray(u)))  # noqa: E731
        g = lambda u: np.asarray(grad(np.asarray(u)))  # noqa: E731
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
        mn, mx = eig_stats(soln.x)
        print(f"## {name}")
        print(f"  grad-norm @ ref: {gnorm:.4e}")
        print(f"  SLSQP: wall {wall:.3f} s | nit {soln.nit} | success {soln.success}")
        print(
            f"  after: min-eig per node in [{mn.min():.3e}, {mn.max():.3e}], "
            f"sum(min-eig)={mn.sum():.4e}\n"
        )


if __name__ == "__main__":
    main()

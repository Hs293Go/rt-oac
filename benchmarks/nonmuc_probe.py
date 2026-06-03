"""Re-measure the OAC solve at a NON-pathological config (the real Phase 0 picture).

At the co-hover MUC the objective had zero gradient and the solve thrashed. Here we use a
non-MUC initial state (tangential relative velocity + non-identity attitude) and ask:
  * is the baseline min-eig gradient now nonzero (does the paper objective work off-MUC)?
  * do surrogates improve ACCUMULATED observability (log-det and #observable directions of
    sum_k W_k), and by how much vs the reference inputs?
  * how fast is SLSQP (wall, nit)?
Scoring uses the accumulated Gramian sum_k W_k (per-node min-eig is always ~T^11 ~ 0).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.scenario import build_scenario


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


CONFIGS = {
    "tangential vel + 20deg att": np.concatenate([
        [2.0, 0, 0],
        quat_z(20),
        [0, 1.0, 0],
    ]),
    "thesis-rec": np.concatenate([[5.0, 0, 0], quat_z(15), [0, 1.0, 0.2]]),
}


def main():
    base = build_scenario()
    dt = base.stlog_dt
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

    def make_scoring(x0):
        gram = jax.jit(
            lambda uf: base.cost(x0, recombine(uf), dt, return_gramians=True).gramians
        )

        def score(uf):
            g = np.asarray(gram(np.asarray(uf)))
            acc = 0.5 * (g + np.swapaxes(g, -1, -2)).sum(0)
            eig = np.linalg.eigvalsh(acc)
            n_obs = int(
                (eig > 1e-6 * eig[-1]).sum()
            )  # observable directions (rel. thresh)
            logdet = float(np.log(np.maximum(eig, 0) + 1e-3).sum())
            return n_obs, eig[0], logdet

        return score

    for cname, x0 in CONFIGS.items():
        score = make_scoring(x0)
        n0, mn0, ld0 = score(free0)
        print(f"\n==== {cname} ====")
        print(
            f"  reference inputs: #observable(acc)={n0}/6  acc-min-eig={mn0:.3e}  acc-logdet={ld0:.3f}"
        )
        cdist = jax.jit(
            lambda uf, x=x0: (
                base.cost.eval_integrator(x, recombine(uf))[0][:, 0:3] ** 2
            ).sum(1)
        )
        cdist_j = jax.jit(
            jax.jacobian(
                lambda uf, x=x0: (
                    base.cost.eval_integrator(x, recombine(uf))[0][:, 0:3] ** 2
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
        for mname, metric in [
            ("baseline min-eig", metrics.reciprocal_min_eig),
            ("neg_logdet", metrics.neg_logdet),
            ("neg_softmin", metrics.neg_softmin_eig),
        ]:
            sc = build_scenario(gramian_metric=metric)
            obj = jax.jit(lambda uf, m=sc.cost, x=x0: m(x, recombine(uf), dt).objective)
            grad = jax.jit(
                jax.grad(lambda uf, m=sc.cost, x=x0: m(x, recombine(uf), dt).objective)
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
                options={"maxiter": 80},
            )
            wall = time.perf_counter() - t
            n1, mn1, ld1 = score(soln.x)
            print(
                f"  {mname:<18} gnorm={gnorm:.2e} | SLSQP {wall:.2f}s nit={soln.nit} | "
                f"#obs {n0}->{n1}  acc-min-eig {mn1:.2e}  acc-logdet {ld0:.2f}->{ld1:.2f}"
            )


if __name__ == "__main__":
    main()

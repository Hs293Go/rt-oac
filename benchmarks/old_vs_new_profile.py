"""The transfer gate: fair, controlled old-vs-new profiling on the quadrotor.

Old setup = trust-constr + min-eigenvalue objective. New setup = SLSQP + log-det.
We must show the efficacy gain is the OBJECTIVE + SOLVER, not "we gave up iterating early".
So we run a 2x2 factorial {trust-constr, SLSQP} x {min-eig, log-det} on the *same* non-MUC
quadrotor config (order 5), and measure, with everything scored on one objective-agnostic
observability metric (accumulated Gramian sum_k W_k):

  1. per-component time attribution (rollout / objective / gradient / constraint /
     constraint-Jac / solver-internal) for OLD vs NEW;
  2. convergence of true observability vs iteration AND vs wall-time, all four cells, so:
     - early-stop justified?  new(@6) vs new(@40): does observability plateau by ~6?
     - efficacy at matched budget: old vs new at the SAME maxiter;
     - objective effect (same solver, swap objective) vs solver effect (same objective).

Per-iteration speedup and speed-at-matched-quality fall out of the same data.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from scipy import optimize

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.scenario import build_scenario

mpl.use("Agg")
import matplotlib.pyplot as plt

MAXITER = 40
EARLY = 6  # the "new" real-time iteration cap


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


# non-MUC operating point: tangential relative velocity + non-identity attitude
X0 = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])


def main():
    base = build_scenario()
    dt, window = base.stlog_dt, base.window
    fi = np.asarray(base.follower_indices)
    min_idx = jnp.asarray(base.follower_indices)
    const_idx = jnp.setdiff1d(jnp.arange(8), min_idx)
    u0 = base.reference_guess(0)
    u_const = jnp.asarray(u0)[..., const_idx]
    free0 = np.asarray(jnp.asarray(u0)[..., min_idx].ravel())
    lb = np.tile(base.cfg["optim"]["lb"], (window, 2))[..., fi].ravel()
    ub = np.tile(base.cfg["optim"]["ub"], (window, 2))[..., fi].ravel()
    lo = base.cfg["opc"]["min_inter_vehicle_distance"] ** 2
    hi = base.cfg["opc"]["max_inter_vehicle_distance"] ** 2

    def recombine(uf):
        full = jnp.zeros((window, 8)).at[:, const_idx].set(u_const)
        return full.at[:, min_idx].set(uf.reshape(window, -1))

    # objective-agnostic TRUE observability: accumulated Gramian sum_k W_k (gramians are
    # raw, independent of which metric a cost was built with).
    gram = jax.jit(
        lambda uf: base.cost(X0, recombine(uf), dt, return_gramians=True).gramians
    )

    def true_obs(uf):
        g = np.asarray(gram(np.asarray(uf)))
        acc = 0.5 * (g + np.swapaxes(g, -1, -2)).sum(0)
        eig = np.linalg.eigvalsh(acc)
        return float(eig[0]), int((eig > 1e-6 * eig[-1]).sum())

    cost = {
        "min-eig": build_scenario(gramian_metric=metrics.reciprocal_min_eig).cost,
        "log-det": build_scenario(gramian_metric=metrics.neg_logdet).cost,
    }
    objgrad = {}
    for k, c in cost.items():
        objgrad[k] = (
            jax.jit(lambda uf, c=c: c(X0, recombine(uf), dt).objective),
            jax.jit(jax.grad(lambda uf, c=c: c(X0, recombine(uf), dt).objective)),
        )
    cdist = jax.jit(
        lambda uf: (base.cost.eval_integrator(X0, recombine(uf))[0][:, 0:3] ** 2).sum(1)
    )
    cdist_j = jax.jit(
        jax.jacobian(
            lambda uf: (
                base.cost.eval_integrator(X0, recombine(uf))[0][:, 0:3] ** 2
            ).sum(1)
        )
    )

    def constraints(method):
        cv = lambda u: np.asarray(cdist(u))  # noqa: E731
        cj = lambda u: np.asarray(cdist_j(u))  # noqa: E731
        if method == "SLSQP":
            return [
                {"type": "ineq", "fun": lambda u: cv(u) - lo, "jac": cj},
                {"type": "ineq", "fun": lambda u: hi - cv(u), "jac": lambda u: -cj(u)},
            ]
        return optimize.NonlinearConstraint(
            cv, np.full(window, lo), np.full(window, hi), jac=cj
        )

    def bounds(method):
        return (
            list(zip(lb, ub, strict=True))
            if method == "SLSQP"
            else optimize.Bounds(lb, ub)
        )

    def solve(objective, method, record=True):
        f0, g0 = objgrad[objective]
        f = lambda u: float(f0(np.asarray(u)))  # noqa: E731
        g = lambda u: np.asarray(g0(np.asarray(u)))  # noqa: E731
        opts = {"maxiter": MAXITER}
        if method != "SLSQP":
            opts["verbose"] = 0
        optimize.minimize(
            f,
            free0,
            jac=g,
            method=method,
            bounds=bounds(method),
            constraints=constraints(method),
            options={"maxiter": 2},
        )  # warmup
        trace = []
        t0 = time.perf_counter()

        def cb(xk, *a):
            if record:
                trace.append((time.perf_counter() - t0, np.array(xk)))

        soln = optimize.minimize(
            f,
            free0,
            jac=g,
            method=method,
            bounds=bounds(method),
            constraints=constraints(method),
            callback=cb,
            options=opts,
        )
        wall = time.perf_counter() - t0
        return soln, wall, trace

    # ---- component time attribution (Timer-wrapped) -------------------------------
    class T:
        def __init__(self, fn):
            self.fn, self.t, self.n = fn, 0.0, 0

        def __call__(self, *a):
            s = time.perf_counter()
            out = np.asarray(jax.block_until_ready(self.fn(*a)))
            self.t += time.perf_counter() - s
            self.n += 1
            return out

    def attribute(objective, method, maxiter):
        f0, g0 = objgrad[objective]
        f = T(lambda u: f0(np.asarray(u)))
        g = T(lambda u: g0(np.asarray(u)))
        roll = T(lambda u: base.cost.eval_integrator(X0, recombine(np.asarray(u)))[0])
        c = T(lambda u: cdist(np.asarray(u)))
        cj = T(lambda u: cdist_j(np.asarray(u)))
        if method == "SLSQP":
            cons = [
                {"type": "ineq", "fun": lambda u: c(u) - lo, "jac": cj},
                {"type": "ineq", "fun": lambda u: hi - c(u), "jac": lambda u: -cj(u)},
            ]
        else:
            cons = optimize.NonlinearConstraint(
                c, np.full(window, lo), np.full(window, hi), jac=cj
            )
        opts = {"maxiter": maxiter}
        if method != "SLSQP":
            opts["verbose"] = 0
        for x in (f, g, c, cj):
            x.t = x.n = 0
        roll(free0)  # warm rollout timer baseline
        roll.t = roll.n = 0
        t0 = time.perf_counter()
        soln = optimize.minimize(
            f,
            free0,
            jac=g,
            method=method,
            bounds=bounds(method),
            constraints=cons,
            options=opts,
        )
        wall = time.perf_counter() - t0
        rollout_est = (
            roll.t
        )  # standalone rollout cost ~ shared by objective & constraint
        comp = {
            "objective(+rollout+eig)": f.t,
            "gradient": g.t,
            "constraint": c.t,
            "constraint_jac": cj.t,
        }
        internal = wall - sum(comp.values())
        return wall, comp, internal, int(soln.get("nit", -1)), rollout_est

    combos = [
        ("trust-constr + min-eig  [OLD]", "min-eig", "trust-constr"),
        ("trust-constr + log-det", "log-det", "trust-constr"),
        ("SLSQP + min-eig", "min-eig", "SLSQP"),
        ("SLSQP + log-det        [NEW]", "log-det", "SLSQP"),
    ]

    print(
        f"# Old-vs-new transfer gate (quadrotor, order 5, NON-MUC config, maxiter={MAXITER})"
    )
    print(
        "# observability scored objective-agnostically as min-eig of accumulated sum_k W_k\n"
    )
    obs0, nobs0 = true_obs(free0)
    print(f"reference (un-optimized): acc-min-eig={obs0:.3e}  #obs={nobs0}/6\n")

    results = {}
    for label, obj, method in combos:
        soln, wall, trace = solve(obj, method)
        # score every recorded iterate
        curve = [(0.0, *true_obs(free0))]  # (wall, acc-min-eig, #obs) at iter 0
        for tw, x in trace:
            me, no = true_obs(x)
            curve.append((tw, me, no))
        nit = int(soln.get("nit", len(trace)))
        results[label] = {"wall": wall, "nit": nit, "curve": curve}

        def at_iter(k):
            i = min(k, len(curve) - 1)
            return curve[i][1], curve[i][2]

        me6, no6 = at_iter(EARLY)
        meF, noF = at_iter(MAXITER)
        per_iter = wall / max(nit, 1) * 1e3
        print(f"## {label}")
        print(f"   nit={nit:>3}  wall={wall:6.2f}s  per-iter={per_iter:6.1f} ms")
        print(
            f"   true acc-min-eig:  @2={at_iter(2)[0]:.3e}  @{EARLY}={me6:.3e}  "
            f"@{MAXITER}={meF:.3e}   (#obs @{EARLY}={no6}, @{MAXITER}={noF})\n"
        )

    # ---- component attribution: OLD@40, NEW@6, NEW@40 ----------------------------
    print("# component time attribution (ms, and % of solve)")
    for label, obj, method, mi in [
        ("OLD trust-constr+min-eig @40", "min-eig", "trust-constr", 40),
        ("NEW SLSQP+log-det @6", "log-det", "SLSQP", 6),
        ("NEW SLSQP+log-det @40", "log-det", "SLSQP", 40),
    ]:
        wall, comp, internal, nit, _ = attribute(obj, method, mi)
        print(f"## {label}: total {wall * 1e3:.0f} ms  (nit={nit})")
        for k, v in comp.items():
            print(f"     {k:<26}{v * 1e3:8.1f} ms  ({100 * v / wall:4.1f}%)")
        print(
            f"     {'solver-internal':<26}{internal * 1e3:8.1f} ms  ({100 * internal / wall:4.1f}%)\n"
        )

    # ---- decisive answers --------------------------------------------------------
    def curve_arr(label):
        c = np.array([(w, m) for (w, m, _) in results[label]["curve"]])
        return c

    new = "SLSQP + log-det        [NEW]"
    old = "trust-constr + min-eig  [OLD]"
    nc = results[new]["curve"]

    def obs_at(label, k):
        c = results[label]["curve"]
        return c[min(k, len(c) - 1)][1]

    best_new = max(m for _, m, _ in nc)
    print("# === did we just give up early? ===")
    print(
        f"NEW acc-min-eig: @{EARLY}={obs_at(new, EARLY):.3e}  @{MAXITER}={obs_at(new, MAXITER):.3e}  "
        f"=> @{EARLY} captures {100 * obs_at(new, EARLY) / max(best_new, 1e-30):.0f}% of NEW's best "
        f"(plateaued => early-stop is free, not giving up)"
    )
    print(
        f"OLD acc-min-eig: @{EARLY}={obs_at(old, EARLY):.3e}  @{MAXITER}={obs_at(old, MAXITER):.3e}  "
        f"(if flat/zero => OLD never climbs, at ANY budget)"
    )
    print(
        f"matched budget @{MAXITER}: NEW={obs_at(new, MAXITER):.3e}  vs  OLD={obs_at(old, MAXITER):.3e}"
        f"  => efficacy ratio {obs_at(new, MAXITER) / max(obs_at(old, MAXITER), 1e-30):.1e}x"
    )
    print(
        f"matched budget @{EARLY}:  NEW={obs_at(new, EARLY):.3e}  vs  OLD={obs_at(old, EARLY):.3e}"
    )
    print(
        f"per-iter: OLD={results[old]['wall'] / max(results[old]['nit'], 1) * 1e3:.0f} ms  "
        f"NEW={results[new]['wall'] / max(results[new]['nit'], 1) * 1e3:.0f} ms"
    )

    # ---- convergence plot --------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for label in results:
        c = np.array([(i, w, m) for i, (w, m, _) in enumerate(results[label]["curve"])])
        a1.plot(c[:, 0], c[:, 2], "-o", ms=3, label=label.strip())
        a2.plot(c[:, 1], c[:, 2], "-o", ms=3, label=label.strip())
    for ax in (a1, a2):
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.grid(alpha=0.3)
        ax.set_ylabel("true acc-min-eig (observability)")
        ax.legend(fontsize=8)
    a1.axvline(EARLY, color="r", ls=":", lw=1, label=f"early-stop={EARLY}")
    a1.set(xlabel="solver iteration", title="Observability vs iteration")
    a2.set(xlabel="wall time [s]", title="Observability vs wall time")
    fig.suptitle(
        "Old (trust-constr+min-eig) vs New (SLSQP+log-det): efficacy is not early-stopping"
    )
    fig.tight_layout()
    import pathlib

    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "old_vs_new.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

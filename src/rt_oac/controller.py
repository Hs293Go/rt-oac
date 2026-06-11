"""Real-time OAC controller: pluggable lean solver + warm-start + numpy boundary.

Phase 0 found the bottleneck is the optimizer, not the observability compute (6% of
solve time): scipy ``trust-constr`` spends ~94% in its own internals while ``SLSQP``
is ~20x leaner per iteration. It also found the paper's per-node min-eigenvalue
objective is numerically degenerate (the ``T^(2r*+1)`` floor) -- a smooth ``log-det``
objective, injected via ``ObservabilityCost(gramian_metric=...)``, restores a strong
gradient and drives full observability. This controller is built around those findings:

* solver is selectable (``SLSQP`` default, ``trust-constr`` available);
* the objective / gradient / constraint / constraint-Jacobian are jitted once with the
  free control vector, the held-constant leader inputs, and the operating state all as
  *runtime* arguments, so each receding-horizon solve reuses one compiled artifact;
* scipy-facing wrappers return numpy (clean boundary);
* the caller supplies the initial guess, so warm-starting (see :mod:`rt_oac.warmstart`)
  is a drop-in.

Only the follower inputs (indices 4..7) are decision variables; the leader inputs are
held constant from the reference.
"""

from collections.abc import Callable
import dataclasses
import inspect
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from observability_aware_control import observability_cost

FOLLOWER_INDICES = (4, 5, 6, 7)


@dataclasses.dataclass
class SolveResult:
    """Outcome of one receding-horizon solve."""

    u: np.ndarray  # (window, n_inputs) full recombined optimal control sequence
    objective: float
    nit: int
    nfev: int
    success: bool
    wall_time: float
    max_constraint_violation: float


class RTController:
    """Observability-aware controller with a selectable, lean solver."""

    def __init__(
        self,
        cost: observability_cost.ObservabilityCost,
        *,
        stlog_dt: float,
        lb: ArrayLike,
        ub: ArrayLike,
        n_inputs: int = 8,
        follower_indices: tuple[int, ...] = FOLLOWER_INDICES,
        method: str = "SLSQP",
        maxiter: int = 80,
        constraint: Callable | None = None,
        constraint_bounds: tuple[float, float] | None = None,
        constraint_mode: str = "hard",
        penalty_weight: float = 10.0,
    ):
        self._cost = cost
        self._dt = float(stlog_dt)
        self._n_inputs = n_inputs
        self._follower = jnp.asarray(follower_indices)
        self._leader = jnp.asarray([
            i for i in range(n_inputs) if i not in follower_indices
        ])
        self._method = method
        self._maxiter = maxiter
        self._constraint = constraint
        self._constraint_bounds = constraint_bounds
        # "hard": pass the constraint to the (SLSQP/trust-constr) solver. "soft": fold
        # it into the objective as a smooth one-sided penalty and solve box-bounded
        # unconstrained (L-BFGS-B) -- faster, fully JIT-able, same constraint reused.
        self._soft = constraint_mode == "soft"
        self._penalty_weight = float(penalty_weight)
        self._con_takes_cost = constraint is not None and (
            "cost" in inspect.signature(constraint).parameters
        )

        # free-variable bounds (follower columns only), flattened over the window
        self._lb = np.asarray(lb)
        self._ub = np.asarray(ub)

        # jit once; (free, u_const, x0) are runtime args -> no per-step recompile
        self._obj_jit = jax.jit(self._objective)
        self._vag_jit = jax.jit(jax.value_and_grad(self._objective))
        # Balanced costs (rt_oac.balanced_cost) expose .evaluate(x0, us, dt, p_ref, w).
        # Thread p_ref/weight as runtime args so a moving reference or a scheduled
        # weight varies per step without recompiling (Stage B dual control).
        self._vag_threaded = (
            jax.jit(jax.value_and_grad(self._objective_threaded))
            if hasattr(cost, "evaluate")
            else None
        )
        if constraint is not None:
            self._con_jit = jax.jit(self._constraint_value)
            self._conjac_jit = jax.jit(jax.jacobian(self._constraint_value))

    # ---- jitted core (operates on the free vector) --------------------------------
    def _recombine(self, free, u_const):
        window = u_const.shape[0]
        full = jnp.zeros((window, self._n_inputs))
        full = full.at[:, self._leader].set(u_const)
        return full.at[:, self._follower].set(free.reshape(window, -1))

    def _eval_constraint(self, u, x0):
        if self._con_takes_cost:
            return self._constraint(x0, u, cost=self._cost)
        return self._constraint(x0, u)

    def _penalty(self, u, x0):
        """Smooth one-sided penalty enforcing ``lo <= constraint <= hi``."""
        c = self._eval_constraint(u, x0)
        lo, hi = self._constraint_bounds
        viol = jnp.maximum(c - hi, 0.0) ** 2 + jnp.maximum(lo - c, 0.0) ** 2
        return self._penalty_weight * viol.sum()

    def _objective(self, free, u_const, x0):
        u = self._recombine(free, u_const)
        obj = self._cost(x0, u, self._dt).objective
        if self._soft and self._constraint is not None:
            obj += self._penalty(u, x0)
        return obj

    def _objective_threaded(self, free, u_const, x0, p_ref, weight):
        u = self._recombine(free, u_const)
        obj = self._cost.evaluate(x0, u, self._dt, p_ref, weight).objective
        if self._soft and self._constraint is not None:
            obj += self._penalty(u, x0)
        return obj

    def _constraint_value(self, free, u_const, x0):
        return self._eval_constraint(self._recombine(free, u_const), x0)

    # ---- solve --------------------------------------------------------------------
    def solve(
        self,
        x0: ArrayLike,
        u_guess: ArrayLike,
        *,
        p_ref: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
    ) -> SolveResult:
        """Solve the OAC problem for operating state ``x0`` seeded with ``u_guess``.

        ``u_guess`` is ``(window, n_inputs)``: its leader columns are held constant, its
        follower columns are the decision-variable initial guess (warm-start friendly).

        When the cost is a balanced cost (``rt_oac.balanced_cost``), pass ``p_ref`` (the
        per-window position reference) and ``weight`` (the observability trade-off,
        maybe covariance-scheduled). Both are threaded as runtime args, so the objective
        compiles once and the reference/weight vary per step.
        """
        x0 = jnp.asarray(x0)
        u_guess = jnp.asarray(u_guess)
        window = u_guess.shape[0]
        u_const = u_guess[:, self._leader]
        free0 = np.asarray(u_guess[:, self._follower].ravel())

        lb = np.broadcast_to(self._lb, (window, self._follower.shape[0])).ravel()
        ub = np.broadcast_to(self._ub, (window, self._follower.shape[0])).ravel()

        if p_ref is not None:
            if self._vag_threaded is None:
                raise ValueError(
                    "p_ref given but the cost has no .evaluate (not balanced)"
                )
            p_ref_j = jnp.asarray(p_ref)
            weight_j = jnp.asarray(weight, dtype=jnp.float64)

            def fun_and_jac(free):
                val, grad = self._vag_threaded(
                    jnp.asarray(free), u_const, x0, p_ref_j, weight_j
                )
                return float(val), np.asarray(grad)

        else:

            def fun_and_jac(free):
                val, grad = self._vag_jit(jnp.asarray(free), u_const, x0)
                return float(val), np.asarray(grad)

        cache: dict[str, Any] = {}

        def fun(free):
            key = free.tobytes()
            if cache.get("key") != key:
                cache["key"], cache["val"] = key, fun_and_jac(free)
            return cache["val"][0]

        def jac(free):
            key = free.tobytes()
            if cache.get("key") != key:
                cache["key"], cache["val"] = key, fun_and_jac(free)
            return cache["val"][1]

        t = time.perf_counter()
        if self._soft:
            # constraint folded into the objective -> box-bounded unconstrained solve
            soln = optimize.minimize(
                fun,
                free0,
                jac=jac,
                method="L-BFGS-B",
                bounds=optimize.Bounds(lb, ub),
                options={"maxiter": self._maxiter},
            )
        elif self._method == "SLSQP":
            soln = optimize.minimize(
                fun,
                free0,
                jac=jac,
                method="SLSQP",
                bounds=list(zip(lb, ub, strict=True)),
                constraints=self._build_constraints(u_const, x0, window),
                options={"maxiter": self._maxiter},
            )
        else:
            soln = optimize.minimize(
                fun,
                free0,
                jac=jac,
                method=self._method,
                bounds=optimize.Bounds(lb, ub),
                constraints=self._build_constraints(u_const, x0, window),
                options={"maxiter": self._maxiter, "verbose": 0},
            )
        wall = time.perf_counter() - t

        u_opt = np.asarray(self._recombine(jnp.asarray(soln.x), u_const))
        viol = self._violation(soln.x, u_const, x0)
        return SolveResult(
            u=u_opt,
            objective=float(soln.fun),
            nit=int(soln.get("nit", -1)),
            nfev=int(soln.get("nfev", -1)),
            success=bool(soln.success),
            wall_time=wall,
            max_constraint_violation=viol,
        )

    def _build_constraints(self, u_const, x0, window):
        if self._constraint is None:
            return ()
        lo, hi = self._constraint_bounds

        def cval(free):
            return np.asarray(self._con_jit(jnp.asarray(free), u_const, x0))

        def cjac(free):
            return np.asarray(self._conjac_jit(jnp.asarray(free), u_const, x0))

        if self._method == "SLSQP":
            return [
                {"type": "ineq", "fun": lambda f: cval(f) - lo, "jac": cjac},
                {
                    "type": "ineq",
                    "fun": lambda f: hi - cval(f),
                    "jac": lambda f: -cjac(f),
                },
            ]
        return optimize.NonlinearConstraint(
            cval, np.full(window, lo), np.full(window, hi), jac=cjac
        )

    def _violation(self, free, u_const, x0):
        if self._constraint is None:
            return 0.0
        lo, hi = self._constraint_bounds
        c = np.asarray(self._con_jit(jnp.asarray(free), u_const, x0))
        return float(np.maximum(0.0, np.maximum(lo - c, c - hi)).max())

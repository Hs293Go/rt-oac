"""Builds the baseline quadrotor OAC problem, matching the companion experiment.

This is the single source of truth for the scenario so the profiler, experiments, and
ablations all solve *identical* problems. It mirrors the construction in
``observability_aware_control/examples/quadrotor_cooperative_navigation.py`` (Euler
integration, ``var = [range_var, *att_var]``, the ``interrobot_distance_squared``
distance constraint, the tiled input bounds, ``x0 = [0,-1,1, 0,0,0,1, 0,0,0]``), while
exposing the knobs the real-time study varies (STLOG order, tangent-space manifold
Gramian, surrogate objective metric).

``import rt_oac`` (done transitively here) puts the companion source on ``sys.path`` and
enables JAX x64.
"""

from collections.abc import Callable
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from example_lib.models import inter_quadrotor_pose as mdl
from observability_aware_control import (
    integrator,
    observability_aware_controller as oac_ctrl,
    observability_cost,
)
import rt_oac  # noqa: F401  (bootstraps companion src + x64)
from rt_oac.controller import RTController

REPO_ROOT = Path(__file__).resolve().parents[2]
# The scenario config lives in the Hydra config tree (a `scenario` group): the quadrotor
# example composes it via `defaults: [scenario: quadrotor]`, and standalone callers
# (experiments/benchmarks) load it here by default. One config framework, no TOML.
SCENARIO_CONF = REPO_ROOT / "examples" / "conf" / "scenario" / "quadrotor.yaml"
DATA_DIR = REPO_ROOT / "data"

U_EQM = np.array([9.81, 0.0, 0.0, 0.0])
"""Hover-equilibrium input for one quadrotor (thrust = g, zero body rates)."""

N_ROBOTS = 2
FOLLOWER_INDICES = (4, 5, 6, 7)
"""Indices of the follower's inputs in the stacked 8-vector (the decision variables)."""


def load_config(path: str | Path = SCENARIO_CONF) -> DictConfig:
    """Load a scenario config (the Hydra/OmegaConf ``scenario`` group YAML)."""
    return OmegaConf.load(Path(path))


def load_leader_trajectory(
    path: str | Path = DATA_DIR / "leader_trajectory.npz",
) -> dict[str, np.ndarray]:
    """Load the cached leader trajectory (states + the leader's 4 inputs over time)."""
    with np.load(Path(path)) as data:
        return {k: data[k] for k in data.files}


@dataclasses.dataclass
class Scenario:
    """A fully-constructed OAC problem ready to solve in a receding-horizon loop."""

    cfg: DictConfig
    cost: observability_cost.ObservabilityCost
    controller: oac_ctrl.ObservabilityAwareController
    leader_dt: float
    x_leader: np.ndarray  # (T, 10)
    u_leader: np.ndarray  # (T, 4)
    window: int
    stlog_dt: float
    x0: np.ndarray  # (10,) relative initial state
    follower_indices: tuple[int, ...] = FOLLOWER_INDICES

    def reference_guess(self, step: int) -> np.ndarray:
        """The baseline (un-warm-started) initial guess at a receding-horizon step.

        Stacks the leader's reference inputs over the window with the follower held at
        hover equilibrium -- exactly what the companion example seeds ``minimize`` with.
        Returns an ``(window, 8)`` array.
        """
        lead = self.u_leader[step : step + self.window]
        if lead.shape[0] < self.window:  # pad the tail with the last leader input
            pad = np.repeat(lead[-1:], self.window - lead.shape[0], axis=0)
            lead = np.concatenate([lead, pad], axis=0)
        follower = np.tile(U_EQM, (self.window, N_ROBOTS - 1))
        return np.hstack([lead, follower])

    def snapshots(
        self, n: int = 16, *, seed: int = 0, states: np.ndarray | None = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Representative ``(x_op, u0)`` problems for micro-benchmarking.

        Samples operating states encountered along the closed-loop run (from the
        companion ``optimization_results_stlog.npz`` if available, else along the leader
        trajectory) paired with the reference initial guess at the same step, so the
        profiler exercises realistic solves.
        """
        rng = np.random.default_rng(seed)
        horizon = len(self.u_leader) - self.window
        idx = np.sort(rng.choice(max(horizon, 1), size=min(n, horizon), replace=False))
        out = []
        for i in idx:
            x_op = states[i] if states is not None else self.x0
            out.append((np.asarray(x_op), self.reference_guess(int(i))))
        return out


def build_rt_controller(
    scenario: Scenario,
    *,
    method: str = "SLSQP",
    maxiter: int = 80,
):
    """Construct an :class:`rt_oac.controller.RTController` for a built scenario.

    Reuses the scenario's cost (whatever ``gramian_metric`` it was built with) and the
    follower input bounds + inter-vehicle distance constraint from the config.
    """
    cfg = scenario.cfg
    return RTController(
        scenario.cost,
        stlog_dt=scenario.stlog_dt,
        lb=np.asarray(cfg["optim"]["lb"]),
        ub=np.asarray(cfg["optim"]["ub"]),
        n_inputs=2 * len(cfg["optim"]["lb"]),
        follower_indices=scenario.follower_indices,
        method=method,
        maxiter=maxiter,
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=(
            cfg["opc"]["min_inter_vehicle_distance"] ** 2,
            cfg["opc"]["max_inter_vehicle_distance"] ** 2,
        ),
    )


def build_scenario(
    config: DictConfig | dict[str, Any] | None = None,
    *,
    order: int | None = None,
    use_manifold: bool = False,
    gramian_metric: Callable[[Any], Any] | None = None,
    maxiter: int | None = None,
    integration: str = "euler",
    observed_indices: tuple[int, ...] | None = None,
) -> Scenario:
    """Construct the quadrotor OAC problem.

    Parameters
    ----------
    config
        The scenario config (an OmegaConf/Hydra ``scenario``-group node, as the example
        passes ``cfg.scenario``, or a plain dict). When ``None`` the default
        ``examples/conf/scenario/quadrotor.yaml`` is loaded (standalone callers need no
        Hydra context).
    order
        Override the STLOG order (baseline 5). Lower orders are a Phase-1 ablation.
    use_manifold
        If True, build the tangent-space (9x9) Gramian via ``inter_quadrotor_pose``'s
        product manifold, removing the quaternion gauge analytically (Phase 1d).
    gramian_metric
        Surrogate objective metric injected into ``ObservabilityCost`` (Phase 2). When
        ``None``, the companion's reciprocal-min-singular-value metric is used.
    maxiter
        Override the optimizer iteration cap (baseline 40).
    integration
        ``"euler"`` (baseline) or ``"rk4"``.
    """
    cfg = OmegaConf.create(config) if config is not None else load_config()
    traj = load_leader_trajectory()
    leader_dt = float(traj["dt"])
    x_leader, u_leader = traj["x_leader"], traj["u_leader"]

    var = np.concatenate([
        np.array([cfg["noise"]["range_var"]]),
        np.full(4, cfg["noise"]["att_var"]),
    ])
    gramian_kw: dict[str, Any] = {
        "order": order if order is not None else cfg["stlog"]["order"],
        "var": var,
    }
    if use_manifold:
        gramian_kw["manifold"] = mdl.MANIFOLD

    method = {"euler": integrator.Methods.EULER, "rk4": integrator.Methods.RK4}[
        integration
    ]
    obs_idx = (
        observed_indices
        if observed_indices is not None
        else tuple(cfg["opc"].get("observed_components", ()))
    )
    cost_kw: dict[str, Any] = {
        "gramian_kw": gramian_kw,
        "integration_method": method,
        "observed_indices": obs_idx,
    }
    if gramian_metric is not None:
        cost_kw["gramian_metric"] = gramian_metric

    cost = observability_cost.ObservabilityCost(
        mdl.dynamics, mdl.observation, leader_dt, **cost_kw
    )

    window = cfg["opc"]["window_size"]
    u_lb = np.tile(np.array(cfg["optim"]["lb"]), (window, N_ROBOTS))
    u_ub = np.tile(np.array(cfg["optim"]["ub"]), (window, N_ROBOTS))

    options = OmegaConf.to_container(cfg["optim"]["options"], resolve=True)
    if maxiter is not None:
        options["maxiter"] = maxiter

    controller = oac_ctrl.ObservabilityAwareController(
        cost,
        lb=u_lb,
        ub=u_ub,
        method=cfg["optim"]["method"],
        optim_options=options,
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=(
            cfg["opc"]["min_inter_vehicle_distance"] ** 2,
            cfg["opc"]["max_inter_vehicle_distance"] ** 2,
        ),
    )

    return Scenario(
        cfg=cfg,
        cost=cost,
        controller=controller,
        leader_dt=leader_dt,
        x_leader=x_leader,
        u_leader=u_leader,
        window=window,
        stlog_dt=float(cfg["stlog"]["dt"]),
        x0=np.asarray(cfg["sim"]["init_state"]),
    )

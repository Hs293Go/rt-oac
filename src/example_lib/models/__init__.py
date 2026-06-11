import importlib
import pkgutil
from typing import NamedTuple

from observability_aware_control.typing import DynamicsFunction, ObservationFunction


class Model(NamedTuple):
    """A dynamical model with dynamics and observation functions.

    ``manifold`` optionally describes the product-manifold structure of the
    state (see ``example_lib.manifold``); when set, integrators and observability
    Gramians can advance attitude by the exponential map and build tangent-space
    Gramians instead of treating the state chart as flat. ``None`` means a purely
    Euclidean state.
    """

    dynamics: DynamicsFunction
    observation: ObservationFunction
    manifold: object | None = None


MODELS: dict[str, Model] = {}

for _, module_name, _ in pkgutil.iter_modules(__path__):
    _ = importlib.import_module(f"{__name__}.{module_name}")

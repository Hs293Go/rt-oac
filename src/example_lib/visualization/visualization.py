import abc
import collections
from typing import override

import numpy as np
from numpy.typing import ArrayLike
import rerun as rr


class VisualizationElementBase(abc.ABC):
    """Base class for visualization elements."""

    def __init__(self, prefix: str) -> None:
        if prefix and not prefix.startswith("/"):
            self._entity_path = f"/{prefix}/{self.element_name}"
        else:
            self._entity_path = f"{prefix}/{self.element_name}"

    def log(self, entity: rr.AsComponents) -> None:
        """Log a rerun component to this element's key."""
        rr.log(self._entity_path, entity)

    @property
    @abc.abstractmethod
    def element_name(self) -> str:
        """Name of the visualization element, used in the rerun key."""


class PoseReferenceFrame(VisualizationElementBase):
    """Helper to log a reference frame at a given pose."""

    def __init__(self, prefix: str = "") -> None:
        super().__init__(prefix)

        self.log(rr.Arrows3D(origins=np.zeros(3), vectors=np.eye(3), colors=np.eye(3)))

    @property
    @override
    def element_name(self) -> str:
        return "pose"

    def set_pose(
        self, position: ArrayLike, rotation: rr.Quaternion | rr.RotationAxisAngle
    ) -> None:
        """Set the pose of the rigid body.

        Args
        ----
        position: (3,) array-like, XYZ position.
        rotation: Quaternion or RotationAxisAngle representing orientation.
        """
        position = np.copy(position)
        self.log(rr.Transform3D(translation=position, rotation=rotation))


class PositionTrace(VisualizationElementBase):
    """Keeps a short trace of recent positions."""

    def __init__(self, prefix: str = "", max_length: int = 50) -> None:
        super().__init__(prefix)
        self._trace = collections.deque(maxlen=max_length)

    @property
    @override
    def element_name(self) -> str:
        return "trace"

    def add_position(self, position: ArrayLike) -> None:
        """Append a new position to the trace.

        Args
        ----
        position: (3,) array-like, XYZ position.
        """
        self._trace.append(np.asarray(position, copy=True))
        self.log(rr.LineStrips3D(np.array(self._trace)[None]))

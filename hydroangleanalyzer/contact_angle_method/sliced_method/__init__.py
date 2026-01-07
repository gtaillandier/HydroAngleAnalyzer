"""Public exports for sliced contact angle method."""

from .angle_fitting_sliced import (
    ContactAngle_sliced as _ContactAngle_sliced,
)
from .angle_fitting_sliced import (
    ContactAngleSliced as _ContactAngleSliced,
)
from .multi_processing import (
    ContactAngle_sliced_parallel as _ContactAngle_sliced_parallel,
)
from .surface_defined import SurfaceDefinition as _SurfaceDefinition

__all__ = [
    "ContactAngle_sliced",
    "ContactAngleSliced",
    "ContactAngle_sliced_parallel",
    "SurfaceDefinition",
]

ContactAngle_sliced = _ContactAngle_sliced
ContactAngleSliced = _ContactAngleSliced
ContactAngle_sliced_parallel = _ContactAngle_sliced_parallel
SurfaceDefinition = _SurfaceDefinition

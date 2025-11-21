"""Public exports for sliced contact angle method."""

from .angle_fitting_sliced import ContactAngle_sliced as _ContactAngle_sliced
from .multi_processing import (
    ContactAngle_sliced_parallel as _ContactAngle_sliced_parallel,
)
from .surface_defined import SurfaceDefinition as _SurfaceDefinition

__all__ = ["ContactAngle_sliced", "ContactAngle_sliced_parallel", "SurfaceDefinition"]

ContactAngle_sliced = _ContactAngle_sliced
ContactAngle_sliced_parallel = _ContactAngle_sliced_parallel
SurfaceDefinition = _SurfaceDefinition

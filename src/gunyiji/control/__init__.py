"""Control modules."""

from gunyiji.control.control_allocation import (
    ACTUATOR_COUNT,
    DEFAULT_ROLLING_WING_PARAMS,
    ActuatorState,
    AllocationConfig,
    AllocationResult,
    FirstOrderActuatorModel,
    NonlinearControlAllocator,
    RollingWingParams,
    hover_actuator_state,
    virtual_input_from_actuators,
)
from gunyiji.control.mpc import (
    CasadiNMPC,
    CircleTrajectory,
    HelicalTrajectory,
    NMPCConfig,
    NMPCResult,
    PathNMPCConfig,
)

__all__ = [
    "ACTUATOR_COUNT",
    "DEFAULT_ROLLING_WING_PARAMS",
    "ActuatorState",
    "AllocationConfig",
    "AllocationResult",
    "CasadiNMPC",
    "CircleTrajectory",
    "FirstOrderActuatorModel",
    "HelicalTrajectory",
    "NMPCConfig",
    "NMPCResult",
    "NonlinearControlAllocator",
    "PathNMPCConfig",
    "RollingWingParams",
    "hover_actuator_state",
    "virtual_input_from_actuators",
]

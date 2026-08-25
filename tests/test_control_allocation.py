import numpy as np

from gunyiji.control import (
    DEFAULT_ROLLING_WING_PARAMS,
    ActuatorState,
    FirstOrderActuatorModel,
    NonlinearControlAllocator,
    hover_actuator_state,
    virtual_input_from_actuators,
)
from gunyiji.dynamics import DEFAULT_VEHICLE_PARAMS


def test_default_hover_actuator_state_matches_virtual_hover_input() -> None:
    """验证暂定滚翼参数能够在允许转速范围内准确产生悬停虚拟输入。"""
    state = hover_actuator_state()
    achieved = virtual_input_from_actuators(state.omega, state.beta)

    assert np.all(state.omega < DEFAULT_ROLLING_WING_PARAMS.omega_max)
    assert np.allclose(
        achieved,
        DEFAULT_VEHICLE_PARAMS.hover_input,
        atol=1.0e-10,
    )


def test_nonlinear_allocator_tracks_representative_virtual_input() -> None:
    """验证非线性分配器能够跟踪一组具有力和三轴力矩的代表性指令。"""
    allocator = NonlinearControlAllocator()
    desired = np.array([3.0, -19.0, 0.12, -0.10, 0.06])
    result = allocator.allocate(desired, dt=0.1)

    assert result.success, result.status
    assert np.max(np.abs(result.residual)) < 2.0e-3
    assert np.all(result.omega >= DEFAULT_ROLLING_WING_PARAMS.omega_min)
    assert np.all(result.omega <= DEFAULT_ROLLING_WING_PARAMS.omega_max)
    assert np.all(result.beta >= DEFAULT_ROLLING_WING_PARAMS.beta_min)
    assert np.all(result.beta <= DEFAULT_ROLLING_WING_PARAMS.beta_max)


def test_first_order_actuator_is_lagged_and_rate_limited() -> None:
    """验证一阶执行机构不会瞬间到达指令，并遵守最大转速和角速度限制。"""
    params = DEFAULT_ROLLING_WING_PARAMS
    initial = hover_actuator_state()
    model = FirstOrderActuatorModel(initial_state=initial)
    command = ActuatorState(
        omega=np.full(4, params.omega_max),
        beta=np.full(4, params.beta_min),
    )
    dt = 0.01
    updated = model.step(command, dt)

    assert np.all(updated.omega > initial.omega)
    assert np.all(updated.omega < command.omega)
    assert np.all(updated.beta < initial.beta)
    assert np.all(updated.beta > command.beta)
    assert np.all(
        updated.omega - initial.omega <= params.omega_rate_max * dt + 1.0e-12
    )
    assert np.all(
        initial.beta - updated.beta <= params.beta_rate_max * dt + 1.0e-12
    )

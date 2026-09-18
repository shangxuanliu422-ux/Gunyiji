
"""四滚翼非线性控制分配与一阶执行机构模型。"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi
from typing import Sequence

import casadi as ca
import numpy as np

from gunyiji.dynamics.vehicle_params import DEFAULT_VEHICLE_PARAMS, VehicleParams

ACTUATOR_COUNT = 4
VIRTUAL_INPUT_SIZE = 5


@dataclass(frozen=True)
class RollingWingParams:
    """
    保存四滚翼控制分配和执行机构模型使用的暂定参数。

    坐标系采用机体系 ``x`` 向前、``y`` 向右、``z`` 向下。四个安装点按俯视
    顺时针编号：1 前左、2 前右、3 后右、4 后左。偏转角 ``beta=0`` 时推力
    沿 ``+x``，正偏转按右手定则绕 ``+y`` 旋转，``beta=90 deg`` 时推力沿
    ``-z``，即竖直向上。沿 ``+z`` 方向俯视机体时，旋转方向 ``+1`` 表示
    逆时针，``-1`` 表示顺时针，四个滚翼暂按逆、顺、逆、顺交替布置。

    这些参数用于算法验证，不代表最终气动设计结果。推力系数、安装位置、角度
    范围和时间常数后续应由 CFD、台架试验及执行机构辨识结果替换。
    """

    thrust_coefficient: float = 1.4e-5
    mount_positions: tuple[tuple[float, float, float], ...] = (
        (0.22, -0.18, 0.0),  # 1 号滚翼：前左
        (0.22, 0.18, 0.0),  # 2 号滚翼：前右
        (-0.22, 0.18, 0.0),  # 3 号滚翼：后右
        (-0.22, -0.18, 0.0),  # 4 号滚翼：后左
    )
    spin_directions: tuple[int, int, int, int] = (1, -1, 1, -1)

    omega_min: float = 0.0
    omega_max: float = 900.0
    omega_rate_max: float = 10000.0
    beta_min: float = 25.0 * pi / 180.0
    beta_max: float = 155.0 * pi / 180.0
    beta_rate_max: float = 15.0

    motor_time_constant: float = 0.025
    tilt_time_constant: float = 0.035

    def __post_init__(self) -> None:
        """检查滚翼几何、气动系数和执行机构限制是否合法。"""
        positions = np.asarray(self.mount_positions, dtype=float)
        if positions.shape != (ACTUATOR_COUNT, 3):
            raise ValueError("mount_positions must have shape (4, 3)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("mount_positions must contain finite values")
        if len(self.spin_directions) != ACTUATOR_COUNT:
            raise ValueError("spin_directions must contain four entries")
        if any(direction not in (-1, 1) for direction in self.spin_directions):
            raise ValueError("spin_directions entries must be -1 or 1")
        if self.thrust_coefficient <= 0.0:
            raise ValueError("thrust_coefficient must be positive")
        if self.omega_min < 0.0 or self.omega_min >= self.omega_max:
            raise ValueError("omega bounds must satisfy 0 <= omega_min < omega_max")
        if self.omega_rate_max <= 0.0:
            raise ValueError("omega_rate_max must be positive")
        if not 0.0 <= self.beta_min < self.beta_max <= pi:
            raise ValueError("beta bounds must satisfy 0 <= beta_min < beta_max <= pi")
        if self.beta_rate_max <= 0.0:
            raise ValueError("beta_rate_max must be positive")
        if self.motor_time_constant <= 0.0 or self.tilt_time_constant <= 0.0:
            raise ValueError("actuator time constants must be positive")

    @property
    def positions(self) -> np.ndarray:
        """返回形状为 ``(4, 3)`` 的滚翼安装位置数组，单位为米。"""
        return np.asarray(self.mount_positions, dtype=float)

    @property
    def neutral_beta(self) -> float:
        """返回竖直向上推力对应的中立偏转角，即 90 度。"""
        return 0.5 * pi

    def hover_omega(self, vehicle: VehicleParams = DEFAULT_VEHICLE_PARAMS) -> float:
        """根据质量和推力系数计算四个滚翼等推力悬停时的转速。"""
        omega = np.sqrt(
            vehicle.mass
            * vehicle.gravity
            / (ACTUATOR_COUNT * self.thrust_coefficient)
        )
        if omega > self.omega_max:
            raise ValueError("hover speed exceeds omega_max")
        return float(max(omega, self.omega_min))


DEFAULT_ROLLING_WING_PARAMS = RollingWingParams()


@dataclass(frozen=True)
class AllocationConfig:
    """保存非线性控制分配的误差权重、平滑权重和 IPOPT 设置。"""

    virtual_input_weights: tuple[float, float, float, float, float] = (
        1.0 / 15.0,
        1.0 / 45.0,
        1.0 / 1.2,
        1.0 / 1.2,
        1.0 / 0.8,
    )
    omega_change_weight: float = 2.0e-4
    beta_change_weight: float = 5.0e-4
    omega_balance_weight: float = 2.0e-5
    enforce_command_rate_limits: bool = True
    ipopt_max_iter: int = 100
    ipopt_tol: float = 1.0e-7

    def __post_init__(self) -> None:
        """检查控制分配权重和求解器参数。"""
        weights = np.asarray(self.virtual_input_weights, dtype=float)
        if weights.shape != (VIRTUAL_INPUT_SIZE,) or np.any(weights <= 0.0):
            raise ValueError("virtual_input_weights must contain five positive values")
        if (
            self.omega_change_weight < 0.0
            or self.beta_change_weight < 0.0
            or self.omega_balance_weight < 0.0
        ):
            raise ValueError("allocation regularization weights must be non-negative")
        if self.ipopt_max_iter <= 0:
            raise ValueError("ipopt_max_iter must be positive")
        if self.ipopt_tol <= 0.0:
            raise ValueError("ipopt_tol must be positive")


@dataclass(frozen=True)
class ActuatorState:
    """保存四个滚翼当前实际转速和实际偏转角。"""

    omega: np.ndarray
    beta: np.ndarray

    def __post_init__(self) -> None:
        """将执行机构状态规范为两个长度为 4 的浮点数组。"""
        omega = _actuator_vector("omega", self.omega)
        beta = _actuator_vector("beta", self.beta)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "beta", beta)


@dataclass(frozen=True)
class AllocationResult:
    """保存一次非线性控制分配得到的指令、实际映射值和求解状态。"""

    omega: np.ndarray
    beta: np.ndarray
    desired_virtual_input: np.ndarray
    achieved_virtual_input: np.ndarray
    residual: np.ndarray
    objective: float
    status: str
    success: bool


def hover_actuator_state(
    params: RollingWingParams = DEFAULT_ROLLING_WING_PARAMS,
    vehicle: VehicleParams = DEFAULT_VEHICLE_PARAMS,
) -> ActuatorState:
    """返回四个滚翼等转速、偏转角为 90 度的悬停执行机构状态。"""
    return ActuatorState(
        omega=np.full(ACTUATOR_COUNT, params.hover_omega(vehicle)),
        beta=np.full(ACTUATOR_COUNT, params.neutral_beta),
    )


def virtual_input_from_actuators(
    omega: Sequence[float],
    beta: Sequence[float],
    params: RollingWingParams = DEFAULT_ROLLING_WING_PARAMS,
) -> np.ndarray:
    """
    把四个滚翼的转速和偏转角映射为 ``[fx, fz, tau_x, tau_y, tau_z]``。

    当前模型只包含推力及推力力臂产生的力矩，暂不包含反扭矩、陀螺力矩、
    翼间气动干扰和机体气动力。
    """
    omega_array = _actuator_vector("omega", omega)
    beta_array = _actuator_vector("beta", beta)
    thrust = params.thrust_coefficient * omega_array**2
    force_x = thrust * np.cos(beta_array)
    upward_force = thrust * np.sin(beta_array)
    positions = params.positions

    return np.array(
        [
            np.sum(force_x),
            -np.sum(upward_force),
            np.sum(-positions[:, 1] * upward_force),
            np.sum(positions[:, 0] * upward_force),
            np.sum(-positions[:, 1] * force_x),
        ],
        dtype=float,
    )


class NonlinearControlAllocator:
    """
    使用 CasADi/IPOPT 将五维虚拟控制量分配为四组滚翼转速和偏转角。

    输入为期望 ``[fx, fz, tau_x, tau_y, tau_z]``、上一拍执行器指令和控制
    周期。输出为满足幅值与变化率约束的 ``omega``、``beta``，以及重新映射
    后真正能够产生的虚拟控制量。
    """

    def __init__(
        self,
        params: RollingWingParams = DEFAULT_ROLLING_WING_PARAMS,
        vehicle: VehicleParams = DEFAULT_VEHICLE_PARAMS,
        config: AllocationConfig | None = None,
    ) -> None:
        """保存参数、建立非线性规划求解器，并以悬停状态初始化热启动值。"""
        self.params = params
        self.vehicle = vehicle
        self.config = config or AllocationConfig()
        self._last_command = hover_actuator_state(params, vehicle)
        self._build_solver()

    @property
    def last_command(self) -> ActuatorState:
        """返回上一拍成功分配的执行器指令副本。"""
        return ActuatorState(
            omega=self._last_command.omega.copy(),
            beta=self._last_command.beta.copy(),
        )

    def reset(self, state: ActuatorState | None = None) -> None:
        """将分配器热启动值重置为指定状态；未指定时恢复到悬停状态。"""
        self._last_command = state or hover_actuator_state(self.params, self.vehicle)

    def allocate(
        self,
        desired_virtual_input: Sequence[float],
        dt: float,
        previous_command: ActuatorState | None = None,
    ) -> AllocationResult:
        """
        求解一次带幅值和变化率约束的非线性控制分配问题。

        当求解失败时返回上一拍指令，并将 ``success`` 标记为 ``False``，便于
        上层控制器采用保持上一拍或切换备用控制器的策略。
        """
        desired = _virtual_input_vector(desired_virtual_input)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        previous = self._bounded_command(previous_command or self._last_command)
        lower, upper = self._decision_bounds(previous, dt)
        initial_guess = np.concatenate(
            (
                np.clip(previous.omega, lower[:ACTUATOR_COUNT], upper[:ACTUATOR_COUNT]),
                np.clip(previous.beta, lower[ACTUATOR_COUNT:], upper[ACTUATOR_COUNT:]),
            )
        )
        parameter = np.concatenate((desired, previous.omega, previous.beta))

        try:
            solution = self._solver(
                x0=initial_guess,
                lbx=lower,
                ubx=upper,
                lbg=[],
                ubg=[],
                p=parameter,
            )
            decision = np.asarray(solution["x"], dtype=float).reshape(-1)
            omega = decision[:ACTUATOR_COUNT]
            beta = decision[ACTUATOR_COUNT:]
            stats = self._solver.stats()
            success = bool(stats.get("success", False))
            status = str(stats.get("return_status", "unknown"))
            objective = float(solution["f"])
        except RuntimeError as error:
            omega = previous.omega.copy()
            beta = previous.beta.copy()
            success = False
            status = f"solver exception: {error}"
            objective = float("inf")

        achieved = virtual_input_from_actuators(omega, beta, self.params)
        result = AllocationResult(
            omega=np.asarray(omega, dtype=float),
            beta=np.asarray(beta, dtype=float),
            desired_virtual_input=desired,
            achieved_virtual_input=achieved,
            residual=achieved - desired,
            objective=objective,
            status=status,
            success=success,
        )
        if success:
            self._last_command = ActuatorState(omega=result.omega, beta=result.beta)
        return result

    def _build_solver(self) -> None:
        """建立滚翼静态映射、跟踪误差和平滑正则项组成的 NLP。"""
        omega = ca.MX.sym("omega", ACTUATOR_COUNT)
        beta = ca.MX.sym("beta", ACTUATOR_COUNT)
        desired = ca.MX.sym("desired", VIRTUAL_INPUT_SIZE)
        previous_omega = ca.MX.sym("previous_omega", ACTUATOR_COUNT)
        previous_beta = ca.MX.sym("previous_beta", ACTUATOR_COUNT)

        achieved = self._virtual_input_expr(omega, beta)
        weights = ca.DM(self.config.virtual_input_weights)
        scaled_error = weights * (achieved - desired)

        omega_scale = self.params.omega_max - self.params.omega_min
        beta_scale = self.params.beta_max - self.params.beta_min
        omega_mean = ca.sum1(omega) / ACTUATOR_COUNT
        objective = ca.dot(scaled_error, scaled_error)
        objective += self.config.omega_change_weight * ca.sumsqr(
            (omega - previous_omega) / omega_scale
        )
        objective += self.config.beta_change_weight * ca.sumsqr(
            (beta - previous_beta) / beta_scale
        )
        objective += self.config.omega_balance_weight * ca.sumsqr(
            (omega - omega_mean) / omega_scale
        )

        decision = ca.vertcat(omega, beta)
        parameter = ca.vertcat(desired, previous_omega, previous_beta)
        problem = {
            "x": decision,
            "p": parameter,
            "f": objective,
            "g": ca.MX.zeros(0, 1),
        }
        options = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.config.ipopt_max_iter,
            "ipopt.tol": self.config.ipopt_tol,
        }
        self._solver = ca.nlpsol(
            f"rolling_wing_allocator_{id(self)}",
            "ipopt",
            problem,
            options,
        )

    def _virtual_input_expr(self, omega: ca.MX, beta: ca.MX) -> ca.MX:
        """构造供 CasADi 自动求导使用的五维滚翼静态映射表达式。"""
        thrust = self.params.thrust_coefficient * ca.power(omega, 2)
        force_x = thrust * ca.cos(beta)
        upward_force = thrust * ca.sin(beta)
        positions = self.params.positions
        x_arm = ca.DM(positions[:, 0])
        y_arm = ca.DM(positions[:, 1])
        return ca.vertcat(
            ca.sum1(force_x),
            -ca.sum1(upward_force),
            ca.dot(-y_arm, upward_force),
            ca.dot(x_arm, upward_force),
            ca.dot(-y_arm, force_x),
        )

    def _decision_bounds(
        self,
        previous: ActuatorState,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """根据物理幅值限制和上一拍指令计算当前 NLP 的变量上下界。"""
        params = self.params
        omega_lower = np.full(ACTUATOR_COUNT, params.omega_min)
        omega_upper = np.full(ACTUATOR_COUNT, params.omega_max)
        beta_lower = np.full(ACTUATOR_COUNT, params.beta_min)
        beta_upper = np.full(ACTUATOR_COUNT, params.beta_max)

        if self.config.enforce_command_rate_limits:
            omega_delta = params.omega_rate_max * dt
            beta_delta = params.beta_rate_max * dt
            omega_lower = np.maximum(omega_lower, previous.omega - omega_delta)
            omega_upper = np.minimum(omega_upper, previous.omega + omega_delta)
            beta_lower = np.maximum(beta_lower, previous.beta - beta_delta)
            beta_upper = np.minimum(beta_upper, previous.beta + beta_delta)

        return (
            np.concatenate((omega_lower, beta_lower)),
            np.concatenate((omega_upper, beta_upper)),
        )

    def _bounded_command(self, command: ActuatorState) -> ActuatorState:
        """将外部传入的上一拍指令限制到滚翼执行器物理范围内。"""
        return ActuatorState(
            omega=np.clip(
                command.omega,
                self.params.omega_min,
                self.params.omega_max,
            ),
            beta=np.clip(
                command.beta,
                self.params.beta_min,
                self.params.beta_max,
            ),
        )


class FirstOrderActuatorModel:
    """
    模拟四个滚翼电机和偏转机构的一阶惯性、幅值限制及最大变化率。

    连续模型分别为 ``dOmega/dt=(Omega_cmd-Omega)/tau_motor`` 和
    ``dBeta/dt=(Beta_cmd-Beta)/tau_tilt``。``step`` 使用一阶系统精确离散
    系数更新，并额外限制每个时间步允许的最大转速和角度变化。
    """

    def __init__(
        self,
        params: RollingWingParams = DEFAULT_ROLLING_WING_PARAMS,
        vehicle: VehicleParams = DEFAULT_VEHICLE_PARAMS,
        initial_state: ActuatorState | None = None,
    ) -> None:
        """保存参数，并将实际执行机构状态初始化为指定值或悬停值。"""
        self.params = params
        self.vehicle = vehicle
        self._state = initial_state or hover_actuator_state(params, vehicle)
        self._state = self._bounded_state(self._state)

    @property
    def state(self) -> ActuatorState:
        """返回当前实际执行机构状态的副本。"""
        return ActuatorState(
            omega=self._state.omega.copy(),
            beta=self._state.beta.copy(),
        )

    @property
    def virtual_input(self) -> np.ndarray:
        """返回当前实际执行机构状态产生的五维虚拟控制量。"""
        return virtual_input_from_actuators(
            self._state.omega,
            self._state.beta,
            self.params,
        )

    def reset(self, state: ActuatorState | None = None) -> ActuatorState:
        """把实际执行机构重置为指定状态；未指定时恢复悬停状态。"""
        self._state = self._bounded_state(
            state or hover_actuator_state(self.params, self.vehicle)
        )
        return self.state

    def derivative(self, command: ActuatorState) -> ActuatorState:
        """计算当前状态下受速率限制的一阶执行机构状态导数。"""
        target = self._bounded_state(command)
        omega_dot = np.clip(
            (target.omega - self._state.omega) / self.params.motor_time_constant,
            -self.params.omega_rate_max,
            self.params.omega_rate_max,
        )
        beta_dot = np.clip(
            (target.beta - self._state.beta) / self.params.tilt_time_constant,
            -self.params.beta_rate_max,
            self.params.beta_rate_max,
        )
        return ActuatorState(omega=omega_dot, beta=beta_dot)

    def step(self, command: ActuatorState, dt: float) -> ActuatorState:
        """
        将一阶执行机构推进一个时间步，并返回更新后的实际转速和偏转角。
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        target = self._bounded_state(command)
        omega_alpha = 1.0 - exp(-dt / self.params.motor_time_constant)
        beta_alpha = 1.0 - exp(-dt / self.params.tilt_time_constant)
        omega_delta = omega_alpha * (target.omega - self._state.omega)
        beta_delta = beta_alpha * (target.beta - self._state.beta)
        omega_delta = np.clip(
            omega_delta,
            -self.params.omega_rate_max * dt,
            self.params.omega_rate_max * dt,
        )
        beta_delta = np.clip(
            beta_delta,
            -self.params.beta_rate_max * dt,
            self.params.beta_rate_max * dt,
        )
        self._state = self._bounded_state(
            ActuatorState(
                omega=self._state.omega + omega_delta,
                beta=self._state.beta + beta_delta,
            )
        )
        return self.state

    def _bounded_state(self, state: ActuatorState) -> ActuatorState:
        """把执行机构状态限制在转速和偏转角的物理范围内。"""
        return ActuatorState(
            omega=np.clip(
                state.omega,
                self.params.omega_min,
                self.params.omega_max,
            ),
            beta=np.clip(
                state.beta,
                self.params.beta_min,
                self.params.beta_max,
            ),
        )


def _actuator_vector(name: str, value: Sequence[float]) -> np.ndarray:
    """把执行机构变量转换为长度为 4 的有限浮点数组。"""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (ACTUATOR_COUNT,):
        raise ValueError(f"{name} must contain {ACTUATOR_COUNT} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _virtual_input_vector(value: Sequence[float]) -> np.ndarray:
    """把虚拟控制输入转换为长度为 5 的有限浮点数组。"""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (VIRTUAL_INPUT_SIZE,):
        raise ValueError("desired_virtual_input must contain five values")
    if not np.all(np.isfinite(array)):
        raise ValueError("desired_virtual_input must contain finite values")
    return array.copy()

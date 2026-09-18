"""对比全状态 NMPC 与位置 MPC 加高频姿态/角速度 PID 两种架构。"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

if "--no-show" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

for search_path in (PROJECT_ROOT, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.run_mpc_tracking import wrap_to_pi
from experiments.run_mpc_tracking_disturbance import (
    ALTITUDE_INTEGRAL_GAIN,
    DISTURBANCE_STD,
    MEASUREMENT_STD,
    PROCESS_STD,
    StateExtendedKalmanFilter,
    VERTICAL_FORCE_BIAS_LIMIT,
)
from gunyiji.control import (
    ActuatorState,
    AllocationConfig,
    CasadiNMPC,
    FirstOrderActuatorModel,
    HelicalTrajectory,
    NMPCConfig,
    NonlinearControlAllocator,
)
from gunyiji.dynamics import (
    DEFAULT_VEHICLE_PARAMS,
    RollingWingSixDOF,
    body_rate_from_euler_rate,
    rk4_step,
)


OUTER_DT = 0.1
INNER_DT = 0.01
POSITION_SIZE = 6
ACCELERATION_SIZE = 3


@dataclass(frozen=True)
class PositionMPCResult:
    """保存一次位置MPC求解得到的惯性系加速度及预测序列。"""

    acceleration: np.ndarray
    predicted_states: np.ndarray
    predicted_accelerations: np.ndarray
    objective: float
    success: bool
    status: str


class CasadiPositionMPC:
    """使用六维双积分模型优化位置、速度和惯性系期望加速度。"""

    def __init__(self, dt: float, horizon: int) -> None:
        """保存位置MPC参数并建立CasADi/IPOPT求解器。"""
        if dt <= 0.0 or horizon <= 0:
            raise ValueError("dt and horizon must be positive")
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.acceleration_lower = np.array([-2.0, -2.0, -3.0], dtype=float)
        self.acceleration_upper = np.array([2.0, 2.0, 3.0], dtype=float)
        self._last_solution: np.ndarray | None = None
        self._build_solver()

    def solve(
        self,
        state: np.ndarray,
        previous_acceleration: np.ndarray,
        reference: np.ndarray,
    ) -> PositionMPCResult:
        """根据当前平动状态和预测域参考求解第一拍惯性系加速度。"""
        x0 = np.asarray(state, dtype=float).reshape(-1)
        u_previous = np.asarray(previous_acceleration, dtype=float).reshape(-1)
        reference_array = np.asarray(reference, dtype=float)
        if x0.shape != (POSITION_SIZE,):
            raise ValueError("position MPC state must have shape (6,)")
        if u_previous.shape != (ACCELERATION_SIZE,):
            raise ValueError("previous_acceleration must have shape (3,)")
        if reference_array.shape != (self.horizon + 1, 9):
            raise ValueError(
                f"reference must have shape ({self.horizon + 1}, 9)"
            )

        parameters = np.concatenate((x0, u_previous, reference_array.reshape(-1)))
        initial_guess = self._initial_guess(x0, u_previous)
        try:
            solution = self._solver(
                x0=initial_guess,
                lbx=self._lbx,
                ubx=self._ubx,
                lbg=self._lbg,
                ubg=self._ubg,
                p=parameters,
            )
            decision = np.asarray(solution["x"], dtype=float).reshape(-1)
            stats = self._solver.stats()
            success = bool(stats.get("success", False))
            status = str(stats.get("return_status", "unknown"))
            objective = float(solution["f"])
            self._last_solution = decision
        except RuntimeError as error:
            decision = initial_guess
            success = False
            status = f"solver exception: {error}"
            objective = float("inf")

        state_count = POSITION_SIZE * (self.horizon + 1)
        predicted_states = decision[:state_count].reshape(
            POSITION_SIZE,
            self.horizon + 1,
            order="F",
        ).T
        predicted_accelerations = decision[state_count:].reshape(
            ACCELERATION_SIZE,
            self.horizon,
            order="F",
        ).T
        return PositionMPCResult(
            acceleration=predicted_accelerations[0].copy(),
            predicted_states=predicted_states,
            predicted_accelerations=predicted_accelerations,
            objective=objective,
            success=success,
            status=status,
        )

    def _build_solver(self) -> None:
        """建立带加速度和控制增量约束的双积分位置MPC。"""
        horizon = self.horizon
        states = ca.MX.sym("position_states", POSITION_SIZE, horizon + 1)
        accelerations = ca.MX.sym(
            "inertial_accelerations",
            ACCELERATION_SIZE,
            horizon,
        )
        parameter_count = POSITION_SIZE + ACCELERATION_SIZE + 9 * (horizon + 1)
        parameters = ca.MX.sym("parameters", parameter_count)

        current_state = parameters[:POSITION_SIZE]
        previous_acceleration = parameters[
            POSITION_SIZE : POSITION_SIZE + ACCELERATION_SIZE
        ]
        reference_offset = POSITION_SIZE + ACCELERATION_SIZE

        position_weight = ca.DM([45.0, 45.0, 70.0])
        velocity_weight = ca.DM([10.0, 10.0, 14.0])
        terminal_position_weight = ca.DM([120.0, 120.0, 180.0])
        terminal_velocity_weight = ca.DM([24.0, 24.0, 34.0])
        acceleration_weight = ca.DM([0.18, 0.18, 0.22])
        acceleration_delta_weight = ca.DM([0.55, 0.55, 0.70])

        constraints = [states[:, 0] - current_state]
        objective = 0
        for index in range(horizon):
            reference_index = reference_offset + 9 * index
            reference_position = parameters[reference_index : reference_index + 3]
            reference_velocity = parameters[reference_index + 3 : reference_index + 6]
            reference_acceleration = parameters[reference_index + 6 : reference_index + 9]

            position_error = states[0:3, index] - reference_position
            velocity_error = states[3:6, index] - reference_velocity
            acceleration_error = accelerations[:, index] - reference_acceleration
            previous = previous_acceleration if index == 0 else accelerations[:, index - 1]
            acceleration_delta = accelerations[:, index] - previous
            objective += ca.dot(position_weight * position_error, position_error)
            objective += ca.dot(velocity_weight * velocity_error, velocity_error)
            objective += ca.dot(
                acceleration_weight * acceleration_error,
                acceleration_error,
            )
            objective += ca.dot(
                acceleration_delta_weight * acceleration_delta,
                acceleration_delta,
            )

            position = states[0:3, index]
            velocity = states[3:6, index]
            acceleration = accelerations[:, index]
            next_state = ca.vertcat(
                position + self.dt * velocity + 0.5 * self.dt**2 * acceleration,
                velocity + self.dt * acceleration,
            )
            constraints.append(states[:, index + 1] - next_state)

        terminal_reference_index = reference_offset + 9 * horizon
        terminal_position_error = (
            states[0:3, horizon]
            - parameters[
                terminal_reference_index : terminal_reference_index + 3
            ]
        )
        terminal_velocity_error = (
            states[3:6, horizon]
            - parameters[
                terminal_reference_index + 3 : terminal_reference_index + 6
            ]
        )
        objective += ca.dot(
            terminal_position_weight * terminal_position_error,
            terminal_position_error,
        )
        objective += ca.dot(
            terminal_velocity_weight * terminal_velocity_error,
            terminal_velocity_error,
        )

        decision = ca.vertcat(
            ca.reshape(states, -1, 1),
            ca.reshape(accelerations, -1, 1),
        )
        problem = {
            "x": decision,
            "f": objective,
            "g": ca.vertcat(*constraints),
            "p": parameters,
        }
        options = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 80,
            "ipopt.tol": 1.0e-5,
            "print_time": False,
        }
        self._solver = ca.nlpsol("position_mpc", "ipopt", problem, options)

        state_variables = POSITION_SIZE * (horizon + 1)
        self._lbx = np.concatenate(
            (
                np.full(state_variables, -np.inf),
                np.tile(self.acceleration_lower, horizon),
            )
        )
        self._ubx = np.concatenate(
            (
                np.full(state_variables, np.inf),
                np.tile(self.acceleration_upper, horizon),
            )
        )
        self._lbg = np.zeros(POSITION_SIZE * (horizon + 1), dtype=float)
        self._ubg = np.zeros_like(self._lbg)

    def _initial_guess(
        self,
        current_state: np.ndarray,
        previous_acceleration: np.ndarray,
    ) -> np.ndarray:
        """首次复制当前状态，后续将上一最优序列移位作为热启动。"""
        state_count = POSITION_SIZE * (self.horizon + 1)
        if self._last_solution is None:
            state_guess = np.tile(current_state[:, None], (1, self.horizon + 1))
            acceleration_guess = np.tile(
                previous_acceleration[:, None],
                (1, self.horizon),
            )
        else:
            old_states = self._last_solution[:state_count].reshape(
                POSITION_SIZE,
                self.horizon + 1,
                order="F",
            )
            old_accelerations = self._last_solution[state_count:].reshape(
                ACCELERATION_SIZE,
                self.horizon,
                order="F",
            )
            state_guess = np.column_stack(
                (current_state, old_states[:, 2:], old_states[:, -1])
            )
            acceleration_guess = np.column_stack(
                (old_accelerations[:, 1:], old_accelerations[:, -1])
            )
        return np.concatenate(
            (
                state_guess.reshape(-1, order="F"),
                acceleration_guess.reshape(-1, order="F"),
            )
        )


class CascadedAttitudeRatePID:
    """姿态P外环和角速度PID内环组成的高频串级控制器。"""

    def __init__(self, torque_limits: np.ndarray) -> None:
        """设置姿态、角速度增益、积分限幅和力矩限幅。"""
        self.angle_kp = np.array([6.0, 6.0, 4.0], dtype=float)
        self.rate_kp = np.array([0.40, 0.45, 0.35], dtype=float)
        self.rate_ki = np.array([0.04, 0.04, 0.03], dtype=float)
        self.rate_kd = np.array([0.001, 0.001, 0.0008], dtype=float)
        self.maximum_rate = np.array([2.0, 2.0, 1.5], dtype=float)
        self.integral_limit = np.array([0.5, 0.5, 0.4], dtype=float)
        self.torque_limits = np.asarray(torque_limits, dtype=float)
        self.integral = np.zeros(3, dtype=float)
        self.previous_rate_error = np.zeros(3, dtype=float)
        self.filtered_rate_error_derivative = np.zeros(3, dtype=float)
        self._initialized = False

    def step(
        self,
        attitude: np.ndarray,
        body_rate: np.ndarray,
        desired_attitude: np.ndarray,
        desired_yaw_rate: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """根据姿态误差和机体系角速度误差计算三轴期望力矩。"""
        angle_error = wrap_to_pi(
            np.asarray(desired_attitude, dtype=float)
            - np.asarray(attitude, dtype=float)
        )
        body_rate_feedforward = body_rate_from_euler_rate(
            desired_attitude,
            (0.0, 0.0, desired_yaw_rate),
        )
        desired_rate = self.angle_kp * angle_error + body_rate_feedforward
        desired_rate = np.clip(desired_rate, -self.maximum_rate, self.maximum_rate)

        rate_error = desired_rate - np.asarray(body_rate, dtype=float)
        self.integral = np.clip(
            self.integral + rate_error * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        if self._initialized:
            raw_derivative = (rate_error - self.previous_rate_error) / dt
            derivative_alpha = 0.18
            self.filtered_rate_error_derivative += derivative_alpha * (
                raw_derivative - self.filtered_rate_error_derivative
            )
        else:
            self._initialized = True
        self.previous_rate_error = rate_error.copy()

        torque = (
            self.rate_kp * rate_error
            + self.rate_ki * self.integral
            + self.rate_kd * self.filtered_rate_error_derivative
        )
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)
        return torque, desired_rate


@dataclass(frozen=True)
class SimulationResult:
    """保存一种控制架构的高频闭环仿真历史和求解统计。"""

    architecture: str
    time: np.ndarray
    states: np.ndarray
    measured_states: np.ndarray
    estimated_states: np.ndarray
    references: np.ndarray
    desired_attitudes: np.ndarray
    desired_rates: np.ndarray
    desired_controls: np.ndarray
    actuator_controls: np.ndarray
    disturbances: np.ndarray
    solver_success: np.ndarray
    allocation_success: np.ndarray
    solver_times: np.ndarray


def main() -> None:
    """运行全状态NMPC与位置MPC加姿态PID的同工况对照实验。"""
    args = parse_args()
    validate_time_steps(OUTER_DT, args.inner_dt)
    random = np.random.default_rng(args.seed)
    inner_steps = int(round(args.duration / args.inner_dt))
    measurement_noise = random.normal(size=(inner_steps + 1, 12)) * MEASUREMENT_STD
    disturbance_white = random.normal(size=(inner_steps, 5))

    architectures = (
        ("full_nmpc", "全状态 NMPC"),
        ("cascade_pid", "位置 MPC + 姿态/角速度 PID"),
    )
    results: list[SimulationResult] = []
    for architecture, description in architectures:
        print(f"正在运行：{description}……")
        results.append(
            run_simulation(
                architecture=architecture,
                args=args,
                measurement_noise=measurement_noise,
                disturbance_white=disturbance_white,
            )
        )

    metrics = {result.architecture: calculate_metrics(result) for result in results}
    print_comparison(metrics)
    save_results(results, metrics, args)
    figure = plot_comparison(results, metrics)
    figure_path = OUTPUT_DIR / "full_nmpc_vs_cascade_pid.png"
    figure.savefig(figure_path, dpi=170)
    print(f"结果图已保存至：{figure_path}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


def run_simulation(
    architecture: str,
    args: Namespace,
    measurement_noise: np.ndarray,
    disturbance_white: np.ndarray,
    *,
    use_ekf: bool = True,
    measurement_std: np.ndarray | None = None,
    position_controller_override=None,
    initial_position_offset: np.ndarray | None = None,
) -> SimulationResult:
    """在共享噪声和外扰下运行指定控制架构的高频闭环仿真。"""
    if architecture not in ("full_nmpc", "cascade_pid"):
        raise ValueError("unknown architecture")
    measurement_std_array = (
        np.asarray(MEASUREMENT_STD, dtype=float)
        if measurement_std is None
        else np.asarray(measurement_std, dtype=float)
    )
    if measurement_std_array.shape != (12,):
        raise ValueError("measurement_std must have shape (12,)")

    nominal_params = DEFAULT_VEHICLE_PARAMS
    true_params = replace(
        nominal_params,
        mass=1.0 * nominal_params.mass,
        ix=1.0 * nominal_params.ix,
        iy=1.0 * nominal_params.iy,
        iz=1.0 * nominal_params.iz,
    )
    trajectory = HelicalTrajectory(
        radius=2.0,
        angular_rate=0.5,
        climb_rate=0.2,
        center=(0.0, 0.0, -1.0),
    )
    # 全状态 NMPC 分支保持原实验的 10 Hz 运行方式；只有串级架构的
    # 姿态控制、控制分配、执行机构、被控对象和 EKF 内环以 100 Hz 运行。
    inner_dt = OUTER_DT if architecture == "full_nmpc" else args.inner_dt
    inner_per_outer = int(round(OUTER_DT / inner_dt))
    outer_steps = int(round(args.duration / OUTER_DT))
    inner_steps = outer_steps * inner_per_outer
    time = np.arange(inner_steps + 1, dtype=float) * inner_dt

    plant = RollingWingSixDOF(true_params)
    estimator_model = RollingWingSixDOF(nominal_params)
    allocator = NonlinearControlAllocator(
        vehicle=nominal_params,
        config=AllocationConfig(enforce_command_rate_limits=False),
    )
    actuator = FirstOrderActuatorModel(vehicle=nominal_params)
    full_controller = (
        CasadiNMPC(
            params=nominal_params,
            config=NMPCConfig(dt=OUTER_DT, horizon=args.horizon),
        )
        if architecture == "full_nmpc"
        else None
    )
    position_controller = (
        (position_controller_override if position_controller_override is not None
         else CasadiPositionMPC(dt=OUTER_DT, horizon=args.horizon))
        if architecture == "cascade_pid"
        else None
    )
    attitude_controller = CascadedAttitudeRatePID(
        np.array(
            [
                nominal_params.max_tau_x,
                nominal_params.max_tau_y,
                nominal_params.max_tau_z,
            ],
            dtype=float,
        )
    )

    states = np.zeros((inner_steps + 1, 12), dtype=float)
    measured_states = np.zeros_like(states)
    estimated_states = np.zeros_like(states)
    references = np.zeros((inner_steps + 1, 6), dtype=float)
    desired_attitudes = np.zeros((inner_steps, 3), dtype=float)
    desired_rates = np.zeros((inner_steps, 3), dtype=float)
    desired_controls = np.zeros((inner_steps, 5), dtype=float)
    actuator_controls = np.zeros((inner_steps, 5), dtype=float)
    disturbances = np.zeros((inner_steps, 5), dtype=float)
    allocation_success = np.zeros(inner_steps, dtype=bool)
    solver_success = np.zeros(outer_steps, dtype=bool)
    solver_times = np.zeros(outer_steps, dtype=float)

    states[0] = trajectory.state(0.0)
    if initial_position_offset is not None:
        states[0, :3] += np.asarray(initial_position_offset, dtype=float)
    measured_states[0] = states[0] + measurement_noise[0]
    estimated_states[0] = measured_states[0]
    references[0] = trajectory.output(0.0)
    process_scale = np.sqrt(inner_dt / OUTER_DT)
    estimator = StateExtendedKalmanFilter(
        model=estimator_model,
        dt=inner_dt,
        initial_state=estimated_states[0],
        initial_covariance=np.diag(measurement_std_array**2),
        process_covariance=np.diag((PROCESS_STD * process_scale) ** 2),
        measurement_covariance=np.diag(measurement_std_array**2),
    )

    previous_nmpc_input = np.asarray(nominal_params.hover_input, dtype=float)
    previous_acceleration = np.zeros(3, dtype=float)
    held_virtual_control = previous_nmpc_input.copy()
    held_body_force = held_virtual_control[:2].copy()
    held_roll_pitch = np.zeros(2, dtype=float)
    colored_disturbance = np.zeros(5, dtype=float)
    fz_integral_bias = 0.0
    disturbance_correlation = np.exp(-inner_dt / 0.6)
    disturbance_scale = np.sqrt(1.0 - disturbance_correlation**2)
    legacy_random = (
        np.random.default_rng(args.seed) if architecture == "full_nmpc" else None
    )
    if legacy_random is not None:
        measured_states[0] = (
            states[0]
            + measurement_std_array * legacy_random.normal(size=12)
        )
        estimated_states[0] = measured_states[0]
        estimator = StateExtendedKalmanFilter(
            model=estimator_model,
            dt=inner_dt,
            initial_state=estimated_states[0],
            initial_covariance=np.diag(measurement_std_array**2),
            process_covariance=np.diag(PROCESS_STD**2),
            measurement_covariance=np.diag(measurement_std_array**2),
        )

    for inner_index in range(inner_steps):
        current_time = time[inner_index]
        feedback_state = (
            estimated_states[inner_index]
            if use_ekf
            else measured_states[inner_index]
        )

        if inner_index % inner_per_outer == 0:
            outer_index = inner_index // inner_per_outer
            solve_start = perf_counter()
            if architecture == "full_nmpc":
                assert full_controller is not None
                horizon_reference = trajectory.horizon_outputs(
                    current_time,
                    OUTER_DT,
                    args.horizon,
                )
                solve_result = full_controller.solve(
                    feedback_state,
                    previous_nmpc_input,
                    horizon_reference,
                )
                fz_integral_bias = update_vertical_force_integral(
                    fz_integral_bias,
                    horizon_reference[0, 2] - feedback_state[2],
                    OUTER_DT,
                )
                held_virtual_control = solve_result.control.copy()
                held_virtual_control[1] = np.clip(
                    held_virtual_control[1] + fz_integral_bias,
                    nominal_params.min_fz,
                    nominal_params.max_fz,
                )
                fz_integral_bias = held_virtual_control[1] - solve_result.control[1]
                previous_nmpc_input = solve_result.control.copy()
                solver_success[outer_index] = solve_result.success
            else:
                assert position_controller is not None
                position_reference = helix_position_horizon(
                    trajectory,
                    current_time,
                    OUTER_DT,
                    args.horizon,
                )
                solve_result = position_controller.solve(
                    feedback_state[0:6],
                    previous_acceleration,
                    position_reference,
                )
                fz_integral_bias = update_vertical_force_integral(
                    fz_integral_bias,
                    position_reference[0, 2] - feedback_state[2],
                    OUTER_DT,
                )
                inertial_force = acceleration_to_inertial_force(
                    solve_result.acceleration,
                    nominal_params.mass,
                    nominal_params.gravity,
                )
                inertial_force[2] += fz_integral_bias
                desired_yaw = float(trajectory.output(current_time)[5])
                held_body_force, desired_attitude = force_to_body_command(
                    inertial_force,
                    desired_yaw,
                    nominal_params,
                )
                held_roll_pitch = desired_attitude[0:2]
                previous_acceleration = solve_result.acceleration.copy()
                solver_success[outer_index] = solve_result.success
            solver_times[outer_index] = perf_counter() - solve_start

        current_reference = trajectory.output(current_time)
        if architecture == "cascade_pid":
            desired_attitude = np.array(
                [held_roll_pitch[0], held_roll_pitch[1], current_reference[5]],
                dtype=float,
            )
            torque, desired_rate = attitude_controller.step(
                feedback_state[6:9],
                feedback_state[9:12],
                desired_attitude,
                trajectory.angular_rate,
                inner_dt,
            )
            desired_control = np.concatenate((held_body_force, torque))
        else:
            desired_attitude = current_reference[3:6]
            desired_rate = body_rate_from_euler_rate(
                desired_attitude,
                (0.0, 0.0, trajectory.angular_rate),
            )
            desired_control = held_virtual_control.copy()

        allocation_result = allocator.allocate(desired_control, dt=inner_dt)
        actuator.step(
            ActuatorState(
                omega=allocation_result.omega,
                beta=allocation_result.beta,
            ),
            inner_dt,
        )
        actuator_input = actuator.virtual_input
        disturbance_innovation = (
            legacy_random.normal(size=5)
            if legacy_random is not None
            else disturbance_white[inner_index]
        )
        colored_disturbance = (
            disturbance_correlation * colored_disturbance
            + disturbance_scale
            * DISTURBANCE_STD
            * disturbance_innovation
        )
        plant_input = actuator_input + colored_disturbance

        desired_attitudes[inner_index] = desired_attitude
        desired_rates[inner_index] = desired_rate
        desired_controls[inner_index] = desired_control
        actuator_controls[inner_index] = actuator_input
        disturbances[inner_index] = colored_disturbance
        allocation_success[inner_index] = allocation_result.success

        states[inner_index + 1] = rk4_step(
            plant.derivative,
            current_time,
            states[inner_index],
            plant_input,
            inner_dt,
        )
        measurement_perturbation = (
            measurement_std_array * legacy_random.normal(size=12)
            if legacy_random is not None
            else measurement_noise[inner_index + 1]
        )
        measured_states[inner_index + 1] = (
            states[inner_index + 1] + measurement_perturbation
        )
        estimator.predict(current_time, actuator_input)
        estimated_states[inner_index + 1], _, _ = estimator.update(
            measured_states[inner_index + 1]
        )
        references[inner_index + 1] = trajectory.output(time[inner_index + 1])

    return SimulationResult(
        architecture=architecture,
        time=time,
        states=states,
        measured_states=measured_states,
        estimated_states=estimated_states,
        references=references,
        desired_attitudes=desired_attitudes,
        desired_rates=desired_rates,
        desired_controls=desired_controls,
        actuator_controls=actuator_controls,
        disturbances=disturbances,
        solver_success=solver_success,
        allocation_success=allocation_success,
        solver_times=solver_times,
    )


def helix_position_horizon(
    trajectory: HelicalTrajectory,
    start_time: float,
    dt: float,
    horizon: int,
) -> np.ndarray:
    """生成位置MPC使用的位置、速度和加速度九维预测域参考。"""
    reference = np.zeros((horizon + 1, 9), dtype=float)
    for index in range(horizon + 1):
        time = start_time + index * dt
        angle = trajectory.angular_rate * time
        reference[index, 0:3] = trajectory.output(time)[0:3]
        reference[index, 3:6] = trajectory.velocity(time)
        reference[index, 6:9] = np.array(
            [
                -trajectory.radius * trajectory.angular_rate**2 * np.cos(angle),
                -trajectory.radius * trajectory.angular_rate**2 * np.sin(angle),
                0.0,
            ],
            dtype=float,
        )
    return reference


def acceleration_to_inertial_force(
    desired_acceleration: np.ndarray,
    mass: float,
    gravity: float,
) -> np.ndarray:
    """将NED惯性系期望净加速度转换为不含重力的期望合外力。"""
    gravity_acceleration = np.array([0.0, 0.0, gravity], dtype=float)
    return mass * (np.asarray(desired_acceleration, dtype=float) - gravity_acceleration)


def force_to_body_command(
    inertial_force: np.ndarray,
    desired_yaw: float,
    vehicle,
) -> tuple[np.ndarray, np.ndarray]:
    """将惯性系合力与航向解算为滚转、零俯仰和机体系fx/fz。"""
    force = np.asarray(inertial_force, dtype=float)
    cosine = np.cos(desired_yaw)
    sine = np.sin(desired_yaw)
    heading_force = np.array(
        [
            cosine * force[0] + sine * force[1],
            -sine * force[0] + cosine * force[1],
            force[2],
        ],
        dtype=float,
    )
    maximum_roll = np.deg2rad(35.0)
    desired_roll = np.clip(
        np.arctan2(heading_force[1], -heading_force[2]),
        -maximum_roll,
        maximum_roll,
    )
    desired_pitch = 0.0
    body_fx = heading_force[0]
    body_fz = (
        -np.sin(desired_roll) * heading_force[1]
        + np.cos(desired_roll) * heading_force[2]
    )
    body_force = np.array(
        [
            np.clip(body_fx, vehicle.min_fx, vehicle.max_fx),
            np.clip(body_fz, vehicle.min_fz, vehicle.max_fz),
        ],
        dtype=float,
    )
    desired_attitude = np.array(
        [desired_roll, desired_pitch, desired_yaw],
        dtype=float,
    )
    return body_force, desired_attitude


def update_vertical_force_integral(
    current_bias: float,
    vertical_position_error: float,
    dt: float,
) -> float:
    """更新用于补偿质量失配的垂向力积分修正。"""
    return float(
        np.clip(
            current_bias
            + ALTITUDE_INTEGRAL_GAIN * vertical_position_error * dt,
            -VERTICAL_FORCE_BIAS_LIMIT,
            VERTICAL_FORCE_BIAS_LIMIT,
        )
    )


def calculate_metrics(result: SimulationResult) -> dict[str, float]:
    """计算真实跟踪、状态估计、控制平滑性和求解耗时指标。"""
    position_error = np.linalg.norm(
        result.states[:, 0:3] - result.references[:, 0:3],
        axis=1,
    )
    yaw_error = np.abs(
        wrap_to_pi(result.states[:, 8] - result.references[:, 5])
    )
    estimation_position_error = np.linalg.norm(
        result.estimated_states[:, 0:3] - result.states[:, 0:3],
        axis=1,
    )
    control_delta = np.linalg.norm(np.diff(result.desired_controls, axis=0), axis=1)
    return {
        "mean_position_error": float(position_error.mean()),
        "max_position_error": float(position_error.max()),
        "final_position_error": float(position_error[-1]),
        "mean_yaw_error_deg": float(np.rad2deg(yaw_error.mean())),
        "mean_estimation_error": float(estimation_position_error.mean()),
        "mean_control_delta": float(control_delta.mean()),
        "solver_success_rate": float(result.solver_success.mean()),
        "allocation_success_rate": float(result.allocation_success.mean()),
        "mean_solver_time_ms": float(1000.0 * result.solver_times.mean()),
        "max_solver_time_ms": float(1000.0 * result.solver_times.max()),
    }


def print_comparison(metrics: dict[str, dict[str, float]]) -> None:
    """在终端输出两种控制架构的关键指标对照。"""
    print("\n以下两组采用相同的模型、噪声参数和随机种子")
    print(f"{'指标':<26}{'全状态 NMPC':>18}{'MPC + PID':>18}")
    labels = (
        ("mean_position_error", "平均位置误差 [m]"),
        ("max_position_error", "最大位置误差 [m]"),
        ("final_position_error", "终点位置误差 [m]"),
        ("mean_yaw_error_deg", "平均偏航误差 [deg]"),
        ("mean_estimation_error", "平均 EKF 位置误差 [m]"),
        ("mean_control_delta", "平均控制增量"),
        ("solver_success_rate", "求解成功率"),
        ("allocation_success_rate", "控制分配成功率"),
        ("mean_solver_time_ms", "外环平均求解时间 [ms]"),
        ("max_solver_time_ms", "外环最大求解时间 [ms]"),
    )
    for key, label in labels:
        full_value = metrics["full_nmpc"][key]
        cascade_value = metrics["cascade_pid"][key]
        if "rate" in key:
            print(f"{label:<30}{100.0 * full_value:>15.1f}%{100.0 * cascade_value:>15.1f}%")
        else:
            print(f"{label:<30}{full_value:>16.4f}{cascade_value:>16.4f}")


def save_results(
    results: list[SimulationResult],
    metrics: dict[str, dict[str, float]],
    args: Namespace,
) -> None:
    """将两组完整仿真历史和指标保存为一个压缩NPZ文件。"""
    payload: dict[str, np.ndarray | float | int] = {
        "measurement_std": MEASUREMENT_STD,
        "process_std_outer_step": PROCESS_STD,
        "disturbance_std": DISTURBANCE_STD,
        "outer_dt": OUTER_DT,
        "inner_dt": args.inner_dt,
        "seed": args.seed,
    }
    for result in results:
        prefix = result.architecture
        for field_name in (
            "time",
            "states",
            "measured_states",
            "estimated_states",
            "references",
            "desired_attitudes",
            "desired_rates",
            "desired_controls",
            "actuator_controls",
            "disturbances",
            "solver_success",
            "allocation_success",
            "solver_times",
        ):
            payload[f"{prefix}_{field_name}"] = getattr(result, field_name)
        for metric_name, value in metrics[prefix].items():
            payload[f"{prefix}_{metric_name}"] = value
    output_path = OUTPUT_DIR / "full_nmpc_vs_cascade_pid_data.npz"
    np.savez_compressed(output_path, **payload)
    print(f"数据已保存至：{output_path}")


def plot_comparison(
    results: list[SimulationResult],
    metrics: dict[str, dict[str, float]],
) -> plt.Figure:
    """绘制两种架构的轨迹、误差、姿态、力矩和求解耗时对照。"""
    full = next(result for result in results if result.architecture == "full_nmpc")
    cascade = next(result for result in results if result.architecture == "cascade_pid")
    figure = plt.figure(figsize=(15, 12), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)

    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    axis_3d.plot(
        full.references[:, 0],
        full.references[:, 1],
        -full.references[:, 2],
        "k--",
        linewidth=1.5,
        label="参考轨迹",
    )
    axis_3d.plot(
        full.states[:, 0],
        full.states[:, 1],
        -full.states[:, 2],
        label="全状态 NMPC",
    )
    axis_3d.plot(
        cascade.states[:, 0],
        cascade.states[:, 1],
        -cascade.states[:, 2],
        label="位置 MPC + PID",
    )
    axis_3d.set_xlabel("x [m]")
    axis_3d.set_ylabel("y [m]")
    axis_3d.set_zlabel("高度 [m]")
    axis_3d.set_title("螺旋线跟踪")
    axis_3d.legend()

    axis_position = figure.add_subplot(grid[0, 1])
    for result, label in ((full, "全状态 NMPC"), (cascade, "MPC + PID")):
        error = np.linalg.norm(
            result.states[:, 0:3] - result.references[:, 0:3],
            axis=1,
        )
        axis_position.plot(result.time, error, label=label)
    axis_position.set_title("位置误差范数")
    axis_position.set_xlabel("时间 [s]")
    axis_position.set_ylabel("误差 [m]")
    axis_position.grid(True)
    axis_position.legend()

    axis_attitude = figure.add_subplot(grid[1, 0])
    attitude_time = cascade.time[:-1]
    for index, label in enumerate(("滚转", "俯仰", "偏航")):
        axis_attitude.plot(
            attitude_time,
            np.rad2deg(cascade.desired_attitudes[:, index]),
            "--",
            linewidth=1.0,
            label=f"{label}期望值",
        )
        axis_attitude.plot(
            cascade.time,
            np.rad2deg(cascade.states[:, 6 + index]),
            linewidth=1.0,
            label=f"{label}实际值",
        )
    axis_attitude.set_title("串级姿态跟踪")
    axis_attitude.set_xlabel("时间 [s]")
    axis_attitude.set_ylabel("角度 [deg]")
    axis_attitude.grid(True)
    axis_attitude.legend(ncol=2, fontsize=7)

    axis_torque = figure.add_subplot(grid[1, 1])
    for result, line_style, prefix in (
        (full, "--", "NMPC"),
        (cascade, "-", "PID"),
    ):
        for index, axis_name in enumerate(("x", "y", "z")):
            axis_torque.plot(
                result.time[:-1],
                result.desired_controls[:, 2 + index],
                line_style,
                linewidth=0.9,
                label=f"{prefix} tau_{axis_name}",
            )
    axis_torque.set_title("期望机体系力矩")
    axis_torque.set_xlabel("时间 [s]")
    axis_torque.set_ylabel("力矩 [N m]")
    axis_torque.grid(True)
    axis_torque.legend(ncol=2, fontsize=7)

    axis_estimation = figure.add_subplot(grid[2, 0])
    for result, label in ((full, "全状态 NMPC"), (cascade, "MPC + PID")):
        estimate_error = np.linalg.norm(
            result.estimated_states[:, 0:3] - result.states[:, 0:3],
            axis=1,
        )
        axis_estimation.plot(result.time, estimate_error, label=label)
    axis_estimation.set_title("EKF 位置估计误差")
    axis_estimation.set_xlabel("时间 [s]")
    axis_estimation.set_ylabel("误差 [m]")
    axis_estimation.grid(True)
    axis_estimation.legend()

    axis_solver = figure.add_subplot(grid[2, 1])
    outer_time = np.arange(len(full.solver_times), dtype=float) * OUTER_DT
    axis_solver.plot(outer_time, 1000.0 * full.solver_times, label="全状态 NMPC")
    axis_solver.plot(
        outer_time,
        1000.0 * cascade.solver_times,
        label="位置 MPC",
    )
    axis_solver.axhline(1000.0 * OUTER_DT, color="k", linestyle="--", label="100 ms")
    axis_solver.set_title("外环求解时间")
    axis_solver.set_xlabel("时间 [s]")
    axis_solver.set_ylabel("求解时间 [ms]")
    axis_solver.grid(True)
    axis_solver.legend()

    full_error = metrics["full_nmpc"]["mean_position_error"]
    cascade_error = metrics["cascade_pid"]["mean_position_error"]
    figure.suptitle(
        "全状态 NMPC 与位置 MPC + 100 Hz 姿态/角速度 PID 对比\n"
        f"平均位置误差：{full_error:.3f} m 对比 {cascade_error:.3f} m",
        fontsize=14,
    )
    return figure


def validate_time_steps(outer_dt: float, inner_dt: float) -> None:
    """检查内外环周期能够构成整数倍频关系。"""
    if inner_dt <= 0.0:
        raise ValueError("inner_dt must be positive")
    ratio = outer_dt / inner_dt
    if not np.isclose(ratio, round(ratio), atol=1.0e-9):
        raise ValueError("outer_dt must be an integer multiple of inner_dt")


def parse_args() -> Namespace:
    """解析仿真时长、预测域、随机种子和内环周期。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--inner-dt", type=float, default=INNER_DT)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration 必须为正数")
    if args.horizon <= 0:
        parser.error("--horizon 必须为正整数")
    if not np.isclose(args.duration / OUTER_DT, round(args.duration / OUTER_DT)):
        parser.error("--duration 必须是 0.1 s 的整数倍")
    return args


if __name__ == "__main__":
    main()

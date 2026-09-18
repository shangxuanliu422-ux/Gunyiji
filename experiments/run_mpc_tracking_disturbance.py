"""测试 NMPC 在噪声、外扰和模型失配条件下的螺旋线跟踪性能。"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

if "--no-show" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

for search_path in (PROJECT_ROOT, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.run_mpc_tracking import animate_helix_3d, wrap_to_pi
from gunyiji.control import (
    ActuatorState,
    AllocationConfig,
    CasadiNMPC,
    FirstOrderActuatorModel,
    HelicalTrajectory,
    NMPCConfig,
    NonlinearControlAllocator,
)
from gunyiji.dynamics import DEFAULT_VEHICLE_PARAMS, RollingWingSixDOF, rk4_step

# 测量噪声标准差，依次对应 [位置、速度、欧拉角、机体系角速度]。
MEASUREMENT_STD = np.array(
    [0.01] * 3 + [0.02] * 3 + [0.003] * 3 + [0.008] * 3,
    dtype=float,
)
# 单步模型不确定性，依次对应 [位置、速度、欧拉角、机体系角速度]。
# 估计器使用标称模型，而仿真对象存在参数失配和外扰，因此这里的取值
# 有意高于单纯数值积分误差。
PROCESS_STD = np.array(
    [0.002, 0.002, 0.020]
    + [0.020, 0.020, 0.080]
    + [0.001] * 3
    + [0.020] * 3,
    dtype=float,
)
# 有色外扰标准差，依次对应 [fx, fz, tau_x, tau_y, tau_z]。
DISTURBANCE_STD = np.array([0.20, 0.35, 0.008, 0.008, 0.006], dtype=float)

# 施加在 NMPC 垂向力指令上的慢速积分修正。在当前 NED 坐标约定下，
# 飞行器低于参考高度时 z_ref-z 为负，积分修正会使 fz 更负，从而增大升力。
ALTITUDE_INTEGRAL_GAIN = 1  # N / (m s)
VERTICAL_FORCE_BIAS_LIMIT = 10.0  # N


class StateExtendedKalmanFilter:
    """使用标称六自由度模型融合十二维带噪状态测量。"""

    def __init__(
        self,
        model: RollingWingSixDOF,
        dt: float,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_covariance: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> None:
        """保存EKF状态、协方差以及预测和测量噪声配置。"""
        self.model = model
        self.dt = float(dt)
        self._state = np.asarray(initial_state, dtype=float).copy()
        self._covariance = np.asarray(initial_covariance, dtype=float).copy()
        self.process_covariance = np.asarray(process_covariance, dtype=float).copy()
        self.measurement_covariance = np.asarray(
            measurement_covariance,
            dtype=float,
        ).copy()
        if self._state.shape != (12,):
            raise ValueError("initial_state must have shape (12,)")
        for name, matrix in (
            ("initial_covariance", self._covariance),
            ("process_covariance", self.process_covariance),
            ("measurement_covariance", self.measurement_covariance),
        ):
            if matrix.shape != (12, 12):
                raise ValueError(f"{name} must have shape (12, 12)")

    @property
    def state(self) -> np.ndarray:
        """返回当前十二维状态估计的副本。"""
        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        """返回当前状态估计误差协方差的副本。"""
        return self._covariance.copy()

    def predict(self, time: float, virtual_input: np.ndarray) -> np.ndarray:
        """使用标称动力学和已知执行机构输入预测下一时刻状态。"""
        state_before_prediction = self._state.copy()
        transition_jacobian = self._transition_jacobian(
            time,
            state_before_prediction,
            virtual_input,
        )
        self._state = self._transition(time, state_before_prediction, virtual_input)
        self._covariance = (
            transition_jacobian
            @ self._covariance
            @ transition_jacobian.T
            + self.process_covariance
        )
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        return self.state

    def update(
        self,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """用十二维状态测量修正预测，并返回估计、新息和卡尔曼增益。"""
        measured = np.asarray(measurement, dtype=float)
        if measured.shape != (12,):
            raise ValueError("measurement must have shape (12,)")

        innovation = measured - self._state
        innovation[6:9] = wrap_to_pi(innovation[6:9])
        innovation_covariance = self._covariance + self.measurement_covariance
        kalman_gain = np.linalg.solve(
            innovation_covariance.T,
            self._covariance.T,
        ).T
        self._state = self._state + kalman_gain @ innovation

        identity = np.eye(12, dtype=float)
        correction = identity - kalman_gain
        self._covariance = (
            correction @ self._covariance @ correction.T
            + kalman_gain @ self.measurement_covariance @ kalman_gain.T
        )
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        return self.state, innovation, kalman_gain

    def _transition(
        self,
        time: float,
        state: np.ndarray,
        virtual_input: np.ndarray,
    ) -> np.ndarray:
        """通过RK4计算标称非线性离散状态转移。"""
        return rk4_step(
            self.model.derivative,
            time,
            state,
            virtual_input,
            self.dt,
        )

    def _transition_jacobian(
        self,
        time: float,
        state: np.ndarray,
        virtual_input: np.ndarray,
    ) -> np.ndarray:
        """使用中心有限差分计算离散动力学对状态的雅可比矩阵。"""
        jacobian = np.empty((12, 12), dtype=float)
        for state_index in range(12):
            epsilon = 1.0e-5 * max(1.0, abs(float(state[state_index])))
            offset = np.zeros(12, dtype=float)
            offset[state_index] = epsilon
            forward = self._transition(time, state + offset, virtual_input)
            backward = self._transition(time, state - offset, virtual_input)
            jacobian[:, state_index] = (forward - backward) / (2.0 * epsilon)
        return jacobian


def main() -> None:
    """运行带模型失配、测量噪声和随机外扰的螺旋线跟踪试验。"""
    args = parse_args()
    use_ekf = not args.no_ekf
    rng = np.random.default_rng(args.seed)

    nominal_params = DEFAULT_VEHICLE_PARAMS
    true_params = replace(
        nominal_params,
        mass=1.15 * nominal_params.mass,
        ix=1.15 * nominal_params.ix,
        iy=0.85 * nominal_params.iy,
        iz=1.15 * nominal_params.iz,
    )
    config = NMPCConfig(dt=0.1, horizon=args.horizon)
    plant = RollingWingSixDOF(true_params)
    estimator_model = RollingWingSixDOF(nominal_params)
    controller = CasadiNMPC(params=nominal_params, config=config)
    allocator = NonlinearControlAllocator(
        vehicle=nominal_params,
        config=AllocationConfig(enforce_command_rate_limits=False),
    )
    actuator = FirstOrderActuatorModel(vehicle=nominal_params)
    trajectory = HelicalTrajectory(
        radius=2.0,
        angular_rate=0.5,
        climb_rate=0.2,
        center=(0.0, 0.0, -1.0),
    )

    steps = int(round(args.duration / config.dt))
    time = np.arange(steps + 1, dtype=float) * config.dt
    states = np.zeros((steps + 1, 12), dtype=float)
    measured_states = np.zeros_like(states)
    estimated_states = np.zeros_like(states)
    predicted_estimates = np.zeros_like(states)
    covariance_diagonal = np.zeros_like(states)
    innovations = np.zeros_like(states)
    kalman_gain_diagonal = np.zeros_like(states)
    references = np.zeros((steps + 1, 6), dtype=float)
    nmpc_controls = np.zeros((steps, 5), dtype=float)
    desired_controls = np.zeros((steps, 5), dtype=float)
    actuator_controls = np.zeros((steps, 5), dtype=float)
    plant_controls = np.zeros((steps, 5), dtype=float)
    disturbances = np.zeros((steps, 5), dtype=float)
    vertical_force_bias = np.zeros(steps, dtype=float)
    altitude_errors = np.zeros(steps, dtype=float)
    nmpc_success = np.zeros(steps, dtype=bool)
    allocation_success = np.zeros(steps, dtype=bool)

    states[0] = trajectory.state(0.0)
    measured_states[0] = noisy_measurement(states[0], rng)
    estimated_states[0] = measured_states[0]
    predicted_estimates[0] = measured_states[0]
    references[0] = trajectory.output(0.0)
    measurement_covariance = np.diag(MEASUREMENT_STD**2)
    process_covariance = np.diag(PROCESS_STD**2)
    estimator = StateExtendedKalmanFilter(
        model=estimator_model,
        dt=config.dt,
        initial_state=estimated_states[0],
        initial_covariance=measurement_covariance,
        process_covariance=process_covariance,
        measurement_covariance=measurement_covariance,
    )
    covariance_diagonal[0] = np.diag(estimator.covariance)
    previous_nmpc_input = np.asarray(nominal_params.hover_input, dtype=float)
    colored_disturbance = np.zeros(5, dtype=float)
    fz_integral_bias = 0.0
    correlation = np.exp(-config.dt / 0.6)
    innovation_scale = np.sqrt(1.0 - correlation**2)

    for index in range(steps):
        current_time = time[index]
        horizon_reference = trajectory.horizon_outputs(
            current_time,
            config.dt,
            config.horizon,
        )
        feedback_state = estimated_states[index] if use_ekf else measured_states[index]
        nmpc_result = controller.solve(
            feedback_state,
            previous_nmpc_input,
            horizon_reference,
        )

        altitude_error = horizon_reference[0, 2] - feedback_state[2]
        fz_integral_bias = np.clip(
            fz_integral_bias
            + ALTITUDE_INTEGRAL_GAIN * altitude_error * config.dt,
            -VERTICAL_FORCE_BIAS_LIMIT,
            VERTICAL_FORCE_BIAS_LIMIT,
        )
        corrected_control = nmpc_result.control.copy()
        unclipped_fz = corrected_control[1] + fz_integral_bias
        corrected_control[1] = np.clip(
            unclipped_fz,
            nominal_params.min_fz,
            nominal_params.max_fz,
        )
        # 垂向力限幅后，同步修正积分器内部状态，避免积分继续累积。
        fz_integral_bias = corrected_control[1] - nmpc_result.control[1]

        allocation_result = allocator.allocate(corrected_control, dt=config.dt)
        actual_actuator_state = actuator.step(
            ActuatorState(
                omega=allocation_result.omega,
                beta=allocation_result.beta,
            ),
            config.dt,
        )
        del actual_actuator_state

        colored_disturbance = (
            correlation * colored_disturbance
            + innovation_scale * DISTURBANCE_STD * rng.normal(size=5)
        )
        actuator_input = actuator.virtual_input
        plant_input = actuator_input + colored_disturbance

        nmpc_controls[index] = nmpc_result.control
        desired_controls[index] = corrected_control
        actuator_controls[index] = actuator_input
        plant_controls[index] = plant_input
        disturbances[index] = colored_disturbance
        vertical_force_bias[index] = fz_integral_bias
        altitude_errors[index] = altitude_error
        nmpc_success[index] = nmpc_result.success
        allocation_success[index] = allocation_result.success

        states[index + 1] = rk4_step(
            plant.derivative,
            current_time,
            states[index],
            plant_input,
            config.dt,
        )
        measured_states[index + 1] = noisy_measurement(states[index + 1], rng)
        if use_ekf:
            predicted_estimates[index + 1] = estimator.predict(
                current_time,
                actuator_input,
            )
            (
                estimated_states[index + 1],
                innovations[index + 1],
                kalman_gain,
            ) = estimator.update(measured_states[index + 1])
            covariance_diagonal[index + 1] = np.diag(estimator.covariance)
            kalman_gain_diagonal[index + 1] = np.diag(kalman_gain)
        else:
            predicted_estimates[index + 1] = measured_states[index + 1]
            estimated_states[index + 1] = measured_states[index + 1]
        references[index + 1] = trajectory.output(time[index + 1])
        # 积分修正属于预测模型外的残余输入。如果把执行机构总输出作为
        # U_PREV 反馈给标称 NMPC，优化器会在下一拍反向抵消这项修正。
        previous_nmpc_input = nmpc_result.control.copy()

    position_error = np.linalg.norm(states[:, 0:3] - references[:, 0:3], axis=1)
    attitude_error = np.linalg.norm(
        wrap_to_pi(states[:, 6:9] - references[:, 3:6]),
        axis=1,
    )
    measurement_position_error = np.linalg.norm(
        measured_states[:, 0:3] - states[:, 0:3],
        axis=1,
    )
    estimation_position_error = np.linalg.norm(
        estimated_states[:, 0:3] - states[:, 0:3],
        axis=1,
    )
    measurement_velocity_error = np.linalg.norm(
        measured_states[:, 3:6] - states[:, 3:6],
        axis=1,
    )
    estimation_velocity_error = np.linalg.norm(
        estimated_states[:, 3:6] - states[:, 3:6],
        axis=1,
    )
    measurement_attitude_error = np.linalg.norm(
        wrap_to_pi(measured_states[:, 6:9] - states[:, 6:9]),
        axis=1,
    )
    estimation_attitude_error = np.linalg.norm(
        wrap_to_pi(estimated_states[:, 6:9] - states[:, 6:9]),
        axis=1,
    )
    control_delta = np.linalg.norm(np.diff(desired_controls, axis=0), axis=1)
    mean_control_delta = float(control_delta.mean()) if len(control_delta) else 0.0

    mode_name = "ekf" if use_ekf else "raw"
    data_path = OUTPUT_DIR / f"nmpc_helix_disturbance_{mode_name}_data.npz"
    np.savez(
        data_path,
        time=time,
        states=states,
        measured_states=measured_states,
        estimated_states=estimated_states,
        predicted_estimates=predicted_estimates,
        covariance_diagonal=covariance_diagonal,
        innovations=innovations,
        kalman_gain_diagonal=kalman_gain_diagonal,
        references=references,
        nmpc_controls=nmpc_controls,
        desired_controls=desired_controls,
        actuator_controls=actuator_controls,
        plant_controls=plant_controls,
        disturbances=disturbances,
        vertical_force_bias=vertical_force_bias,
        altitude_errors=altitude_errors,
        position_error=position_error,
        attitude_error=attitude_error,
        nmpc_success=nmpc_success,
        allocation_success=allocation_success,
        nominal_vehicle=np.array(
            [nominal_params.mass, nominal_params.ix, nominal_params.iy, nominal_params.iz]
        ),
        true_vehicle=np.array(
            [true_params.mass, true_params.ix, true_params.iy, true_params.iz]
        ),
        measurement_std=MEASUREMENT_STD,
        process_std=PROCESS_STD,
        disturbance_std=DISTURBANCE_STD,
        altitude_integral_gain=ALTITUDE_INTEGRAL_GAIN,
        vertical_force_bias_limit=VERTICAL_FORCE_BIAS_LIMIT,
        use_ekf=use_ekf,
        seed=args.seed,
    )

    figure_path = OUTPUT_DIR / f"nmpc_helix_disturbance_{mode_name}.png"
    figure = plot_results(
        time,
        states,
        measured_states,
        estimated_states,
        references,
        desired_controls,
        actuator_controls,
        disturbances,
        position_error,
        attitude_error,
        use_ekf,
        figure_path,
    )

    if args.no_show:
        plt.close(figure)
        print("已设置 --no-show，跳过动画窗口。")
    else:
        plt.show()
        animation_figure, animation = animate_helix_3d(
            time,
            states,
            ref_hist=references,
            step=args.animation_step,
            playback_speed=args.playback_speed,
            show=False,
        )
        if animation_figure is not None:
            animation_figure.axes[0].set_title(
                f"扰动条件下的螺旋线跟踪（{mode_name.upper()}）"
            )
            animation_figure._ani_ref = animation
            plt.show()

    mass_mismatch = 100.0 * (true_params.mass / nominal_params.mass - 1.0)
    inertia_mismatch = 100.0 * (
        np.asarray(true_params.inertia) / np.asarray(nominal_params.inertia) - 1.0
    )
    print(f"状态反馈：{'EKF 估计值' if use_ekf else '原始带噪测量值'}")
    print(
        f"鲁棒性工况：质量失配 {mass_mismatch:+.0f}%；"
        f"惯量失配 {inertia_mismatch.round().astype(int).tolist()}%"
    )
    print(f"测量噪声标准差：{MEASUREMENT_STD}")
    if use_ekf:
        print(f"EKF 单步过程噪声标准差：{PROCESS_STD}")
    print(f"有色外扰标准差：{DISTURBANCE_STD}")
    print(
        "高度积分修正："
        f"Ki={ALTITUDE_INTEGRAL_GAIN:.2f} N/(m s), "
        f"最终 fz 偏置={vertical_force_bias[-1]:.3f} N"
    )
    print(f"NMPC 求解成功率：{nmpc_success.mean() * 100.0:.1f}%")
    print(f"控制分配成功率：{allocation_success.mean() * 100.0:.1f}%")
    print(f"平均位置误差：{position_error.mean():.3f} m")
    print(f"最大位置误差：{position_error.max():.3f} m")
    print(f"终点位置误差：{position_error[-1]:.3f} m")
    print(
        "平均姿态误差："
        f"{np.rad2deg(attitude_error.mean()):.2f} deg"
    )
    print(
        "平均位置测量误差/估计误差："
        f"{measurement_position_error.mean():.4f} / "
        f"{estimation_position_error.mean():.4f} m"
    )
    print(
        "平均速度测量误差/估计误差："
        f"{measurement_velocity_error.mean():.4f} / "
        f"{estimation_velocity_error.mean():.4f} m/s"
    )
    print(
        "平均姿态测量误差/估计误差："
        f"{np.rad2deg(measurement_attitude_error.mean()):.3f} / "
        f"{np.rad2deg(estimation_attitude_error.mean()):.3f} deg"
    )
    print(f"平均控制增量 ||u[k]-u[k-1]||：{mean_control_delta:.4f}")
    print(f"结果图已保存至：{figure_path}")
    print(f"数据已保存至：{data_path}")


def noisy_measurement(state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """给真实状态叠加零均值高斯测量噪声，并保留连续偏航角表示。"""
    measured = np.asarray(state, dtype=float) + MEASUREMENT_STD * rng.normal(size=12)
    return measured


def parse_args() -> Namespace:
    """解析试验时长、随机种子、NMPC预测步长和动画参数。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--no-ekf",
        action="store_true",
        help="将原始带噪测量值直接反馈给 NMPC，不使用 EKF",
    )
    parser.add_argument("--animation-step", type=int, default=1)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration 必须为正数")
    if args.horizon <= 0:
        parser.error("--horizon 必须为正整数")
    if args.animation_step <= 0:
        parser.error("--animation-step 必须为正整数")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed 必须为正数")
    return args


def plot_results(
    time: np.ndarray,
    states: np.ndarray,
    measured_states: np.ndarray,
    estimated_states: np.ndarray,
    references: np.ndarray,
    desired_controls: np.ndarray,
    actuator_controls: np.ndarray,
    disturbances: np.ndarray,
    position_error: np.ndarray,
    attitude_error: np.ndarray,
    use_ekf: bool,
    figure_path: Path,
) -> plt.Figure:
    """绘制扰动条件下的三维轨迹、误差、控制输入、噪声和外扰。"""
    figure = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)
    control_time = time[:-1]

    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    axis_3d.plot(
        references[:, 0], references[:, 1], -references[:, 2], "k--", label="参考轨迹"
    )
    axis_3d.plot(states[:, 0], states[:, 1], -states[:, 2], label="实际轨迹")
    axis_3d.set_xlabel("x [m]")
    axis_3d.set_ylabel("y [m]")
    axis_3d.set_zlabel("高度 [m]")
    axis_3d.set_title("螺旋线跟踪")
    axis_3d.legend()

    axis_position = figure.add_subplot(grid[0, 1])
    for index, label in enumerate(("x", "y", "z")):
        axis_position.plot(time, states[:, index] - references[:, index], label=label)
    axis_position.set_title("位置跟踪误差")
    axis_position.set_xlabel("时间 [s]")
    axis_position.set_ylabel("误差 [m]")
    axis_position.grid(True)
    axis_position.legend()

    axis_norm = figure.add_subplot(grid[1, 0])
    axis_norm.plot(time, position_error, label="位置 [m]")
    axis_norm.plot(time, np.rad2deg(attitude_error), label="姿态 [deg]")
    axis_norm.set_title("跟踪误差范数")
    axis_norm.set_xlabel("时间 [s]")
    axis_norm.grid(True)
    axis_norm.legend()

    axis_control = figure.add_subplot(grid[1, 1])
    for index, label in enumerate(("fx", "fz", "tau_x", "tau_y", "tau_z")):
        axis_control.plot(
            control_time,
            desired_controls[:, index],
            "--",
            linewidth=1.0,
            label=f"{label} 指令",
        )
        axis_control.plot(
            control_time,
            actuator_controls[:, index],
            linewidth=1.0,
            label=f"{label} 实际值",
        )
    axis_control.set_title("期望虚拟输入与执行机构实际输入")
    axis_control.set_xlabel("时间 [s]")
    axis_control.grid(True)
    axis_control.legend(ncol=2, fontsize=7)

    axis_noise = figure.add_subplot(grid[2, 0])
    measurement_position_error = np.linalg.norm(
        measured_states[:, 0:3] - states[:, 0:3],
        axis=1,
    )
    estimate_position_error = np.linalg.norm(
        estimated_states[:, 0:3] - states[:, 0:3],
        axis=1,
    )
    axis_noise.plot(time, measurement_position_error, alpha=0.7, label="测量误差")
    estimate_label = "EKF 估计误差" if use_ekf else "NMPC 原始反馈误差"
    axis_noise.plot(time, estimate_position_error, label=estimate_label)
    axis_noise.set_title("位置状态估计误差")
    axis_noise.set_xlabel("时间 [s]")
    axis_noise.set_ylabel("误差范数 [m]")
    axis_noise.grid(True)
    axis_noise.legend()

    axis_disturbance = figure.add_subplot(grid[2, 1])
    for index, label in enumerate(("d_fx", "d_fz", "d_tau_x", "d_tau_y", "d_tau_z")):
        axis_disturbance.plot(control_time, disturbances[:, index], label=label)
    axis_disturbance.set_title("有色外部扰动")
    axis_disturbance.set_xlabel("时间 [s]")
    axis_disturbance.grid(True)
    axis_disturbance.legend(ncol=2, fontsize=7)

    feedback_name = "EKF 反馈" if use_ekf else "原始测量反馈"
    figure.suptitle(
        f"包含控制分配、执行机构滞后和{feedback_name}的 NMPC 鲁棒性测试",
        fontsize=15,
    )
    figure.savefig(figure_path, dpi=160)
    return figure


if __name__ == "__main__":
    main()

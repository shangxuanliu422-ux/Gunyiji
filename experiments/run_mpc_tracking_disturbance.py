"""Test helical NMPC tracking under noise, disturbances, and model mismatch."""

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

# Measurement-noise standard deviations for [position, velocity, Euler angle, angle rate].
MEASUREMENT_STD = np.array(
    [0.01] * 3 + [0.02] * 3 + [0.003] * 3 + [0.008] * 3,
    dtype=float,
)
# Colored external disturbance standard deviations for [fx, fz, tau_x, tau_y, tau_z].
DISTURBANCE_STD = np.array([0.20, 0.35, 0.008, 0.008, 0.006], dtype=float)


def main() -> None:
    """运行带模型失配、测量噪声和随机外扰的螺旋线跟踪试验。"""
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    nominal_params = DEFAULT_VEHICLE_PARAMS
    true_params = replace(
        nominal_params,
        mass=1.05 * nominal_params.mass,
        ix=1.10 * nominal_params.ix,
        iy=0.92 * nominal_params.iy,
        iz=1.12 * nominal_params.iz,
    )
    config = NMPCConfig(dt=0.1, horizon=args.horizon)
    plant = RollingWingSixDOF(true_params)
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
    references = np.zeros((steps + 1, 6), dtype=float)
    desired_controls = np.zeros((steps, 5), dtype=float)
    actuator_controls = np.zeros((steps, 5), dtype=float)
    plant_controls = np.zeros((steps, 5), dtype=float)
    disturbances = np.zeros((steps, 5), dtype=float)
    nmpc_success = np.zeros(steps, dtype=bool)
    allocation_success = np.zeros(steps, dtype=bool)

    states[0] = trajectory.state(0.0)
    measured_states[0] = noisy_measurement(states[0], rng)
    references[0] = trajectory.output(0.0)
    previous_actuator_input = actuator.virtual_input
    colored_disturbance = np.zeros(5, dtype=float)
    correlation = np.exp(-config.dt / 0.6)
    innovation_scale = np.sqrt(1.0 - correlation**2)

    for index in range(steps):
        current_time = time[index]
        horizon_reference = trajectory.horizon_outputs(
            current_time,
            config.dt,
            config.horizon,
        )
        nmpc_result = controller.solve(
            measured_states[index],
            previous_actuator_input,
            horizon_reference,
        )
        allocation_result = allocator.allocate(nmpc_result.control, dt=config.dt)
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

        desired_controls[index] = nmpc_result.control
        actuator_controls[index] = actuator_input
        plant_controls[index] = plant_input
        disturbances[index] = colored_disturbance
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
        references[index + 1] = trajectory.output(time[index + 1])
        previous_actuator_input = actuator_input

    position_error = np.linalg.norm(states[:, 0:3] - references[:, 0:3], axis=1)
    attitude_error = np.linalg.norm(
        wrap_to_pi(states[:, 6:9] - references[:, 3:6]),
        axis=1,
    )

    data_path = OUTPUT_DIR / "nmpc_helix_disturbance_data.npz"
    np.savez(
        data_path,
        time=time,
        states=states,
        measured_states=measured_states,
        references=references,
        desired_controls=desired_controls,
        actuator_controls=actuator_controls,
        plant_controls=plant_controls,
        disturbances=disturbances,
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
        disturbance_std=DISTURBANCE_STD,
        seed=args.seed,
    )

    figure_path = OUTPUT_DIR / "nmpc_helix_disturbance.png"
    figure = plot_results(
        time,
        states,
        measured_states,
        references,
        desired_controls,
        actuator_controls,
        disturbances,
        position_error,
        attitude_error,
        figure_path,
    )

    if args.no_show:
        plt.close(figure)
        print("animation window skipped because --no-show was set")
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
            animation_figure.axes[0].set_title("Helix Tracking Under Disturbances")
            animation_figure._ani_ref = animation
            plt.show()

    print("robustness test: 5% mass mismatch; inertia mismatch [+10%, -8%, +12%]")
    print(f"measurement noise std: {MEASUREMENT_STD}")
    print(f"colored disturbance std: {DISTURBANCE_STD}")
    print(f"NMPC success rate: {nmpc_success.mean() * 100.0:.1f}%")
    print(f"allocation success rate: {allocation_success.mean() * 100.0:.1f}%")
    print(f"mean position error: {position_error.mean():.3f} m")
    print(f"max position error: {position_error.max():.3f} m")
    print(f"final position error: {position_error[-1]:.3f} m")
    print(
        "mean attitude error: "
        f"{np.rad2deg(attitude_error.mean()):.2f} deg"
    )
    print(f"saved figure: {figure_path}")
    print(f"saved data: {data_path}")


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
    parser.add_argument("--animation-step", type=int, default=1)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.animation_step <= 0:
        parser.error("--animation-step must be positive")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    return args


def plot_results(
    time: np.ndarray,
    states: np.ndarray,
    measured_states: np.ndarray,
    references: np.ndarray,
    desired_controls: np.ndarray,
    actuator_controls: np.ndarray,
    disturbances: np.ndarray,
    position_error: np.ndarray,
    attitude_error: np.ndarray,
    figure_path: Path,
) -> plt.Figure:
    """绘制扰动条件下的三维轨迹、误差、控制输入、噪声和外扰。"""
    figure = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)
    control_time = time[:-1]

    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    axis_3d.plot(
        references[:, 0], references[:, 1], -references[:, 2], "k--", label="reference"
    )
    axis_3d.plot(states[:, 0], states[:, 1], -states[:, 2], label="actual")
    axis_3d.set_xlabel("x [m]")
    axis_3d.set_ylabel("y [m]")
    axis_3d.set_zlabel("height [m]")
    axis_3d.set_title("Helix Tracking")
    axis_3d.legend()

    axis_position = figure.add_subplot(grid[0, 1])
    for index, label in enumerate(("x", "y", "z")):
        axis_position.plot(time, states[:, index] - references[:, index], label=label)
    axis_position.set_title("Position Tracking Error")
    axis_position.set_xlabel("time [s]")
    axis_position.set_ylabel("error [m]")
    axis_position.grid(True)
    axis_position.legend()

    axis_norm = figure.add_subplot(grid[1, 0])
    axis_norm.plot(time, position_error, label="position [m]")
    axis_norm.plot(time, np.rad2deg(attitude_error), label="attitude [deg]")
    axis_norm.set_title("Tracking Error Norms")
    axis_norm.set_xlabel("time [s]")
    axis_norm.grid(True)
    axis_norm.legend()

    axis_control = figure.add_subplot(grid[1, 1])
    for index, label in enumerate(("fx", "fz", "tau_x", "tau_y", "tau_z")):
        axis_control.plot(
            control_time,
            desired_controls[:, index],
            "--",
            linewidth=1.0,
            label=f"{label} cmd",
        )
        axis_control.plot(
            control_time,
            actuator_controls[:, index],
            linewidth=1.0,
            label=f"{label} actual",
        )
    axis_control.set_title("Desired And Actuator Virtual Inputs")
    axis_control.set_xlabel("time [s]")
    axis_control.grid(True)
    axis_control.legend(ncol=2, fontsize=7)

    axis_noise = figure.add_subplot(grid[2, 0])
    measurement_error = measured_states - states
    axis_noise.plot(time, measurement_error[:, 0:3])
    axis_noise.set_title("Position Measurement Noise")
    axis_noise.set_xlabel("time [s]")
    axis_noise.set_ylabel("noise [m]")
    axis_noise.grid(True)

    axis_disturbance = figure.add_subplot(grid[2, 1])
    for index, label in enumerate(("d_fx", "d_fz", "d_tau_x", "d_tau_y", "d_tau_z")):
        axis_disturbance.plot(control_time, disturbances[:, index], label=label)
    axis_disturbance.set_title("Colored External Disturbances")
    axis_disturbance.set_xlabel("time [s]")
    axis_disturbance.grid(True)
    axis_disturbance.legend(ncol=2, fontsize=7)

    figure.suptitle("NMPC Robustness Test With Allocation And Actuator Lag", fontsize=15)
    figure.savefig(figure_path, dpi=160)
    return figure


if __name__ == "__main__":
    main()

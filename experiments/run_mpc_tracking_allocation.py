"""运行包含 NMPC、控制分配和执行机构的螺旋线跟踪实验。"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
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

from experiments.run_mpc_tracking import (
    animate_helix_3d,
    plot_wrapped_angle,
    set_equal_3d_axes,
    wrap_to_pi,
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
from gunyiji.dynamics import DEFAULT_VEHICLE_PARAMS, RollingWingSixDOF, rk4_step


def main() -> None:
    """
    运行带非线性控制分配和一阶执行机构的螺旋线 NMPC 跟踪实验。

    NMPC 输出五维期望虚拟输入，控制分配器求出四个滚翼的转速和偏转角指令，
    一阶执行机构模型计算实际转速与偏转角，再将实际产生的五维虚拟输入送入
    六自由度动力学。原有 ``run_mpc_tracking.py`` 不会被修改。
    """
    args = parse_args()
    params = DEFAULT_VEHICLE_PARAMS
    config = NMPCConfig(dt=0.1, horizon=args.horizon)
    model = RollingWingSixDOF(params)
    controller = CasadiNMPC(params=params, config=config)
    # 执行机构模型已经施加物理变化率限制，这里不在静态分配器中重复限速。
    allocator = NonlinearControlAllocator(
        vehicle=params,
        config=AllocationConfig(enforce_command_rate_limits=False),
    )
    actuator = FirstOrderActuatorModel(vehicle=params)
    trajectory = HelicalTrajectory(
        radius=2.0,
        angular_rate=0.5,
        climb_rate=0.2,
        center=(0.0, 0.0, -1.0),
    )

    steps = int(round(args.duration / config.dt))
    time = np.arange(steps + 1, dtype=float) * config.dt

    states = np.zeros((steps + 1, 12), dtype=float)
    refs = np.zeros((steps + 1, 6), dtype=float)
    desired_controls = np.zeros((steps, 5), dtype=float)
    allocated_controls = np.zeros((steps, 5), dtype=float)
    actual_controls = np.zeros((steps, 5), dtype=float)
    omega_commands = np.zeros((steps, 4), dtype=float)
    beta_commands = np.zeros((steps, 4), dtype=float)
    omega_actual = np.zeros((steps, 4), dtype=float)
    beta_actual = np.zeros((steps, 4), dtype=float)
    objectives = np.zeros(steps, dtype=float)
    nmpc_success = np.zeros(steps, dtype=bool)
    allocation_success = np.zeros(steps, dtype=bool)

    if args.perturbed_start:
        # 复现原螺旋线实验使用的较大初始位置、姿态和角速度偏差。
        states[0] = np.array(
            [
                0.0,
                0.0,
                -1.0,
                0.0,
                1.0,
                0.0,
                0.5,
                0.1,
                0.0,
                0.4,
                0.5,
                0.0,
            ],
            dtype=float,
        )
    else:
        # 默认从参考轨迹初始状态进入，观察正常工作区间内的分配和执行器滞后。
        states[0] = trajectory.state(0.0)
    refs[0] = trajectory.output(0.0)
    previous_applied_input = actuator.virtual_input

    for index in range(steps):
        current_time = time[index]
        horizon_reference = trajectory.horizon_outputs(
            current_time,
            config.dt,
            config.horizon,
        )
        nmpc_result = controller.solve(
            states[index],
            previous_applied_input,
            horizon_reference,
        )
        allocation_result = allocator.allocate(
            nmpc_result.control,
            dt=config.dt,
        )
        command = ActuatorState(
            omega=allocation_result.omega,
            beta=allocation_result.beta,
        )
        actual_actuator_state = actuator.step(command, config.dt)
        applied_input = actuator.virtual_input

        desired_controls[index] = nmpc_result.control
        allocated_controls[index] = allocation_result.achieved_virtual_input
        actual_controls[index] = applied_input
        omega_commands[index] = allocation_result.omega
        beta_commands[index] = allocation_result.beta
        omega_actual[index] = actual_actuator_state.omega
        beta_actual[index] = actual_actuator_state.beta
        objectives[index] = nmpc_result.objective
        nmpc_success[index] = nmpc_result.success
        allocation_success[index] = allocation_result.success

        states[index + 1] = rk4_step(
            model.derivative,
            current_time,
            states[index],
            applied_input,
            config.dt,
        )
        refs[index + 1] = trajectory.output(time[index + 1])
        previous_applied_input = applied_input

    position_error = np.linalg.norm(states[:, 0:3] - refs[:, 0:3], axis=1)
    attitude_error = np.linalg.norm(
        wrap_to_pi(states[:, 6:9] - refs[:, 3:6]),
        axis=1,
    )
    allocation_error = np.linalg.norm(
        allocated_controls - desired_controls,
        axis=1,
    )
    actuator_error = np.linalg.norm(
        actual_controls - allocated_controls,
        axis=1,
    )
    total_control_error = np.linalg.norm(
        actual_controls - desired_controls,
        axis=1,
    )

    data_path = OUTPUT_DIR / "nmpc_helix_tracking_allocation_data.npz"
    np.savez(
        data_path,
        time=time,
        states=states,
        references=refs,
        desired_controls=desired_controls,
        allocated_controls=allocated_controls,
        actual_controls=actual_controls,
        omega_commands=omega_commands,
        beta_commands=beta_commands,
        omega_actual=omega_actual,
        beta_actual=beta_actual,
        nmpc_success=nmpc_success,
        allocation_success=allocation_success,
        objectives=objectives,
        position_error=position_error,
        attitude_error=attitude_error,
        allocation_error=allocation_error,
        actuator_error=actuator_error,
    )

    figure_path = OUTPUT_DIR / "nmpc_helix_tracking_allocation.png"
    summary_figure = plot_results(
        time=time,
        states=states,
        refs=refs,
        desired_controls=desired_controls,
        allocated_controls=allocated_controls,
        actual_controls=actual_controls,
        omega_commands=omega_commands,
        beta_commands=beta_commands,
        omega_actual=omega_actual,
        beta_actual=beta_actual,
        position_error=position_error,
        allocation_error=allocation_error,
        actuator_error=actuator_error,
        figure_path=figure_path,
    )

    if args.no_show:
        plt.close(summary_figure)
        print("已设置 --no-show，跳过动画窗口。")
    else:
        plt.show()
        animation_figure, animation = animate_helix_3d(
            time,
            states,
            ref_hist=refs,
            step=args.animation_step,
            playback_speed=args.playback_speed,
            show=False,
        )
        if animation_figure is not None:
            animation_figure.axes[0].set_title(
                "含控制分配的 NMPC 螺旋线跟踪"
            )
            animation_figure._ani_ref = animation
            plt.show()

    print(f"结果图已保存至：{figure_path}")
    print(f"数据已保存至：{data_path}")
    print(f"NMPC 求解成功率：{nmpc_success.mean() * 100.0:.1f}%")
    print(
        "控制分配成功率："
        f"{allocation_success.mean() * 100.0:.1f}%"
    )
    print(f"平均位置误差：{position_error.mean():.3f} m")
    print(f"最大位置误差：{position_error.max():.3f} m")
    print(f"终点位置误差：{position_error[-1]:.3f} m")
    print(
        "平均姿态误差："
        f"{attitude_error.mean():.3f} rad "
        f"({np.rad2deg(attitude_error.mean()):.2f} deg)"
    )
    print(f"平均控制分配误差 ||u_alloc-u_cmd||：{allocation_error.mean():.4f}")
    print(f"平均执行机构滞后 ||u_actual-u_alloc||：{actuator_error.mean():.4f}")
    print(
        "平均总控制误差 ||u_actual-u_cmd||："
        f"{total_control_error.mean():.4f}"
    )
    print(f"最后一拍期望虚拟输入：{desired_controls[-1]}")
    print(f"最后一拍实际虚拟输入：{actual_controls[-1]}")
    print(f"最后一拍滚翼转速 [rad/s]：{omega_actual[-1]}")
    print(f"最后一拍偏转角 [deg]：{np.rad2deg(beta_actual[-1])}")


def parse_args() -> Namespace:
    """解析仿真时长、预测步长和动画播放参数。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=24.0,
        help="仿真时长，单位为秒",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
        help="NMPC 预测步数",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="保存结果但不打开 Matplotlib 窗口",
    )
    parser.add_argument(
        "--perturbed-start",
        action="store_true",
        help="复用 run_mpc_tracking.py 中较大的初始偏差",
    )
    parser.add_argument(
        "--animation-step",
        type=int,
        default=1,
        help="动画帧下采样步长",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="动画播放速度倍数，1.0 表示与仿真时间一致",
    )
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
    refs: np.ndarray,
    desired_controls: np.ndarray,
    allocated_controls: np.ndarray,
    actual_controls: np.ndarray,
    omega_commands: np.ndarray,
    beta_commands: np.ndarray,
    omega_actual: np.ndarray,
    beta_actual: np.ndarray,
    position_error: np.ndarray,
    allocation_error: np.ndarray,
    actuator_error: np.ndarray,
    figure_path: Path,
) -> plt.Figure:
    """绘制轨迹跟踪、控制分配和一阶执行机构响应的综合结果图。"""
    figure = plt.figure(figsize=(16, 17), constrained_layout=True)
    grid = figure.add_gridspec(4, 2)
    control_time = time[:-1]

    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    axis_3d.plot(
        refs[:, 0],
        refs[:, 1],
        -refs[:, 2],
        "k--",
        linewidth=1.5,
        label="参考轨迹",
    )
    axis_3d.plot(
        states[:, 0],
        states[:, 1],
        -states[:, 2],
        color="tab:blue",
        linewidth=1.8,
        label="实际轨迹",
    )
    axis_3d.scatter(
        states[0, 0],
        states[0, 1],
        -states[0, 2],
        color="tab:green",
        s=35,
        label="起点",
    )
    axis_3d.scatter(
        states[-1, 0],
        states[-1, 1],
        -states[-1, 2],
        color="tab:red",
        s=35,
        label="终点",
    )
    set_equal_3d_axes(axis_3d, refs, states)
    axis_3d.set_xlabel("x [m]")
    axis_3d.set_ylabel("y [m]")
    axis_3d.set_zlabel("高度 h=-z [m]")
    axis_3d.set_title("含控制分配与执行机构滞后的螺旋线跟踪")
    axis_3d.legend()

    axis_position = figure.add_subplot(grid[0, 1])
    actual_position = np.column_stack(
        (states[:, 0], states[:, 1], -states[:, 2])
    )
    reference_position = np.column_stack(
        (refs[:, 0], refs[:, 1], -refs[:, 2])
    )
    colors = ("tab:blue", "tab:orange", "tab:green")
    for index, (label, color) in enumerate(
        zip(("x", "y", "h"), colors, strict=True)
    ):
        axis_position.plot(
            time,
            reference_position[:, index],
            "--",
            color=color,
            label=f"{label} 参考值",
        )
        axis_position.plot(
            time,
            actual_position[:, index],
            "-",
            color=color,
            label=f"{label} 实际值",
        )
    axis_position.set_title("参考位置与实际位置")
    axis_position.set_xlabel("时间 [s]")
    axis_position.set_ylabel("位置 [m]")
    axis_position.grid(True)
    axis_position.legend(ncol=2, fontsize=8)

    axis_euler = figure.add_subplot(grid[1, 0])
    actual_euler = np.rad2deg(wrap_to_pi(states[:, 6:9]))
    reference_euler = np.rad2deg(wrap_to_pi(refs[:, 3:6]))
    for index, (label, color) in enumerate(
        zip(
            ("phi", "theta", "psi"),
            ("tab:red", "tab:purple", "tab:brown"),
            strict=True,
        )
    ):
        plot_wrapped_angle(
            axis_euler,
            time,
            reference_euler[:, index],
            "--",
            color,
            f"{label} 参考值",
        )
        plot_wrapped_angle(
            axis_euler,
            time,
            actual_euler[:, index],
            "-",
            color,
            f"{label} 实际值",
        )
    axis_euler.set_title("参考欧拉角与实际欧拉角")
    axis_euler.set_xlabel("时间 [s]")
    axis_euler.set_ylabel("欧拉角 [deg]")
    axis_euler.set_ylim(-185.0, 185.0)
    axis_euler.grid(True)
    axis_euler.legend(ncol=2, fontsize=8)

    axis_error = figure.add_subplot(grid[1, 1])
    axis_error.plot(time, position_error, color="tab:red", label="位置误差")
    axis_error.plot(
        control_time,
        allocation_error,
        color="tab:blue",
        label="控制分配误差",
    )
    axis_error.plot(
        control_time,
        actuator_error,
        color="tab:orange",
        label="执行机构滞后",
    )
    axis_error.set_title("跟踪误差与控制实现误差")
    axis_error.set_xlabel("时间 [s]")
    axis_error.set_ylabel("误差范数")
    axis_error.grid(True)
    axis_error.legend()

    axis_force = figure.add_subplot(grid[2, 0])
    for index, (label, color) in enumerate(
        zip(("fx", "fz"), ("tab:blue", "tab:red"), strict=True)
    ):
        axis_force.plot(
            control_time,
            desired_controls[:, index],
            "--",
            color=color,
            label=f"{label} 期望值",
        )
        axis_force.plot(
            control_time,
            allocated_controls[:, index],
            ":",
            color=color,
            label=f"{label} 分配值",
        )
        axis_force.plot(
            control_time,
            actual_controls[:, index],
            "-",
            color=color,
            label=f"{label} 实际值",
        )
    axis_force.set_title("虚拟力")
    axis_force.set_xlabel("时间 [s]")
    axis_force.set_ylabel("力 [N]")
    axis_force.grid(True)
    axis_force.legend(ncol=2, fontsize=8)

    axis_torque = figure.add_subplot(grid[2, 1])
    for offset, (label, color) in enumerate(
        zip(
            ("tau_x", "tau_y", "tau_z"),
            ("tab:red", "tab:green", "tab:blue"),
            strict=True,
        )
    ):
        index = offset + 2
        axis_torque.plot(
            control_time,
            desired_controls[:, index],
            "--",
            color=color,
            label=f"{label} 期望值",
        )
        axis_torque.plot(
            control_time,
            actual_controls[:, index],
            "-",
            color=color,
            label=f"{label} 实际值",
        )
    axis_torque.set_title("虚拟力矩")
    axis_torque.set_xlabel("时间 [s]")
    axis_torque.set_ylabel("力矩 [N m]")
    axis_torque.grid(True)
    axis_torque.legend(ncol=2, fontsize=8)

    axis_omega = figure.add_subplot(grid[3, 0])
    rotor_colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for index, color in enumerate(rotor_colors):
        axis_omega.plot(
            control_time,
            omega_commands[:, index],
            "--",
            color=color,
            linewidth=1.0,
            label=f"omega{index + 1} 指令",
        )
        axis_omega.plot(
            control_time,
            omega_actual[:, index],
            "-",
            color=color,
            linewidth=1.5,
            label=f"omega{index + 1} 实际值",
        )
    axis_omega.set_title("滚翼转速")
    axis_omega.set_xlabel("时间 [s]")
    axis_omega.set_ylabel("转速 omega [rad/s]")
    axis_omega.grid(True)
    axis_omega.legend(ncol=2, fontsize=7)

    axis_beta = figure.add_subplot(grid[3, 1])
    beta_command_deg = np.rad2deg(beta_commands)
    beta_actual_deg = np.rad2deg(beta_actual)
    for index, color in enumerate(rotor_colors):
        axis_beta.plot(
            control_time,
            beta_command_deg[:, index],
            "--",
            color=color,
            linewidth=1.0,
            label=f"beta{index + 1} 指令",
        )
        axis_beta.plot(
            control_time,
            beta_actual_deg[:, index],
            "-",
            color=color,
            linewidth=1.5,
            label=f"beta{index + 1} 实际值",
        )
    axis_beta.set_title("滚翼偏转角")
    axis_beta.set_xlabel("时间 [s]")
    axis_beta.set_ylabel("偏转角 beta [deg]")
    axis_beta.grid(True)
    axis_beta.legend(ncol=2, fontsize=7)

    figure.suptitle(
        "含非线性控制分配的 NMPC 螺旋线跟踪",
        fontsize=16,
    )
    figure.savefig(figure_path, dpi=160)
    return figure


if __name__ == "__main__":
    main()

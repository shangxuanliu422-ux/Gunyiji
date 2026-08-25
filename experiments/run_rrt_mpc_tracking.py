"""Plan an urban RRT* path, smooth it, and track it with NMPC."""

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
from matplotlib.animation import FuncAnimation

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gunyiji.control import CasadiNMPC, PathNMPCConfig
from gunyiji.dynamics import RollingWingSixDOF, make_state, rk4_step
from gunyiji.dynamics.vehicle_params import DEFAULT_VEHICLE_PARAMS
from gunyiji.environment import Building, UrbanMap2_5D, make_default_city_map
from gunyiji.planning import (
    PlannerConfig,
    RRTPlanner,
    TrajectoryGenerationConfig,
    generate_reference_trajectory,
)

START = np.array([10.0, 20.0, 15.0], dtype=float)
GOAL = np.array([190.0, 170.0, 15.0], dtype=float)


def main() -> None:
    """
    依次执行 RRT*、轨迹平滑与时间参数化、NMPC 跟踪和动画展示。
    
    输入：
        无。
    
    输出：
        无；副作用为运行实验、保存数据并显示或导出图像。
    """
    args = parse_args()
    city_map = make_default_city_map()

    planner_config = PlannerConfig(
        mode="rrt_star",
        max_iterations=args.max_iterations,
        step_size=8.0,
        goal_sample_rate=0.14,
        goal_tolerance=8.5,
        neighbor_radius=22.0,
        collision_resolution=1.0,
        height_weight=2.2,
        random_seed=args.seed,
    )
    planner_result = RRTPlanner(city_map, planner_config).plan(
        START,
        GOAL,
        snapshot_stride=100,
    )
    if not planner_result.success:
        raise RuntimeError("RRT* did not reach the goal")

    trajectory_config = TrajectoryGenerationConfig(
        dt=0.1,
        cruise_speed=args.cruise_speed,
        max_climb_speed=1.5,
        max_acceleration=2.0,
    )
    trajectory = generate_reference_trajectory(
        planner_result.path,
        city_map,
        trajectory_config,
    )
    save_reference_trajectory(trajectory)

    params = DEFAULT_VEHICLE_PARAMS
    mpc_config = PathNMPCConfig(
        dt=trajectory_config.dt,
        horizon=args.horizon,
    )
    model = RollingWingSixDOF(params)
    controller = CasadiNMPC(params=params, config=mpc_config)

    references = trajectory.outputs
    steps = len(trajectory.time) - 1
    states = np.zeros((steps + 1, 12), dtype=float)
    controls = np.zeros((steps, 5), dtype=float)
    solver_success = np.zeros(steps, dtype=bool)
    objectives = np.zeros(steps, dtype=float)

    states[0] = make_state(
        position=trajectory.position[0],
        velocity=trajectory.velocity[0],
        attitude=(0.0, 0.0, trajectory.yaw[0]),
        attitude_rate=(0.0, 0.0, 0.0),
    )
    previous_input = model.hover_input()

    for index in range(steps):
        horizon_reference = trajectory.horizon_outputs(index, mpc_config.horizon)
        result = controller.solve(
            states[index],
            previous_input,
            horizon_reference,
        )
        controls[index] = result.control
        solver_success[index] = result.success
        objectives[index] = result.objective
        states[index + 1] = rk4_step(
            model.derivative,
            trajectory.time[index],
            states[index],
            controls[index],
            mpc_config.dt,
        )
        previous_input = controls[index]

    position_error = np.linalg.norm(states[:, 0:3] - trajectory.position, axis=1)
    velocity_error = np.linalg.norm(states[:, 3:6] - trajectory.velocity, axis=1)
    yaw_error = wrap_to_pi(states[:, 8] - trajectory.yaw)
    save_tracking_result(
        trajectory.time,
        states,
        controls,
        solver_success,
        objectives,
    )

    figure_path = OUTPUT_DIR / "rrt_star_nmpc_tracking.png"
    figure = plot_results(
        city_map,
        trajectory,
        states,
        controls,
        position_error,
        velocity_error,
        yaw_error,
        figure_path,
    )
    if args.no_show:
        plt.close(figure)
    else:
        animate_path_tracking_3d(
            trajectory.time,
            trajectory.position_planning,
            states,
            step=args.animation_step,
            playback_speed=args.playback_speed,
            show=True,
        )

    print(f"raw RRT* points: {len(trajectory.raw_path)}")
    print(f"shortcut points: {len(trajectory.shortcut_path)}")
    print(f"spline samples: {len(trajectory.smooth_path)}")
    print(f"spline smoothing strength: {trajectory.spline_smoothing:.3f}")
    print(f"spline control spacing: {trajectory.spline_control_spacing:.2f} m")
    print(f"trajectory duration: {trajectory.time[-1]:.2f} s")
    print(f"maximum reference speed: {trajectory.speed.max():.2f} m/s")
    print(
        "maximum reference acceleration: "
        f"{np.linalg.norm(trajectory.acceleration, axis=1).max():.2f} m/s^2"
    )
    print(f"solver success rate: {solver_success.mean() * 100.0:.1f}%")
    print(f"mean position error: {position_error.mean():.3f} m")
    print(f"max position error: {position_error.max():.3f} m")
    print(f"final position error: {position_error[-1]:.3f} m")
    print(f"mean velocity error: {velocity_error.mean():.3f} m/s")
    print(f"mean yaw error: {np.rad2deg(np.abs(yaw_error)).mean():.2f} deg")
    print(f"saved reference: {OUTPUT_DIR / 'rrt_star_reference_trajectory.npz'}")
    print(f"saved tracking result: {OUTPUT_DIR / 'rrt_star_nmpc_tracking_data.npz'}")
    print(f"saved figure: {figure_path}")


def parse_args() -> Namespace:
    """
    解析实验脚本的命令行参数并返回参数对象。
    
    输入：
        无。
    
    输出：
        解析后的 argparse.Namespace 参数对象。
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--max-iterations", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cruise-speed", type=float, default=7.0)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument(
        "--animation-step",
        type=int,
        default=1,
        help="downsample animation frames; 1 keeps every NMPC sample",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="animation speed multiplier; 1.0 follows simulation time",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def save_reference_trajectory(trajectory) -> None:
    """
    将时标参考轨迹保存为 NPZ 和 CSV 文件。
    
    输入：
        trajectory: 生成的时标参考轨迹对象。
    
    输出：
        无；参考轨迹写入输出目录。
    """
    data_path = OUTPUT_DIR / "rrt_star_reference_trajectory.npz"
    csv_path = OUTPUT_DIR / "rrt_star_reference_trajectory.csv"

    np.savez_compressed(
        data_path,
        time=trajectory.time,
        raw_path=trajectory.raw_path,
        shortcut_path=trajectory.shortcut_path,
        smooth_path=trajectory.smooth_path,
        position_planning=trajectory.position_planning,
        position=trajectory.position,
        velocity=trajectory.velocity,
        acceleration=trajectory.acceleration,
        yaw=trajectory.yaw,
        yaw_rate=trajectory.yaw_rate,
        speed=trajectory.speed,
    )
    table = np.column_stack(
        (
            trajectory.time,
            trajectory.position,
            trajectory.velocity,
            trajectory.acceleration,
            trajectory.yaw,
            trajectory.yaw_rate,
        )
    )
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        comments="",
        header=(
            "time_s,x_m,y_m,z_down_m,vx_mps,vy_mps,vz_mps,"
            "ax_mps2,ay_mps2,az_mps2,yaw_rad,yaw_rate_radps"
        ),
    )


def save_tracking_result(
    time: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    solver_success: np.ndarray,
    objectives: np.ndarray,
) -> None:
    """
    将 NMPC 状态、控制量和求解信息保存为 NPZ 和 CSV。
    
    输入：
        time: 时间序列，单位为秒。
        states: 飞行器状态历史数组。
        controls: 五维虚拟控制输入历史。
        solver_success: 各控制周期求解器是否成功的布尔历史。
        objectives: 各控制周期 NMPC 目标函数值历史。
    
    输出：
        无；跟踪历史写入输出目录。
    """
    data_path = OUTPUT_DIR / "rrt_star_nmpc_tracking_data.npz"
    csv_path = OUTPUT_DIR / "rrt_star_nmpc_tracking_data.csv"
    control_history = np.vstack((controls, controls[-1]))
    success_history = np.append(solver_success, solver_success[-1])
    objective_history = np.append(objectives, objectives[-1])

    np.savez_compressed(
        data_path,
        time=time,
        states=states,
        controls=controls,
        solver_success=solver_success,
        objectives=objectives,
    )
    table = np.column_stack(
        (
            time,
            states,
            control_history,
            success_history.astype(float),
            objective_history,
        )
    )
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        comments="",
        header=(
            "time_s,x_m,y_m,z_down_m,vx_mps,vy_mps,vz_mps,"
            "phi_rad,theta_rad,psi_rad,phi_rate_radps,theta_rate_radps,"
            "psi_rate_radps,fx_n,fz_n,tau_x_nm,tau_y_nm,tau_z_nm,"
            "solver_success,objective"
        ),
    )


def plot_results(
    city_map: UrbanMap2_5D,
    trajectory,
    states: np.ndarray,
    controls: np.ndarray,
    position_error: np.ndarray,
    velocity_error: np.ndarray,
    yaw_error: np.ndarray,
    figure_path: Path,
) -> plt.Figure:
    """
    绘制参考轨迹、实际状态、误差和控制量的静态汇总图。
    
    输入：
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        trajectory: 生成的时标参考轨迹对象。
        states: 飞行器状态历史数组。
        controls: 五维虚拟控制输入历史。
        position_error: 位置跟踪误差时间序列。
        velocity_error: 速度跟踪误差时间序列。
        yaw_error: 偏航角跟踪误差时间序列。
        figure_path: 静态结果图保存路径。
    
    输出：
        生成并保存的 Matplotlib Figure 对象。
    """
    time = trajectory.time
    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = fig.add_gridspec(3, 2)

    ax_3d = fig.add_subplot(grid[0:2, 0], projection="3d")
    draw_city(ax_3d, city_map)
    ax_3d.plot(
        trajectory.raw_path[:, 0],
        trajectory.raw_path[:, 1],
        trajectory.raw_path[:, 2],
        "o--",
        color="0.45",
        linewidth=1.0,
        markersize=3,
        label="RRT* path",
    )
    ax_3d.plot(
        trajectory.shortcut_path[:, 0],
        trajectory.shortcut_path[:, 1],
        trajectory.shortcut_path[:, 2],
        "o-",
        color="tab:orange",
        linewidth=1.4,
        markersize=4,
        label="shortcut",
    )
    ax_3d.plot(
        trajectory.position_planning[:, 0],
        trajectory.position_planning[:, 1],
        trajectory.position_planning[:, 2],
        color="tab:red",
        linewidth=2.5,
        label="smooth reference",
    )
    ax_3d.plot(
        states[:, 0],
        states[:, 1],
        -states[:, 2],
        color="tab:blue",
        linewidth=2.0,
        label="NMPC",
    )
    ax_3d.set_xlim(city_map.bounds.x_min, city_map.bounds.x_max)
    ax_3d.set_ylim(city_map.bounds.y_min, city_map.bounds.y_max)
    ax_3d.set_zlim(0.0, city_map.bounds.h_max)
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("height h=-z [m]")
    ax_3d.set_title("RRT* Path Processing And NMPC Tracking")
    ax_3d.set_box_aspect((200, 200, 80))
    ax_3d.view_init(elev=34.0, azim=-62.0)
    ax_3d.legend(loc="upper left", fontsize=8)

    ax_position = fig.add_subplot(grid[0, 1])
    actual_position = states[:, 0:3].copy()
    actual_position[:, 2] *= -1.0
    reference_position = trajectory.position_planning
    for index, (label, color) in enumerate(
        zip(("x", "y", "h"), ("tab:blue", "tab:orange", "tab:green"), strict=True)
    ):
        ax_position.plot(time, reference_position[:, index], "--", color=color, label=f"{label} ref")
        ax_position.plot(time, actual_position[:, index], "-", color=color, label=f"{label} actual")
    ax_position.set_title("Position")
    ax_position.set_xlabel("time [s]")
    ax_position.set_ylabel("position [m]")
    ax_position.grid(True)
    ax_position.legend(ncol=2, fontsize=8)

    ax_velocity = fig.add_subplot(grid[1, 1])
    for index, (label, color) in enumerate(
        zip(("vx", "vy", "vz"), ("tab:blue", "tab:orange", "tab:green"), strict=True)
    ):
        ax_velocity.plot(time, trajectory.velocity[:, index], "--", color=color, label=f"{label} ref")
        ax_velocity.plot(time, states[:, 3 + index], "-", color=color, label=f"{label} actual")
    ax_velocity.set_title("Velocity In Dynamics Coordinates")
    ax_velocity.set_xlabel("time [s]")
    ax_velocity.set_ylabel("velocity [m/s]")
    ax_velocity.grid(True)
    ax_velocity.legend(ncol=2, fontsize=8)

    ax_error = fig.add_subplot(grid[2, 0])
    ax_error.plot(time, position_error, label="position [m]")
    ax_error.plot(time, velocity_error, label="velocity [m/s]")
    ax_error.plot(time, np.rad2deg(np.abs(yaw_error)), label="|yaw| [deg]")
    ax_error.set_title("Tracking Errors")
    ax_error.set_xlabel("time [s]")
    ax_error.grid(True)
    ax_error.legend()

    ax_control = fig.add_subplot(grid[2, 1])
    for index, label in enumerate(("fx", "fz", "tau_x", "tau_y", "tau_z")):
        ax_control.plot(time[:-1], controls[:, index], label=label)
    ax_control.set_title("Virtual Inputs")
    ax_control.set_xlabel("time [s]")
    ax_control.grid(True)
    ax_control.legend(ncol=2, fontsize=8)

    fig.savefig(figure_path, dpi=160)
    return fig


def draw_city(ax: plt.Axes, city_map: UrbanMap2_5D) -> None:
    """
    在三维坐标轴中绘制城市内全部建筑物。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
    
    输出：
        无。
    """
    for building in city_map.buildings:
        draw_building(ax, building)


def draw_building(ax: plt.Axes, building: Building) -> None:
    """
    在三维坐标轴中绘制单个不透明建筑物。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        building: 需要绘制或处理的建筑物对象。
    
    输出：
        无。
    """
    width, depth, height = building.size
    ax.bar3d(
        building.xmin,
        building.ymin,
        0.0,
        width,
        depth,
        height,
        color="lightgray",
        edgecolor="0.3",
        linewidth=0.4,
        alpha=1.0,
        shade=True,
    )


def animate_path_tracking_3d(
    time: np.ndarray,
    reference_path: np.ndarray,
    states: np.ndarray,
    step: int = 1,
    playback_speed: float = 1.0,
    show: bool = True,
) -> tuple[plt.Figure | None, FuncAnimation | None]:
    """
    动画展示飞机沿最终平滑参考轨迹飞向终点的过程。
    
    输入：
        time: 时间序列，单位为秒。
        reference_path: 最终平滑参考路径点数组。
        states: 飞行器状态历史数组。
        step: 动画帧下采样步长。
        playback_speed: 动画相对真实仿真时间的播放倍率。
        show: 是否在函数内部调用 Matplotlib 显示窗口。
    
    输出：
        Matplotlib 图对象与 FuncAnimation 动画对象。
    """

    if len(time) == 0 or len(reference_path) == 0 or len(states) == 0:
        return None, None
    if len(time) != len(reference_path) or len(time) != len(states):
        raise ValueError("time, reference_path and states must have equal lengths")
    if playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive")

    sample_indices = np.arange(0, len(time), max(1, step), dtype=int)
    if sample_indices[-1] != len(time) - 1:
        sample_indices = np.append(sample_indices, len(time) - 1)

    sampled_time = time[sample_indices]
    if len(sampled_time) > 1:
        frame_dt = float(np.median(np.diff(sampled_time)))
    else:
        frame_dt = 0.1
    interval_ms = max(1, int(round(1000.0 * frame_dt / playback_speed)))

    reference = np.asarray(reference_path, dtype=float)
    actual = np.column_stack(
        (
            states[sample_indices, 0],
            states[sample_indices, 1],
            -states[sample_indices, 2],
        )
    )
    euler = states[sample_indices, 6:9]
    velocity = states[sample_indices, 3:6]

    all_points = np.vstack((reference, actual))
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    horizontal_span = max(float(span[0]), float(span[1]), 1.0)
    horizontal_padding = 0.06 * horizontal_span
    vertical_padding = max(2.0, 0.18 * float(span[2]))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("RRT* Smoothed Path NMPC Tracking")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("height h=-z [m]")
    ax.grid(True)
    ax.view_init(elev=30.0, azim=-60.0)

    ax.set_xlim(minimum[0] - horizontal_padding, maximum[0] + horizontal_padding)
    ax.set_ylim(minimum[1] - horizontal_padding, maximum[1] + horizontal_padding)
    ax.set_zlim(
        max(0.0, minimum[2] - vertical_padding),
        maximum[2] + vertical_padding,
    )
    ax.set_box_aspect(
        (
            span[0] + 2.0 * horizontal_padding,
            span[1] + 2.0 * horizontal_padding,
            max(0.32 * horizontal_span, span[2] + 2.0 * vertical_padding),
        )
    )

    ax.plot(
        reference[:, 0],
        reference[:, 1],
        reference[:, 2],
        "--",
        color="tab:red",
        linewidth=2.2,
        label="Final smoothed reference",
    )
    actual_trail, = ax.plot(
        [],
        [],
        [],
        color="tab:blue",
        linewidth=2.3,
        label="NMPC actual trail",
    )
    vehicle_dot = ax.scatter(
        actual[0, 0],
        actual[0, 1],
        actual[0, 2],
        color="black",
        s=55,
        depthshade=False,
        label="Vehicle",
    )

    body_axis_length = max(3.0, 0.035 * horizontal_span)
    body_axis_quivers = draw_body_axes(
        ax,
        actual[0],
        euler[0],
        body_axis_length,
    )
    status_text = ax.text2D(0.025, 0.94, "", transform=ax.transAxes)
    ax.text2D(
        0.025,
        0.03,
        "Body axes: x red, y green, z blue",
        transform=ax.transAxes,
        color="0.25",
    )
    ax.legend(loc="upper right")

    def update(frame_index: int) -> tuple[object, ...]:
        """
        根据当前动画帧更新轨迹、飞行器位置、机体坐标轴和状态文字。
        
        输入：
            frame_index: 当前动画帧索引。
        
        输出：
            本帧更新过的 Matplotlib 图形对象元组。
        """
        nonlocal body_axis_quivers, vehicle_dot

        point = actual[frame_index]
        trail = actual[: frame_index + 1]
        actual_trail.set_data(trail[:, 0], trail[:, 1])
        actual_trail.set_3d_properties(trail[:, 2])

        vehicle_dot.remove()
        vehicle_dot = ax.scatter(
            point[0],
            point[1],
            point[2],
            color="black",
            s=55,
            depthshade=False,
        )

        for quiver in body_axis_quivers:
            quiver.remove()
        body_axis_quivers = draw_body_axes(
            ax,
            point,
            euler[frame_index],
            body_axis_length,
        )

        speed = float(np.linalg.norm(velocity[frame_index]))
        status_text.set_text(
            f"t = {sampled_time[frame_index]:.1f} s\n"
            f"speed = {speed:.2f} m/s\n"
            f"position = ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}) m"
        )
        return actual_trail, vehicle_dot, status_text, *body_axis_quivers

    animation = FuncAnimation(
        fig,
        update,
        frames=len(sample_indices),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    fig._animation_reference = animation
    plt.tight_layout()
    if show:
        plt.show()
    return fig, animation


def draw_body_axes(
    ax: plt.Axes,
    origin: np.ndarray,
    euler: np.ndarray,
    length: float,
) -> list[object]:
    """
    在飞机当前位置绘制机体 x、y、z 三根短箭头。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        origin: 机体坐标轴箭头的三维起点。
        euler: 滚转、俯仰、偏航欧拉角，单位为弧度。
        length: 箭头长度或曲线长度。
    
    输出：
        三个 Matplotlib 三维箭头对象组成的列表。
    """

    axes = body_axes_display(euler)
    colors = ("tab:red", "tab:green", "tab:blue")
    arrows = []
    for axis, color in zip(axes.T, colors, strict=True):
        arrows.append(
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                length * axis[0],
                length * axis[1],
                length * axis[2],
                color=color,
                linewidth=2.0,
                arrow_length_ratio=0.28,
            )
        )
    return arrows


def body_axes_display(euler: np.ndarray) -> np.ndarray:
    """
    将机体坐标轴旋转到 x、y、高度显示坐标系。
    
    输入：
        euler: 滚转、俯仰、偏航欧拉角，单位为弧度。
    
    输出：
        3×3 矩阵，每一列是一根机体轴在显示坐标系中的方向。
    """

    phi, theta, psi = np.asarray(euler, dtype=float)
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    body_to_earth = np.array(
        [
            [
                c_psi * c_theta,
                c_psi * s_theta * s_phi - s_psi * c_phi,
                c_psi * s_theta * c_phi + s_psi * s_phi,
            ],
            [
                s_psi * c_theta,
                s_psi * s_theta * s_phi + c_psi * c_phi,
                s_psi * s_theta * c_phi - c_psi * s_phi,
            ],
            [-s_theta, c_theta * s_phi, c_theta * c_phi],
        ],
        dtype=float,
    )
    body_to_display = body_to_earth.copy()
    body_to_display[2, :] *= -1.0
    return body_to_display


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """
    将角度逐元素折返到 [-π, π) 区间。
    
    输入：
        angle: 待折返的角度标量或数组。
    
    输出：
        折返到 [-π, π) 的角度数组。
    """
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


if __name__ == "__main__":
    print("Running...")
    main()

"""Run virtual-input NMPC tracking on an ascending helical trajectory."""

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
from matplotlib.animation import FuncAnimation
import numpy as np

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gunyiji.control import CasadiNMPC, HelicalTrajectory, NMPCConfig
from gunyiji.dynamics import (
    DEFAULT_VEHICLE_PARAMS,
    RollingWingSixDOF,
    rk4_step,
)


def main() -> None:
    """
    运行螺旋上升参考轨迹的 NMPC 闭环跟踪实验。
    
    输入：
        无。
    
    输出：
        无；副作用为运行实验、保存数据并显示或导出图像。
    """
    args = parse_args()
    params = DEFAULT_VEHICLE_PARAMS
    config = NMPCConfig(dt=0.1, horizon=60)
    model = RollingWingSixDOF(params)
    controller = CasadiNMPC(params=params, config=config)
    trajectory = HelicalTrajectory(
        radius=2.0,
        angular_rate=0.5,
        climb_rate=0.2,
        center=(0.0, 0.0, -1.0),
    )

    duration = 24.0
    steps = int(duration / config.dt)
    time = np.arange(steps + 1, dtype=float) * config.dt # time是一个从0到24的数组，间隔为0.1

    states = np.zeros((steps + 1, 12), dtype=float)
    refs = np.zeros((steps + 1, 6), dtype=float)
    controls = np.zeros((steps, 5), dtype=float)
    objectives = np.zeros(steps, dtype=float)
    success = np.zeros(steps, dtype=bool)

    """ states[0] = trajectory.state(0.0) """
    """ position: Sequence[float] = (0.0, 0.0, 0.0),
    velocity: Sequence[float] = (0.0, 0.0, 0.0),
    attitude: Sequence[float] = (0.0, 0.0, 0.0),
    attitude_rate: Sequence[float] = (0.0, 0.0, 0.0), """
    states[0] = np.array([0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.5, 0.1, 0.0, 0.4, 0.5, 0.0], dtype=float)
    previous_input = model.hover_input()
    refs[0] = trajectory.output(0.0) # 即0.0s时的六个状态

    for k in range(steps):
        t = time[k]
        horizon_ref = trajectory.horizon_outputs(t, config.dt, config.horizon) # t是这一段预测域的起始时间
        result = controller.solve(states[k], previous_input, horizon_ref)

        controls[k] = result.control
        objectives[k] = result.objective
        success[k] = result.success

        states[k + 1] = rk4_step(model.derivative, t, states[k], controls[k], config.dt)
        refs[k + 1] = trajectory.output(time[k + 1])
        previous_input = controls[k]

    position_error = np.linalg.norm(states[:, 0:3] - refs[:, 0:3], axis=1)
    attitude_error = np.linalg.norm(wrap_to_pi(states[:, 6:9] - refs[:, 3:6]), axis=1)
    control_delta = np.linalg.norm(np.diff(controls, axis=0), axis=1)

    figure_path = OUTPUT_DIR / "nmpc_helix_tracking.png"

    summary_fig = plot_results(
        time,
        states,
        refs,
        controls,
        position_error,
        figure_path,
        show=not args.no_show,
    )
    if args.no_show:
        print("animation window skipped because --no-show was set")
        if summary_fig is not None:
            plt.close(summary_fig)
    else:
        animate_helix_3d(
            time,
            states,
            ref_hist=refs,
            step=args.animation_step,
            playback_speed=args.playback_speed,
            show=True,
        )

    print(f"saved figure: {figure_path}")
    print(f"solver success rate: {success.mean() * 100.0:.1f}%")
    print(f"mean position error: {position_error.mean():.3f} m")
    print(f"max position error: {position_error.max():.3f} m")
    print(f"final position error: {position_error[-1]:.3f} m")
    print(
        "mean attitude error: "
        f"{attitude_error.mean():.3f} rad ({np.rad2deg(attitude_error.mean()):.2f} deg)"
    )
    print(f"mean ||u[k]-u[k-1]||: {control_delta.mean():.3f}" if len(control_delta) else "mean ||u[k]-u[k-1]||: 0.000")
    print(f"last control [fx, fz, tau_x, tau_y, tau_z]: {controls[-1]}")


def parse_args() -> Namespace: # Namespace是argparse模块中的一个类，用于存储命令行参数的解析结果
    """
    解析实验脚本的命令行参数并返回参数对象。
    
    输入：
        无。
    
    输出：
        解析后的 argparse.Namespace 参数对象。
    """
    parser = ArgumentParser(description=__doc__) # 创建一个ArgumentParser对象，用于解析命令行参数，并将脚本的文档字符串作为描述信息
    parser.add_argument(
        "--no-show",
        action="store_true", # 如果命令行传了"--no-show"就会将args.no_show设置为True，否则为False
        help="skip the Matplotlib animation window",
    )
    parser.add_argument(
        "--animation-step",
        type=int,
        default=1, # 可以传一个整数，不传的话默认是1
        help="downsample animation frames to keep playback smooth",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="animation playback speed multiplier; 1.0 means real simulation time",
    )
    return parser.parse_args()


def plot_results(
    time: np.ndarray,
    states: np.ndarray,
    refs: np.ndarray,
    controls: np.ndarray,
    position_error: np.ndarray,
    figure_path: Path,
    show: bool = True,
) -> plt.Figure:
    """
    绘制螺旋轨迹的位置、姿态、误差和控制输入汇总图。
    
    输入：
        time: 时间序列，单位为秒。
        states: 飞行器状态历史数组。
        refs: 参考输出历史数组。
        controls: 五维虚拟控制输入历史。
        position_error: 位置跟踪误差时间序列。
        figure_path: 静态结果图保存路径。
        show: 是否在函数内部调用 Matplotlib 显示窗口。
    
    输出：
        生成并保存的 Matplotlib Figure 对象。
    """
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 2)

    ax_3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_3d.plot(refs[:, 0], refs[:, 1], -refs[:, 2], "k--", label="reference")
    ax_3d.plot(states[:, 0], states[:, 1], -states[:, 2], "tab:blue", label="NMPC")
    ax_3d.scatter(states[0, 0], states[0, 1], -states[0, 2], color="tab:green", s=35, label="start")
    ax_3d.scatter(states[-1, 0], states[-1, 1], -states[-1, 2], color="tab:red", s=35, label="end")
    set_equal_3d_axes(ax_3d, refs, states)
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("height h=-z [m]")
    ax_3d.set_title("3D Helix Tracking")
    ax_3d.legend()

    ax_position = fig.add_subplot(grid[0, 1])
    position_labels = ("x", "y", "h=-z")
    position_colors = ("tab:blue", "tab:orange", "tab:green")
    actual_position = np.column_stack((states[:, 0], states[:, 1], -states[:, 2]))
    reference_position = np.column_stack((refs[:, 0], refs[:, 1], -refs[:, 2]))
    for idx, (label, color) in enumerate(zip(position_labels, position_colors, strict=True)):
        ax_position.plot(time, reference_position[:, idx], "--", color=color, label=f"{label} ref")
        ax_position.plot(time, actual_position[:, idx], "-", color=color, label=f"{label} actual")
    ax_position.set_xlabel("time [s]")
    ax_position.set_ylabel("position [m]")
    ax_position.set_title("Reference And Actual Position")
    ax_position.ticklabel_format(useOffset=False)
    ax_position.grid(True)
    ax_position.legend(ncol=2, fontsize=8)

    ax_euler = fig.add_subplot(grid[1, 0])
    euler_labels = ("phi", "theta", "psi")
    euler_colors = ("tab:red", "tab:purple", "tab:brown")
    actual_euler_deg = np.rad2deg(wrap_to_pi(states[:, 6:9]))
    reference_euler_deg = np.rad2deg(wrap_to_pi(refs[:, 3:6]))
    for idx, (label, color) in enumerate(zip(euler_labels, euler_colors, strict=True)):
        plot_wrapped_angle(
            ax_euler,
            time,
            reference_euler_deg[:, idx],
            "--",
            color,
            f"{label} ref",
        )
        plot_wrapped_angle(
            ax_euler,
            time,
            actual_euler_deg[:, idx],
            "-",
            color,
            f"{label} actual",
        )
        ax_euler.set_xlabel("time [s]")
        ax_euler.set_ylabel("Euler angle [deg]")
        ax_euler.set_title("Reference And Actual Euler Angles (Wrapped)")
        ax_euler.set_ylim(-185, 185)
        ax_euler.ticklabel_format(useOffset=False)
        ax_euler.grid(True)
        ax_euler.legend(ncol=2, fontsize=8)

        ax_error = fig.add_subplot(grid[1, 1])
        ax_error.plot(time, position_error, "tab:red")
        ax_error.set_xlabel("time [s]")
        ax_error.set_ylabel("position error [m]")
        ax_error.set_title("Tracking Error")
        ax_error.grid(True)

    ax_control = fig.add_subplot(grid[2, :])
    control_time = time[:-1]
    labels = ("fx", "fz", "tau_x", "tau_y", "tau_z")
    for idx, label in enumerate(labels):
        ax_control.plot(control_time, controls[:, idx], label=label)
    ax_control.set_xlabel("time [s]")
    ax_control.set_title("Virtual Inputs")
    ax_control.grid(True)
    ax_control.legend(ncol=2)

    fig.savefig(figure_path, dpi=160)
    if not show:
        plt.close(fig)
    return fig


def animate_helix_3d(
    t_vec: np.ndarray,
    state_hist: np.ndarray,
    ref_hist: np.ndarray | None = None,
    step: int = 2,
    playback_speed: float = 1.0,
    show: bool = True,
) -> tuple[plt.Figure | None, FuncAnimation | None]:
    """
    动画展示 NMPC 对螺旋上升轨迹的跟踪过程。
    
    输入：
        t_vec: 动画使用的时间序列，单位为秒。
        state_hist: 飞行器状态历史数组。
        ref_hist: 参考输出历史数组。
        step: 动画帧下采样步长。
        playback_speed: 动画相对真实仿真时间的播放倍率。
        show: 是否在函数内部调用 Matplotlib 显示窗口。
    
    输出：
        Matplotlib 图对象与 FuncAnimation 动画对象。
    """

    if len(t_vec) == 0 or len(state_hist) == 0:
        return None, None
    if playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive")

    sample_idx = np.arange(0, len(t_vec), max(1, step))
    if sample_idx[-1] != len(t_vec) - 1:
        sample_idx = np.append(sample_idx, len(t_vec) - 1)

    t_s = t_vec[sample_idx]
    if len(t_s) > 1:
        frame_dt = float(np.median(np.diff(t_s)))
    else:
        frame_dt = 0.1
    interval_ms = max(1, int(round(1000.0 * frame_dt / playback_speed)))

    pos_s = np.column_stack(
        (
            state_hist[sample_idx, 0],
            state_hist[sample_idx, 1],
            -state_hist[sample_idx, 2],
        )
    )
    euler_s = state_hist[sample_idx, 6:9]

    all_pts = pos_s
    ref_s = None
    if ref_hist is not None and len(ref_hist) == len(t_vec):
        ref_s = np.column_stack(
            (
                ref_hist[sample_idx, 0],
                ref_hist[sample_idx, 1],
                -ref_hist[sample_idx, 2],
            )
        )
        all_pts = np.vstack((all_pts, ref_s))

    xyz_min = all_pts.min(axis=0)
    xyz_max = all_pts.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    radius = 0.55 * np.max(xyz_max - xyz_min) + 0.2
    radius = max(float(radius), 0.5)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("NMPC Helix Tracking")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Height h=-z (m)")
    ax.grid(True)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_box_aspect([1, 1, 1])

    if ref_s is not None:
        ax.plot(
            ref_s[:, 0],
            ref_s[:, 1],
            ref_s[:, 2],
            "--",
            color="tab:green",
            lw=1.5,
            label="Desired Helix",
        )

    tracked_line, = ax.plot([], [], [], color="tab:blue", linewidth=2.0, label="NMPC")
    vehicle_dot = ax.scatter([], [], [], color="tab:red", s=45, label="Vehicle")
    tracked_trail, = ax.plot([], [], [], color="tab:orange", linewidth=1.8, label="Trail")
    body_axis_length = 0.16 * radius
    body_axis_quivers = draw_body_axes(ax, pos_s[0], euler_s[0], body_axis_length)
    time_text = ax.text2D(0.03, 0.94, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(frame_idx: int) -> tuple[object, ...]:
        """
        根据当前动画帧更新轨迹、飞行器位置、机体坐标轴和状态文字。
        
        输入：
            frame_idx: 当前动画帧索引。
        
        输出：
            本帧更新过的 Matplotlib 图形对象元组。
        """
        nonlocal body_axis_quivers, vehicle_dot
        point = pos_s[frame_idx]
        trail = pos_s[: frame_idx + 1]

        tracked_line.set_data(trail[:, 0], trail[:, 1])
        tracked_line.set_3d_properties(trail[:, 2])
        tracked_trail.set_data(trail[:, 0], trail[:, 1])
        tracked_trail.set_3d_properties(trail[:, 2])

        vehicle_dot.remove()
        vehicle_dot = ax.scatter(point[0], point[1], point[2], color="tab:red", s=45)
        for quiver in body_axis_quivers:
            quiver.remove()
        body_axis_quivers = draw_body_axes(
            ax,
            point,
            euler_s[frame_idx],
            body_axis_length,
        )
        time_text.set_text(f"t = {t_s[frame_idx]:.2f} s")
        return tracked_line, tracked_trail, vehicle_dot, time_text, *body_axis_quivers

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t_s),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    fig._ani_ref = ani
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ani


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """
    将角度逐元素折返到 [-π, π) 区间。
    
    输入：
        angle: 待折返的角度标量或数组。
    
    输出：
        折返到 [-π, π) 的角度数组。
    """

    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def plot_wrapped_angle(
    ax: plt.Axes,
    time: np.ndarray,
    angle_deg: np.ndarray,
    linestyle: str,
    color: str,
    label: str,
) -> None:
    """
    绘制周期角度并在跨越 ±180 度处断开曲线。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        time: 时间序列，单位为秒。
        angle_deg: 以度为单位的周期角度序列。
        linestyle: Matplotlib 线型字符串。
        color: 绘图颜色。
        label: 图例或悬停信息标签。
    
    输出：
        无。
    """

    y = np.asarray(angle_deg, dtype=float).copy()
    if len(y) > 1:
        jumps = np.abs(np.diff(y)) > 180.0
        y[np.where(jumps)[0] + 1] = np.nan
    ax.plot(time, y, linestyle=linestyle, color=color, label=label)


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
    labels = ("body x", "body y", "body z")
    quivers = []
    for axis, color, label in zip(axes.T, colors, labels, strict=True):
        quivers.append(
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                length * axis[0],
                length * axis[1],
                length * axis[2],
                color=color,
                linewidth=1.8,
                arrow_length_ratio=0.35,
                label=label,
            )
        )
    return quivers


def body_axes_display(euler: np.ndarray) -> np.ndarray:
    """
    将机体坐标轴旋转到 x、y、高度显示坐标系。
    
    输入：
        euler: 滚转、俯仰、偏航欧拉角，单位为弧度。
    
    输出：
        3×3 矩阵，每一列是一根机体轴在显示坐标系中的方向。
    """

    phi, theta, psi = euler
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    rotation_body_to_earth = np.array(
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
    display_axes = rotation_body_to_earth.copy()
    display_axes[2, :] *= -1.0
    return display_axes


def set_equal_3d_axes(ax: plt.Axes, refs: np.ndarray, states: np.ndarray) -> None:
    """
    依据参考与实际轨迹设置等尺度三维显示范围。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        refs: 参考输出历史数组。
        states: 飞行器状态历史数组。
    
    输出：
        无。
    """
    points = np.vstack((refs[:, 0:3], states[:, 0:3])).copy()
    points[:, 2] *= -1.0

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    radius = max(radius, 0.5)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)


if __name__ == "__main__":
    print("Running...")
    main()

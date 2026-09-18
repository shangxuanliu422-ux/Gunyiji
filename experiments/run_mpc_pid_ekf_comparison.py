"""对比串级 MPC/PID 使用 EKF 反馈和原始带噪反馈时的控制效果。"""

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

from experiments.run_mpc_tracking import wrap_to_pi
from experiments.run_mpc_tracking_cascade_pid import (
    INNER_DT,
    OUTER_DT,
    SimulationResult,
    calculate_metrics,
    run_simulation,
    validate_time_steps,
)
from experiments.run_mpc_tracking_disturbance import (
    DISTURBANCE_STD,
    MEASUREMENT_STD,
    PROCESS_STD,
)


# 修改这里即可改变相对于基础 MEASUREMENT_STD 的测量噪声倍数。
DEFAULT_MEASUREMENT_NOISE_SCALE = 10.0


def main() -> None:
    """运行指定测量噪声倍数下的 EKF 与原始测量反馈对照实验。"""
    args = parse_args()
    validate_time_steps(OUTER_DT, args.inner_dt)

    inner_steps = int(round(args.duration / args.inner_dt))
    random = np.random.default_rng(args.seed)
    measurement_std = args.noise_scale * MEASUREMENT_STD
    measurement_noise = (
        random.normal(size=(inner_steps + 1, 12)) * measurement_std
    )
    disturbance_white = random.normal(size=(inner_steps, 5))

    results: list[SimulationResult] = []
    for use_ekf, name, description in (
        (True, "ekf_feedback", "使用 EKF 反馈的 MPC + PID"),
        (False, "raw_feedback", "使用原始带噪反馈的 MPC + PID"),
    ):
        print(f"正在运行：{description}……")
        result = run_simulation(
            architecture="cascade_pid",
            args=args,
            measurement_noise=measurement_noise,
            disturbance_white=disturbance_white,
            use_ekf=use_ekf,
            measurement_std=measurement_std,
        )
        results.append(replace(result, architecture=name))

    metrics = {
        result.architecture: comparison_metrics(result)
        for result in results
    }
    print_metrics(metrics, measurement_std, args)
    save_results(results, metrics, measurement_std, args)

    figure = plot_results(results, metrics)
    figure_path = OUTPUT_DIR / "mpc_pid_ekf_vs_raw_threefold_noise.png"
    figure.savefig(figure_path, dpi=170)
    print(f"结果图已保存至：{figure_path}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


def comparison_metrics(result: SimulationResult) -> dict[str, float]:
    """计算真实跟踪误差、反馈误差、控制抖动和求解成功率。"""
    metrics = calculate_metrics(result)
    position_error = np.linalg.norm(
        result.states[:, 0:3] - result.references[:, 0:3],
        axis=1,
    )
    measurement_error = np.linalg.norm(
        result.measured_states[:, 0:3] - result.states[:, 0:3],
        axis=1,
    )
    feedback_states = (
        result.estimated_states
        if result.architecture == "ekf_feedback"
        else result.measured_states
    )
    feedback_error = np.linalg.norm(
        feedback_states[:, 0:3] - result.states[:, 0:3],
        axis=1,
    )
    control_delta = np.linalg.norm(
        np.diff(result.desired_controls, axis=0),
        axis=1,
    )
    metrics.update(
        rms_position_error=float(np.sqrt(np.mean(position_error**2))),
        mean_measurement_position_error=float(measurement_error.mean()),
        mean_feedback_position_error=float(feedback_error.mean()),
        rms_control_delta=float(np.sqrt(np.mean(control_delta**2))),
    )
    return metrics


def print_metrics(
    metrics: dict[str, dict[str, float]],
    measurement_std: np.ndarray,
    args: Namespace,
) -> None:
    """打印两种反馈方式在完全相同随机工况下的性能指标。"""
    print("\n两组采用相同的模型失配、执行机构滞后、外扰和噪声样本")
    print(f"测量噪声倍数：{args.noise_scale:.1f} 倍")
    print(f"测量噪声标准差：{measurement_std}")
    print(f"有色外扰标准差：{DISTURBANCE_STD}")
    print(f"{'指标':<32}{'EKF 反馈':>18}{'原始测量反馈':>18}")
    labels = (
        ("mean_position_error", "平均真实位置误差 [m]"),
        ("rms_position_error", "真实位置误差 RMS [m]"),
        ("max_position_error", "最大真实位置误差 [m]"),
        ("final_position_error", "终点真实位置误差 [m]"),
        ("mean_yaw_error_deg", "平均真实偏航误差 [deg]"),
        ("mean_feedback_position_error", "平均反馈位置误差 [m]"),
        ("mean_control_delta", "平均控制增量"),
        ("rms_control_delta", "控制增量 RMS"),
        ("solver_success_rate", "位置 MPC 求解成功率"),
        ("allocation_success_rate", "控制分配成功率"),
    )
    for key, label in labels:
        ekf_value = metrics["ekf_feedback"][key]
        raw_value = metrics["raw_feedback"][key]
        if "rate" in key:
            print(
                f"{label:<38}{100.0 * ekf_value:>15.1f}%"
                f"{100.0 * raw_value:>15.1f}%"
            )
        else:
            print(f"{label:<38}{ekf_value:>16.5f}{raw_value:>16.5f}")

    ekf_mean = metrics["ekf_feedback"]["mean_position_error"]
    raw_mean = metrics["raw_feedback"]["mean_position_error"]
    improvement = 100.0 * (raw_mean - ekf_mean) / raw_mean
    print(f"EKF 对平均位置误差的改善比例：{improvement:.1f}%")


def save_results(
    results: list[SimulationResult],
    metrics: dict[str, dict[str, float]],
    measurement_std: np.ndarray,
    args: Namespace,
) -> None:
    """保存两组仿真历史、噪声配置和汇总指标。"""
    payload: dict[str, np.ndarray | float | int] = {
        "measurement_noise_scale": args.noise_scale,
        "measurement_std": measurement_std,
        "base_measurement_std": MEASUREMENT_STD,
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

    output_path = OUTPUT_DIR / "mpc_pid_ekf_vs_raw_threefold_noise_data.npz"
    np.savez_compressed(output_path, **payload)
    print(f"数据已保存至：{output_path}")


def plot_results(
    results: list[SimulationResult],
    metrics: dict[str, dict[str, float]],
) -> plt.Figure:
    """绘制轨迹、真实误差、反馈误差、姿态和控制变化对照图。"""
    ekf = next(result for result in results if result.architecture == "ekf_feedback")
    raw = next(result for result in results if result.architecture == "raw_feedback")
    figure = plt.figure(figsize=(15, 12), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)

    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    axis_3d.plot(
        ekf.references[:, 0],
        ekf.references[:, 1],
        -ekf.references[:, 2],
        "k--",
        linewidth=1.4,
        label="参考轨迹",
    )
    for result, label in ((ekf, "EKF 反馈"), (raw, "原始测量反馈")):
        axis_3d.plot(
            result.states[:, 0],
            result.states[:, 1],
            -result.states[:, 2],
            linewidth=1.2,
            label=label,
        )
    axis_3d.set_xlabel("x [m]")
    axis_3d.set_ylabel("y [m]")
    axis_3d.set_zlabel("高度 [m]")
    axis_3d.set_title("真实螺旋线轨迹")
    axis_3d.legend()

    axis_position = figure.add_subplot(grid[0, 1])
    for result, label in ((ekf, "EKF 反馈"), (raw, "原始测量反馈")):
        error = np.linalg.norm(
            result.states[:, 0:3] - result.references[:, 0:3],
            axis=1,
        )
        axis_position.plot(result.time, error, label=label)
    axis_position.set_title("真实位置跟踪误差")
    axis_position.set_xlabel("时间 [s]")
    axis_position.set_ylabel("误差 [m]")
    axis_position.grid(True)
    axis_position.legend()

    axis_feedback = figure.add_subplot(grid[1, 0])
    ekf_error = np.linalg.norm(ekf.estimated_states[:, 0:3] - ekf.states[:, 0:3], axis=1)
    raw_error = np.linalg.norm(raw.measured_states[:, 0:3] - raw.states[:, 0:3], axis=1)
    axis_feedback.plot(ekf.time, ekf_error, label="EKF 估计值")
    axis_feedback.plot(raw.time, raw_error, alpha=0.75, label="原始测量值")
    axis_feedback.set_title("位置反馈误差")
    axis_feedback.set_xlabel("时间 [s]")
    axis_feedback.set_ylabel("误差 [m]")
    axis_feedback.grid(True)
    axis_feedback.legend()

    axis_yaw = figure.add_subplot(grid[1, 1])
    for result, label in ((ekf, "EKF 反馈"), (raw, "原始测量反馈")):
        yaw_error = np.rad2deg(
            np.abs(wrap_to_pi(result.states[:, 8] - result.references[:, 5]))
        )
        axis_yaw.plot(result.time, yaw_error, label=label)
    axis_yaw.set_title("真实偏航角跟踪误差")
    axis_yaw.set_xlabel("时间 [s]")
    axis_yaw.set_ylabel("误差 [deg]")
    axis_yaw.grid(True)
    axis_yaw.legend()

    axis_torque = figure.add_subplot(grid[2, 0])
    for result, line_style, prefix in (
        (ekf, "-", "EKF"),
        (raw, "--", "原始测量"),
    ):
        torque_norm = np.linalg.norm(result.desired_controls[:, 2:5], axis=1)
        axis_torque.plot(
            result.time[:-1],
            torque_norm,
            line_style,
            linewidth=0.9,
            label=prefix,
        )
    axis_torque.set_title("期望力矩范数")
    axis_torque.set_xlabel("时间 [s]")
    axis_torque.set_ylabel("力矩 [N m]")
    axis_torque.grid(True)
    axis_torque.legend()

    axis_delta = figure.add_subplot(grid[2, 1])
    for result, label in ((ekf, "EKF 反馈"), (raw, "原始测量反馈")):
        control_delta = np.linalg.norm(np.diff(result.desired_controls, axis=0), axis=1)
        axis_delta.plot(result.time[1:-1], control_delta, linewidth=0.9, label=label)
    axis_delta.set_title("虚拟控制增量范数")
    axis_delta.set_xlabel("时间 [s]")
    axis_delta.set_ylabel("控制增量")
    axis_delta.grid(True)
    axis_delta.legend()

    figure.suptitle(
        "位置 MPC + 100 Hz 姿态/角速度 PID：EKF 与原始反馈对比\n"
        f"平均真实位置误差："
        f"{metrics['ekf_feedback']['mean_position_error']:.3f} m 对比 "
        f"{metrics['raw_feedback']['mean_position_error']:.3f} m",
        fontsize=14,
    )
    return figure


def parse_args() -> Namespace:
    """解析仿真时长、预测域、噪声倍数、随机种子和内环周期。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--inner-dt", type=float, default=INNER_DT)
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=DEFAULT_MEASUREMENT_NOISE_SCALE,
        help="相对于基础 MEASUREMENT_STD 的测量噪声倍数",
    )
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration 必须为正数")
    if args.horizon <= 0:
        parser.error("--horizon 必须为正整数")
    if args.noise_scale <= 0.0:
        parser.error("--noise-scale 必须为正数")
    if not np.isclose(args.duration / OUTER_DT, round(args.duration / OUTER_DT)):
        parser.error("--duration 必须是 0.1 s 的整数倍")
    return args


if __name__ == "__main__":
    main()

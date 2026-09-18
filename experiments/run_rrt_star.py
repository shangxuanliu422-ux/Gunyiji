"""在 2.5D 城市地图中运行 RRT 或 RRT* 路径规划。"""

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
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gunyiji.environment import Building, UrbanMap2_5D, make_default_city_map
from gunyiji.planning import PlannerConfig, PlannerResult, RRTPlanner
from gunyiji.planning.path_utils import height_change, path_cost, path_length

# 不使用命令行参数时，可修改该变量选择规划算法。
PLANNER_MODE = "rrt_star"  # "rrt" or "rrt_star"

START = np.array([10.0, 20.0, 15.0], dtype=float)  # [x, y, h]，h 向上为正
GOAL = np.array([190.0, 170.0, 15.0], dtype=float)


def main() -> None:
    """
    运行 RRT 或 RRT* 城市路径规划并绘制树生长过程。
    
    输入：
        无。
    
    输出：
        无；副作用为运行实验、保存数据并显示或导出图像。
    """
    args = parse_args()
    city_map = make_default_city_map()

    config = PlannerConfig(
        mode=args.planner,
        max_iterations=args.max_iterations,
        step_size=8.0,
        goal_sample_rate=0.14,
        goal_tolerance=8.5,
        neighbor_radius=22.0,
        collision_resolution=1.0,
        height_weight=args.height_weight,
        random_seed=args.seed,
    )

    planner = RRTPlanner(city_map, config)
    result = planner.plan(START, GOAL, snapshot_stride=args.snapshot_stride)

    figure_path = OUTPUT_DIR / f"{result.mode}_city_path.png"
    data_path = OUTPUT_DIR / f"{result.mode}_path_data.npz"
    planning_csv_path = OUTPUT_DIR / f"{result.mode}_path_planning.csv"
    dynamics_csv_path = OUTPUT_DIR / f"{result.mode}_path_dynamics.csv"
    save_path_data(
        result,
        config,
        START,
        GOAL,
        data_path,
        planning_csv_path,
        dynamics_csv_path,
    )
    summary_fig = plot_planning_result(city_map, result, START, GOAL, figure_path, show=not args.no_show)

    if args.no_show:
        print("已设置 --no-show，跳过规划树生长动画。")
        if summary_fig is not None:
            plt.close(summary_fig)
    else:
        animate_tree_growth(
            city_map,
            result,
            START,
            GOAL,
            step=args.animation_step,
            interval_ms=args.interval_ms,
            show=True,
        )

    print(f"规划器：{result.mode}")
    print(f"规划状态：{result.message}")
    print(f"是否成功：{result.success}")
    print(f"迭代次数：{result.iterations}")
    print(f"树节点数量：{len(result.nodes)}")
    print(f"路径点数量：{len(result.path)}")
    print(f"路径长度：{path_length(result.path):.2f} m")
    print(f"累计高度变化：{height_change(result.path):.2f} m")
    print(f"含高度惩罚的路径代价：{path_cost(result.path, config.height_weight):.2f}")
    print(f"结果图已保存至：{figure_path}")
    print(f"路径数据已保存至：{data_path}")
    print(f"规划坐标 CSV 已保存至：{planning_csv_path}")
    print(f"动力学坐标 CSV 已保存至：{dynamics_csv_path}")
    print("动力学坐标首点 [x, y, z=-h]：", result.dynamics_path[0])
    print("动力学坐标末点 [x, y, z=-h]：", result.dynamics_path[-1])


def parse_args() -> Namespace:
    """
    解析实验脚本的命令行参数并返回参数对象。
    
    输入：
        无。
    
    输出：
        解析后的 argparse.Namespace 参数对象。
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner",
        choices=("rrt", "rrt_star"),
        default=PLANNER_MODE,
        help="选择规划算法，默认值由本文件中的 PLANNER_MODE 决定",
    )
    parser.add_argument("--max-iterations", type=int, default=3200)
    parser.add_argument("--height-weight", type=float, default=2.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--snapshot-stride",
        type=int,
        default=100,
        help="每经过指定次数的规划迭代保存一帧动画",
    )
    parser.add_argument("--animation-step", type=int, default=1)
    parser.add_argument("--interval-ms", type=int, default=35)
    parser.add_argument("--no-show", action="store_true", help="不打开 Matplotlib 动画窗口")
    return parser.parse_args()


def save_path_data(
    result: PlannerResult,
    config: PlannerConfig,
    start: np.ndarray,
    goal: np.ndarray,
    data_path: Path,
    planning_csv_path: Path,
    dynamics_csv_path: Path,
) -> None:
    """
    将规划路径同时保存为压缩 NPZ 和规划/动力学坐标 CSV。
    
    输入：
        result: 规划器或控制器返回的结果对象。
        config: 当前算法或控制器配置对象。
        start: 路径或线段起点 [x, y, h]。
        goal: 路径规划终点 [x, y, h]。
        data_path: 压缩数据文件保存路径。
        planning_csv_path: 规划坐标路径 CSV 保存路径。
        dynamics_csv_path: 动力学坐标路径 CSV 保存路径。
    
    输出：
        无；路径数据写入指定文件。
    """

    np.savez_compressed(
        data_path,
        planning_path=result.path,
        dynamics_path=result.dynamics_path,
        start=np.asarray(start, dtype=float),
        goal=np.asarray(goal, dtype=float),
        success=np.asarray(result.success),
        planner=np.asarray(result.mode),
        cost=np.asarray(result.cost),
        height_weight=np.asarray(config.height_weight),
    )
    np.savetxt(
        planning_csv_path,
        result.path,
        delimiter=",",
        header="x_m,y_m,height_h_m",
        comments="",
    )
    np.savetxt(
        dynamics_csv_path,
        result.dynamics_path,
        delimiter=",",
        header="x_m,y_m,z_down_m",
        comments="",
    )


def plot_planning_result(
    city_map: UrbanMap2_5D,
    result: PlannerResult,
    start: np.ndarray,
    goal: np.ndarray,
    figure_path: Path,
    show: bool = True,
) -> plt.Figure:
    """
    绘制规划树、最终路径、俯视图和高度曲线的静态结果图。
    
    输入：
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        result: 规划器或控制器返回的结果对象。
        start: 路径或线段起点 [x, y, h]。
        goal: 路径规划终点 [x, y, h]。
        figure_path: 静态结果图保存路径。
        show: 是否在函数内部调用 Matplotlib 显示窗口。
    
    输出：
        生成并保存的 Matplotlib Figure 对象。
    """
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    draw_city_3d(ax_3d, city_map)
    draw_tree_3d(ax_3d, result.nodes, color="0.68", linewidth=0.35, alpha=0.20)
    draw_path_3d(ax_3d, result.path, color="tab:red", linewidth=3.0, label="最终路径")
    ax_3d.scatter(start[0], start[1], start[2], color="tab:green", s=65, label="起点")
    ax_3d.scatter(goal[0], goal[1], goal[2], color="tab:red", s=65, label="终点")
    configure_3d_axes(ax_3d, city_map)
    ax_3d.set_title(f"{result.mode.upper()} 2.5D 城市路径规划")
    ax_3d.legend(loc="upper left")

    ax_top = fig.add_subplot(grid[0, 1])
    draw_city_top(ax_top, city_map)
    draw_tree_top(ax_top, result.nodes, color="0.75", linewidth=0.35, alpha=0.20)
    ax_top.plot(result.path[:, 0], result.path[:, 1], color="tab:red", linewidth=2.5, label="路径")
    ax_top.scatter(start[0], start[1], color="tab:green", s=45, label="起点")
    ax_top.scatter(goal[0], goal[1], color="tab:red", s=45, label="终点")
    ax_top.set_title("俯视图")
    ax_top.set_xlabel("x [m]")
    ax_top.set_ylabel("y [m]")
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.grid(True)
    ax_top.legend(loc="upper left")

    ax_height = fig.add_subplot(grid[1, 1])
    cumulative_distance = path_distance_axis(result.path)
    ax_height.plot(cumulative_distance, result.path[:, 2], "o-", color="tab:blue", label="路径高度")
    ax_height.axhline(city_map.bounds.h_min, color="0.5", linestyle="--", linewidth=1.0, label="高度边界")
    ax_height.axhline(city_map.bounds.h_max, color="0.5", linestyle="--", linewidth=1.0)
    ax_height.set_title("路径高度剖面")
    ax_height.set_xlabel("沿路径距离 [m]")
    ax_height.set_ylabel("高度 h [m]")
    ax_height.grid(True)
    ax_height.legend()

    fig.savefig(figure_path, dpi=160)
    if not show:
        plt.close(fig)
    return fig


def animate_tree_growth(
    city_map: UrbanMap2_5D,
    result: PlannerResult,
    start: np.ndarray,
    goal: np.ndarray,
    step: int = 1,
    interval_ms: int = 35,
    show: bool = True,
) -> tuple[plt.Figure | None, FuncAnimation | None]:
    """
    使用 Matplotlib 动画展示 RRT 树随迭代增长的过程。
    
    输入：
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        result: 规划器或控制器返回的结果对象。
        start: 路径或线段起点 [x, y, h]。
        goal: 路径规划终点 [x, y, h]。
        step: 动画帧下采样步长。
        interval_ms: 相邻动画帧之间的时间间隔，单位为毫秒。
        show: 是否在函数内部调用 Matplotlib 显示窗口。
    
    输出：
        Matplotlib 图对象与 FuncAnimation 动画对象；无快照时均为 None。
    """
    if not result.snapshots:
        return None, None

    snapshot_indices = np.arange(0, len(result.snapshots), max(1, step), dtype=int)
    if snapshot_indices[-1] != len(result.snapshots) - 1:
        snapshot_indices = np.append(snapshot_indices, len(result.snapshots) - 1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_city_3d(ax, city_map)
    configure_3d_axes(ax, city_map)
    ax.scatter(start[0], start[1], start[2], color="tab:green", s=65, label="起点")
    ax.scatter(goal[0], goal[1], goal[2], color="tab:red", s=65, label="终点")
    ax.set_title(f"{result.mode.upper()} 规划树生长过程")

    empty_segment = np.full((2, 3), np.nan, dtype=float)
    tree_collection = Line3DCollection(
        [empty_segment],
        colors="0.45",
        linewidths=0.55,
        alpha=0.30,
    )
    ax.add_collection3d(tree_collection)
    best_path_line, = ax.plot([], [], [], color="tab:red", linewidth=3.0, label="当前最优路径")
    time_text = ax.text2D(0.03, 0.94, "", transform=ax.transAxes)
    ax.legend(loc="upper left")

    def update(frame_idx: int) -> tuple[object, ...]:
        """
        根据当前动画帧更新轨迹、飞行器位置、机体坐标轴和状态文字。
        
        输入：
            frame_idx: 当前动画帧索引。
        
        输出：
            本帧更新过的 Matplotlib 图形对象元组。
        """
        snapshot = result.snapshots[int(snapshot_indices[frame_idx])]
        points = snapshot.points
        parents = snapshot.parents
        segments = [
            np.vstack((points[parent_idx], points[idx]))
            for idx, parent_idx in enumerate(parents)
            if parent_idx >= 0
        ]
        tree_collection.set_segments(segments if segments else [empty_segment])

        if snapshot.best_path is not None:
            best_path_line.set_data(snapshot.best_path[:, 0], snapshot.best_path[:, 1])
            best_path_line.set_3d_properties(snapshot.best_path[:, 2])
        else:
            best_path_line.set_data([], [])
            best_path_line.set_3d_properties([])

        time_text.set_text(
            f"iteration = {snapshot.iteration} / {result.iterations}\n"
            f"nodes = {len(points)} / {len(result.nodes)}"
        )
        return tree_collection, best_path_line, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(snapshot_indices),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    fig._ani_ref = ani
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ani


def draw_city_3d(ax: plt.Axes, city_map: UrbanMap2_5D) -> None:
    """
    在 Matplotlib 三维坐标轴中绘制城市建筑和安全边界。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
    
    输出：
        无。
    """
    for building in city_map.buildings:
        draw_building_3d(ax, building, color="lightgray", alpha=0.5, edgecolor="0.35")
    for inflated in city_map.inflated_buildings:
        draw_building_wireframe(ax, inflated, color="tab:red", alpha=0.32)


def draw_building_3d(
    ax: plt.Axes,
    building: Building,
    color: str,
    alpha: float,
    edgecolor: str,
) -> None:
    """
    在 Matplotlib 三维坐标轴中绘制单个实体长方体建筑。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        building: 需要绘制或处理的建筑物对象。
        color: 绘图颜色。
        alpha: 图形透明度。
        edgecolor: 建筑物边缘颜色。
    
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
        color=color,
        edgecolor=edgecolor,
        linewidth=0.4,
        alpha=alpha,
        shade=True,
    )


def draw_building_wireframe(
    ax: plt.Axes,
    building: Building,
    color: str,
    alpha: float,
) -> None:
    """
    在 Matplotlib 三维坐标轴中绘制建筑安全膨胀线框。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        building: 需要绘制或处理的建筑物对象。
        color: 绘图颜色。
        alpha: 图形透明度。
    
    输出：
        无。
    """
    x0, x1 = building.xmin, building.xmax
    y0, y1 = building.ymin, building.ymax
    z0, z1 = 0.0, building.height
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    edge_indices = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    segments = [corners[[start, end]] for start, end in edge_indices]
    collection = Line3DCollection(
        segments,
        colors=color,
        linewidths=0.65,
        alpha=alpha,
    )
    ax.add_collection3d(collection)


def draw_city_top(ax: plt.Axes, city_map: UrbanMap2_5D) -> None:
    """
    在 Matplotlib 二维坐标轴中绘制建筑物俯视轮廓。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
    
    输出：
        无。
    """
    for building in city_map.inflated_buildings:
        rect = plt.Rectangle(
            (building.xmin, building.ymin),
            building.xmax - building.xmin,
            building.ymax - building.ymin,
            facecolor="tab:red",
            edgecolor="none",
            alpha=0.10,
        )
        ax.add_patch(rect)
    for building in city_map.buildings:
        rect = plt.Rectangle(
            (building.xmin, building.ymin),
            building.xmax - building.xmin,
            building.ymax - building.ymin,
            facecolor="0.62",
            edgecolor="0.25",
            alpha=0.80,
        )
        ax.add_patch(rect)
        ax.text(
            0.5 * (building.xmin + building.xmax),
            0.5 * (building.ymin + building.ymax),
            f"{building.height:.0f}m",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
        )
    ax.set_xlim(city_map.bounds.x_min, city_map.bounds.x_max)
    ax.set_ylim(city_map.bounds.y_min, city_map.bounds.y_max)


def draw_tree_3d(
    ax: plt.Axes,
    nodes: tuple,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    """
    使用三维线段集合绘制规划树的全部边。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        nodes: 当前规划树节点序列。
        color: 绘图颜色。
        linewidth: 绘图线宽。
        alpha: 图形透明度。
    
    输出：
        无。
    """
    points = np.asarray([node.point for node in nodes], dtype=float)
    segments = [
        np.vstack((points[node.parent], points[idx]))
        for idx, node in enumerate(nodes)
        if node.parent >= 0
    ]
    collection = Line3DCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(collection)


def draw_tree_top(
    ax: plt.Axes,
    nodes: tuple,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    """
    在俯视坐标轴中绘制规划树的全部边。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        nodes: 当前规划树节点序列。
        color: 绘图颜色。
        linewidth: 绘图线宽。
        alpha: 图形透明度。
    
    输出：
        无。
    """
    points = np.asarray([node.point for node in nodes], dtype=float)
    for idx, node in enumerate(nodes):
        if node.parent < 0:
            continue
        segment = np.vstack((points[node.parent], points[idx]))
        ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=linewidth, alpha=alpha)


def draw_path_3d(
    ax: plt.Axes,
    path: np.ndarray,
    color: str,
    linewidth: float,
    label: str,
) -> None:
    """
    在三维坐标轴中绘制带白色轮廓的最终路径。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        path: 由 [x, y, h] 路径点组成的二维数组。
        color: 绘图颜色。
        linewidth: 绘图线宽。
        label: 图例或悬停信息标签。
    
    输出：
        无。
    """
    ax.plot(
        path[:, 0],
        path[:, 1],
        path[:, 2],
        color="white",
        linewidth=linewidth + 3.0,
        alpha=0.90,
    )
    ax.plot(
        path[:, 0],
        path[:, 1],
        path[:, 2],
        "o-",
        color=color,
        linewidth=linewidth,
        markersize=3.5,
        label=label,
    )


def configure_3d_axes(ax: plt.Axes, city_map: UrbanMap2_5D) -> None:
    """
    根据城市地图范围统一设置三维坐标轴和观察视角。
    
    输入：
        ax: 目标 Matplotlib 坐标轴对象。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
    
    输出：
        无。
    """
    bounds = city_map.bounds
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_zlim(0.0, bounds.h_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("高度 h [m]")
    ax.set_proj_type("ortho")
    ax.view_init(elev=38.0, azim=-62.0)
    ax.set_box_aspect(
        [
            bounds.x_max - bounds.x_min,
            bounds.y_max - bounds.y_min,
            1.35 * (bounds.h_max - 0.0),
        ]
    )


def path_distance_axis(path: np.ndarray) -> np.ndarray:
    """
    计算路径点对应的累计路程横坐标。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
    
    输出：
        与路径点一一对应的累计路程数组。
    """
    if len(path) == 0:
        return np.array([], dtype=float)
    if len(path) == 1:
        return np.array([0.0], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


if __name__ == "__main__":
    main()

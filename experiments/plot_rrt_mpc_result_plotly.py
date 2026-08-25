"""Render saved RRT*, smoothed-reference and NMPC data with Plotly."""

from __future__ import annotations

import sys
import webbrowser
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gunyiji.environment import Building, UrbanMap2_5D, make_default_city_map

REFERENCE_DATA = OUTPUT_DIR / "rrt_star_reference_trajectory.npz"
TRACKING_DATA = OUTPUT_DIR / "rrt_star_nmpc_tracking_data.npz"
HTML_OUTPUT = OUTPUT_DIR / "rrt_star_nmpc_plotly.html"

COLORS = {
    "raw": "#7A828E",
    "shortcut": "#F59E0B",
    "spline": "#8B5CF6",
    "reference": "#DC2626",
    "actual": "#087EA4",
    "building": "#A8ADB5",
    "building_edge": "#4B5563",
    "clearance": "#EF4444",
    "start": "#16A34A",
    "goal": "#DC2626",
}


def main() -> None:
    """
    读取已保存的规划与跟踪数据，生成响应式 Plotly 综合结果页。
    
    输入：
        无。
    
    输出：
        无；副作用为运行实验、保存数据并显示或导出图像。
    """
    args = parse_args()
    reference, tracking = load_saved_results()
    city_map = make_default_city_map()
    figure = build_result_figure(city_map, reference, tracking)
    html = figure.to_html(
        include_plotlyjs=True,
        full_html=True,
        default_width="100vw",
        default_height="100vh",
        div_id="rrt-nmpc-result",
        post_script=(
            "window.addEventListener('resize', function() {"
            "Plotly.Plots.resize(document.getElementById('rrt-nmpc-result'));"
            "});"
        ),
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "rrt_star_nmpc_result",
                "height": 1000,
                "width": 1700,
                "scale": 2,
            },
        },
    )
    html = html.replace(
        "<head>",
        (
            "<head>"
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            "<style>"
            "html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; "
            "background: #f8fafc; }"
            "#rrt-nmpc-result { width: 100vw !important; height: 100vh !important; }"
            "</style>"
        ),
        1,
    )
    HTML_OUTPUT.write_text(html, encoding="utf-8")
    if not args.no_show:
        webbrowser.open(HTML_OUTPUT.resolve().as_uri())
    print(f"saved Plotly result: {HTML_OUTPUT}")


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
        "--no-show",
        action="store_true",
        help="write the HTML without opening it in the browser",
    )
    return parser.parse_args()


def load_saved_results() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    从输出目录读取已保存的参考轨迹和 NMPC 跟踪数据。
    
    输入：
        无。
    
    输出：
        参考轨迹数据字典与实际跟踪数据字典组成的元组。
    """
    if not REFERENCE_DATA.exists():
        raise FileNotFoundError(
            f"missing {REFERENCE_DATA}; run experiments/run_rrt_mpc_tracking.py first"
        )
    if not TRACKING_DATA.exists():
        raise FileNotFoundError(
            f"missing {TRACKING_DATA}; run experiments/run_rrt_mpc_tracking.py first"
        )

    with np.load(REFERENCE_DATA) as data:
        reference = {name: data[name].copy() for name in data.files}
    with np.load(TRACKING_DATA) as data:
        tracking = {name: data[name].copy() for name in data.files}
    return reference, tracking


def build_result_figure(
    city_map: UrbanMap2_5D,
    reference: dict[str, np.ndarray],
    tracking: dict[str, np.ndarray],
) -> go.Figure:
    """
    创建包含三维城市、俯视图和时序曲线的 Plotly 综合图。
    
    输入：
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        reference: 已加载的参考轨迹数据字典。
        tracking: 已加载的 NMPC 实际跟踪数据字典。
    
    输出：
        包含三维、俯视和时序子图的 Plotly Figure。
    """
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy", "secondary_y": True}],
        ],
        column_widths=[0.60, 0.40],
        row_heights=[0.56, 0.44],
        horizontal_spacing=0.045,
        vertical_spacing=0.12,
        subplot_titles=(
            "Urban 3D Path Planning And Tracking",
            "Top View",
            "Height And Speed",
        ),
    )

    add_city_3d(figure, city_map)
    add_paths_3d(figure, reference, tracking)
    add_top_view(figure, city_map, reference, tracking)
    add_profiles(figure, reference, tracking)
    apply_layout(figure, city_map, reference, tracking)
    return figure


def add_city_3d(figure: go.Figure, city_map: UrbanMap2_5D) -> None:
    """
    向 Plotly 三维子图添加实体建筑和膨胀安全线框。
    
    输入：
        figure: 目标 Plotly 图对象。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
    
    输出：
        无。
    """
    for index, building in enumerate(city_map.buildings):
        figure.add_trace(
            building_mesh(building, show_legend=index == 0),
            row=1,
            col=1,
        )
    for index, building in enumerate(city_map.inflated_buildings):
        figure.add_trace(
            building_wireframe(building, show_legend=index == 0),
            row=1,
            col=1,
        )


def add_paths_3d(
    figure: go.Figure,
    reference: dict[str, np.ndarray],
    tracking: dict[str, np.ndarray],
) -> None:
    """
    向 Plotly 三维子图添加原始、简化、平滑、参考和实际轨迹。
    
    输入：
        figure: 目标 Plotly 图对象。
        reference: 已加载的参考轨迹数据字典。
        tracking: 已加载的 NMPC 实际跟踪数据字典。
    
    输出：
        无。
    """
    raw = reference["raw_path"]
    shortcut = reference["shortcut_path"]
    spline = reference["smooth_path"]
    timed = reference["position_planning"]
    time = reference["time"]
    speed = reference["speed"]
    states = tracking["states"]
    actual = np.column_stack((states[:, 0], states[:, 1], -states[:, 2]))

    figure.add_trace(
        path_3d_trace(
            raw,
            name="RRT* waypoints",
            color=COLORS["raw"],
            width=3,
            dash="dash",
            mode="lines+markers",
            marker_size=4,
            hover_text=_point_hover(raw, "RRT*"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        path_3d_trace(
            shortcut,
            name="Shortcut path",
            color=COLORS["shortcut"],
            width=5,
            mode="lines+markers",
            marker_size=5,
            hover_text=_point_hover(shortcut, "Shortcut"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        path_3d_trace(
            spline,
            name="Cubic B-spline",
            color=COLORS["spline"],
            width=5,
            hover_text=_point_hover(spline, "B-spline"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        path_3d_trace(
            timed,
            name="Timed reference",
            color=COLORS["reference"],
            width=7,
            dash="dash",
            hover_text=[
                (
                    f"Reference<br>t={time[index]:.1f} s"
                    f"<br>x={point[0]:.2f} m"
                    f"<br>y={point[1]:.2f} m"
                    f"<br>h={point[2]:.2f} m"
                    f"<br>speed={speed[index]:.2f} m/s"
                )
                for index, point in enumerate(timed)
            ],
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        path_3d_trace(
            actual,
            name="NMPC actual",
            color=COLORS["actual"],
            width=6,
            hover_text=[
                (
                    f"NMPC<br>t={time[index]:.1f} s"
                    f"<br>x={point[0]:.2f} m"
                    f"<br>y={point[1]:.2f} m"
                    f"<br>h={point[2]:.2f} m"
                )
                for index, point in enumerate(actual)
            ],
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter3d(
            x=[timed[0, 0]],
            y=[timed[0, 1]],
            z=[timed[0, 2]],
            mode="markers",
            marker={"size": 7, "color": COLORS["start"], "symbol": "circle"},
            name="Start",
            legendgroup="endpoints",
            hovertemplate="Start<br>x=%{x:.2f}<br>y=%{y:.2f}<br>h=%{z:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter3d(
            x=[timed[-1, 0]],
            y=[timed[-1, 1]],
            z=[timed[-1, 2]],
            mode="markers",
            marker={"size": 7, "color": COLORS["goal"], "symbol": "diamond"},
            name="Goal",
            legendgroup="endpoints",
            hovertemplate="Goal<br>x=%{x:.2f}<br>y=%{y:.2f}<br>h=%{z:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def add_top_view(
    figure: go.Figure,
    city_map: UrbanMap2_5D,
    reference: dict[str, np.ndarray],
    tracking: dict[str, np.ndarray],
) -> None:
    """
    向 Plotly 俯视子图添加建筑轮廓和各阶段路径。
    
    输入：
        figure: 目标 Plotly 图对象。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        reference: 已加载的参考轨迹数据字典。
        tracking: 已加载的 NMPC 实际跟踪数据字典。
    
    输出：
        无。
    """
    for building in city_map.inflated_buildings:
        add_footprint(
            figure,
            building,
            fill_color="rgba(239,68,68,0.08)",
            line_color="rgba(239,68,68,0.45)",
            dash="dot",
        )
    for building in city_map.buildings:
        add_footprint(
            figure,
            building,
            fill_color="rgba(107,114,128,0.65)",
            line_color="#4B5563",
            label=f"{building.height:.0f} m",
        )

    states = tracking["states"]
    actual = np.column_stack((states[:, 0], states[:, 1], -states[:, 2]))
    top_paths = (
        ("RRT*", reference["raw_path"], COLORS["raw"], "dash", 2),
        ("Shortcut", reference["shortcut_path"], COLORS["shortcut"], "solid", 3),
        ("B-spline", reference["smooth_path"], COLORS["spline"], "dot", 3),
        ("Reference", reference["position_planning"], COLORS["reference"], "dash", 4),
        ("NMPC", actual, COLORS["actual"], "solid", 3),
    )
    for name, path, color, dash, width in top_paths:
        figure.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines",
                line={"color": color, "width": width, "dash": dash},
                name=name,
                legendgroup=name,
                showlegend=False,
                hovertemplate=(
                    f"{name}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )


def add_profiles(
    figure: go.Figure,
    reference: dict[str, np.ndarray],
    tracking: dict[str, np.ndarray],
) -> None:
    """
    向 Plotly 时序子图添加参考与实际高度和速度曲线。
    
    输入：
        figure: 目标 Plotly 图对象。
        reference: 已加载的参考轨迹数据字典。
        tracking: 已加载的 NMPC 实际跟踪数据字典。
    
    输出：
        无。
    """
    time = reference["time"]
    states = tracking["states"]
    reference_height = reference["position_planning"][:, 2]
    actual_height = -states[:, 2]
    reference_speed = reference["speed"]
    actual_speed = np.linalg.norm(states[:, 3:6], axis=1)

    figure.add_trace(
        go.Scatter(
            x=time,
            y=reference_height,
            mode="lines",
            line={"color": COLORS["reference"], "width": 3, "dash": "dash"},
            name="Height reference",
            legendgroup="profiles",
            showlegend=False,
            hovertemplate="t=%{x:.1f} s<br>h ref=%{y:.2f} m<extra></extra>",
        ),
        row=2,
        col=2,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=time,
            y=actual_height,
            mode="lines",
            line={"color": COLORS["actual"], "width": 2.5},
            name="Height actual",
            legendgroup="profiles",
            showlegend=False,
            hovertemplate="t=%{x:.1f} s<br>h actual=%{y:.2f} m<extra></extra>",
        ),
        row=2,
        col=2,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=time,
            y=reference_speed,
            mode="lines",
            line={"color": COLORS["shortcut"], "width": 2.5, "dash": "dash"},
            name="Speed reference",
            legendgroup="profiles",
            showlegend=False,
            hovertemplate="t=%{x:.1f} s<br>speed ref=%{y:.2f} m/s<extra></extra>",
        ),
        row=2,
        col=2,
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=time,
            y=actual_speed,
            mode="lines",
            line={"color": "#0F766E", "width": 2},
            name="Speed actual",
            legendgroup="profiles",
            showlegend=False,
            hovertemplate="t=%{x:.1f} s<br>speed actual=%{y:.2f} m/s<extra></extra>",
        ),
        row=2,
        col=2,
        secondary_y=True,
    )


def apply_layout(
    figure: go.Figure,
    city_map: UrbanMap2_5D,
    reference: dict[str, np.ndarray],
    tracking: dict[str, np.ndarray],
) -> None:
    """
    设置 Plotly 图的响应式布局、坐标轴、相机、标题和统计信息。
    
    输入：
        figure: 目标 Plotly 图对象。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        reference: 已加载的参考轨迹数据字典。
        tracking: 已加载的 NMPC 实际跟踪数据字典。
    
    输出：
        无。
    """
    states = tracking["states"]
    position_error = np.linalg.norm(states[:, 0:3] - reference["position"], axis=1)
    speed_error = np.abs(
        np.linalg.norm(states[:, 3:6], axis=1) - reference["speed"]
    )
    success_rate = 100.0 * np.mean(tracking["solver_success"])

    figure.update_layout(
        template="plotly_white",
        autosize=True,
        margin={"l": 42, "r": 46, "t": 155, "b": 82},
        title={
            "text": "<b>RRT* Path Processing And NMPC Tracking</b>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.985,
            "yanchor": "top",
            "font": {"size": 26, "color": "#111827"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.055,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.86)",
            "bordercolor": "#D1D5DB",
            "borderwidth": 1,
            "font": {"size": 11},
            "entrywidth": 120,
            "entrywidthmode": "pixels",
        },
        hoverlabel={"bgcolor": "white", "font_size": 12, "font_family": "Arial"},
    )
    figure.update_scenes(
        xaxis={
            "title": "x [m]",
            "range": [city_map.bounds.x_min, city_map.bounds.x_max],
            "backgroundcolor": "#F8FAFC",
            "gridcolor": "#D1D5DB",
        },
        yaxis={
            "title": "y [m]",
            "range": [city_map.bounds.y_min, city_map.bounds.y_max],
            "backgroundcolor": "#F8FAFC",
            "gridcolor": "#D1D5DB",
        },
        zaxis={
            "title": "height h=-z [m]",
            "range": [0.0, city_map.bounds.h_max],
            "backgroundcolor": "#F8FAFC",
            "gridcolor": "#D1D5DB",
        },
        aspectmode="manual",
        aspectratio={"x": 1.0, "y": 1.0, "z": 0.48},
        camera={
            "eye": {"x": 1.55, "y": -1.7, "z": 1.15},
            "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        },
        dragmode="orbit",
    )
    figure.update_xaxes(
        title_text="x [m]",
        range=[city_map.bounds.x_min, city_map.bounds.x_max],
        showgrid=True,
        gridcolor="#D1D5DB",
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title_text="y [m]",
        range=[city_map.bounds.y_min, city_map.bounds.y_max],
        scaleanchor="x",
        scaleratio=1,
        showgrid=True,
        gridcolor="#D1D5DB",
        row=1,
        col=2,
    )
    figure.update_xaxes(
        title_text="time [s]",
        showgrid=True,
        gridcolor="#D1D5DB",
        row=2,
        col=2,
    )
    figure.update_yaxes(
        title_text="height [m]",
        showgrid=True,
        gridcolor="#D1D5DB",
        row=2,
        col=2,
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="speed [m/s]",
        showgrid=False,
        row=2,
        col=2,
        secondary_y=True,
    )
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.050,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        align="center",
        text=(
            f"RRT*: {len(reference['raw_path'])} points"
            f" &nbsp;&nbsp; | &nbsp;&nbsp; Shortcut: {len(reference['shortcut_path'])} points"
            f" &nbsp;&nbsp; | &nbsp;&nbsp; Duration: {reference['time'][-1]:.1f} s"
            f" &nbsp;&nbsp; | &nbsp;&nbsp; NMPC success: {success_rate:.1f}%"
            "<br>"
            f"Mean position error: {position_error.mean():.4f} m"
            f" &nbsp;&nbsp; | &nbsp;&nbsp; Max position error: {position_error.max():.3f} m"
            f" &nbsp;&nbsp; | &nbsp;&nbsp; Mean speed error: {speed_error.mean():.4f} m/s"
        ),
        font={"size": 12, "color": "#374151"},
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#D1D5DB",
        borderwidth=1,
        borderpad=6,
    )

    for annotation in figure.layout.annotations[:3]:
        annotation.font = {"size": 16, "color": "#334155"}
        annotation.yshift = -4


def building_mesh(building: Building, show_legend: bool) -> go.Mesh3d:
    """
    将建筑物转换为不透明 Plotly 三角网格对象。
    
    输入：
        building: 需要绘制或处理的建筑物对象。
        show_legend: 是否为当前图形对象显示图例项。
    
    输出：
        表示实体建筑物的 Plotly Mesh3d 对象。
    """
    vertices = building_vertices(building)
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i,
        j=j,
        k=k,
        color=COLORS["building"],
        opacity=1.0,
        flatshading=True,
        lighting={
            "ambient": 0.68,
            "diffuse": 0.72,
            "specular": 0.12,
            "roughness": 0.82,
        },
        lightposition={"x": 100, "y": -80, "z": 180},
        name="Buildings",
        legendgroup="buildings",
        showlegend=show_legend,
        hovertext=(
            f"{building.name}<br>height={building.height:.1f} m"
            f"<br>size={building.size[0]:.0f} x {building.size[1]:.0f} m"
        ),
        hovertemplate="%{hovertext}<extra></extra>",
    )


def building_wireframe(building: Building, show_legend: bool) -> go.Scatter3d:
    """
    将膨胀建筑物转换为 Plotly 三维线框对象。
    
    输入：
        building: 需要绘制或处理的建筑物对象。
        show_legend: 是否为当前图形对象显示图例项。
    
    输出：
        表示安全边界的 Plotly Scatter3d 对象。
    """
    vertices = building_vertices(building)
    edge_indices = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for start, end in edge_indices:
        x.extend((vertices[start, 0], vertices[end, 0], None))
        y.extend((vertices[start, 1], vertices[end, 1], None))
        z.extend((vertices[start, 2], vertices[end, 2], None))
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": COLORS["clearance"], "width": 2},
        opacity=0.42,
        name="Inflated safety boundary",
        legendgroup="clearance",
        showlegend=show_legend,
        hoverinfo="skip",
    )


def building_vertices(building: Building) -> np.ndarray:
    """
    计算长方体建筑物的八个顶点坐标。
    
    输入：
        building: 需要绘制或处理的建筑物对象。
    
    输出：
        形状为 (8, 3) 的长方体顶点数组。
    """
    return np.array(
        [
            [building.xmin, building.ymin, 0.0],
            [building.xmax, building.ymin, 0.0],
            [building.xmax, building.ymax, 0.0],
            [building.xmin, building.ymax, 0.0],
            [building.xmin, building.ymin, building.height],
            [building.xmax, building.ymin, building.height],
            [building.xmax, building.ymax, building.height],
            [building.xmin, building.ymax, building.height],
        ],
        dtype=float,
    )


def path_3d_trace(
    path: np.ndarray,
    name: str,
    color: str,
    width: int,
    hover_text: list[str],
    dash: str = "solid",
    mode: str = "lines",
    marker_size: int = 3,
) -> go.Scatter3d:
    """
    根据路径点和样式参数创建 Plotly 三维轨迹对象。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
        name: 对象名称或校验字段名称。
        color: 绘图颜色。
        width: 三维轨迹线宽。
        hover_text: 每个路径点对应的悬停文本。
        dash: 轨迹线的虚线样式。
        mode: 轨迹线的绘制模式。
        marker_size: 路径点标记大小。
    
    输出：
        配置完成的 Plotly Scatter3d 轨迹对象。
    """
    return go.Scatter3d(
        x=path[:, 0],
        y=path[:, 1],
        z=path[:, 2],
        mode=mode,
        line={"color": color, "width": width, "dash": dash},
        marker={"size": marker_size, "color": color},
        name=name,
        legendgroup=name,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
    )


def add_footprint(
    figure: go.Figure,
    building: Building,
    fill_color: str,
    line_color: str,
    dash: str = "solid",
    label: str | None = None,
) -> None:
    """
    在 Plotly 俯视图中添加建筑物矩形轮廓和可选高度文字。
    
    输入：
        figure: 目标 Plotly 图对象。
        building: 需要绘制或处理的建筑物对象。
        fill_color: 矩形区域填充颜色。
        line_color: 矩形边界颜色。
        dash: 轨迹线的虚线样式。
        label: 图例或悬停信息标签。
    
    输出：
        无。
    """
    x = [building.xmin, building.xmax, building.xmax, building.xmin, building.xmin]
    y = [building.ymin, building.ymin, building.ymax, building.ymax, building.ymin]
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="toself",
            fillcolor=fill_color,
            line={"color": line_color, "width": 1, "dash": dash},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    if label is not None:
        figure.add_annotation(
            x=0.5 * (building.xmin + building.xmax),
            y=0.5 * (building.ymin + building.ymax),
            text=label,
            xref="x",
            yref="y",
            showarrow=False,
            font={"size": 10, "color": "white"},
        )


def _point_hover(path: np.ndarray, label: str) -> list[str]:
    """
    为路径点生成包含序号和三维坐标的悬停文本。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
        label: 图例或悬停信息标签。
    
    输出：
        与路径点数量相同的悬停文本列表。
    """
    return [
        (
            f"{label}<br>point={index}"
            f"<br>x={point[0]:.2f} m"
            f"<br>y={point[1]:.2f} m"
            f"<br>h={point[2]:.2f} m"
        )
        for index, point in enumerate(path)
    ]


if __name__ == "__main__":
    main()

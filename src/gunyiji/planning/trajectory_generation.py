"""将无碰撞离散路径转换为 NMPC 可跟踪的连续参考轨迹。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import splprep, splev

from gunyiji.environment.urban_map import UrbanMap2_5D
from gunyiji.planning.path_utils import remove_redundant_points


@dataclass(frozen=True)
class TrajectoryGenerationConfig:
    """
    保存路径简化、B 样条平滑和时间参数化约束。
    
    输入：
        dt: 离散控制或积分时间步长，单位为秒。
        cruise_speed: 期望水平巡航速度上限，单位为米每秒。
        max_climb_speed: 允许的最大爬升或下降速度，单位为米每秒。
        max_acceleration: 允许的最大合加速度，单位为米每二次方秒。
        shortcut_resolution: shortcut 线段碰撞检测分辨率，单位为米。
        control_point_spacing: B 样条控制点的基础间距，单位为米。
        collision_sample_spacing: 平滑曲线碰撞复检的采样间距，单位为米。
        arc_sample_spacing: 弧长与速度规划使用的曲线采样间距，单位为米。
        smoothing_strengths: B 样条平滑失败时依次尝试的平滑强度序列。
        control_spacing_scales: 平滑失败时用于增加控制点密度的间距缩放序列。
        low_speed_yaw_threshold: 低于该水平速度时保持上一偏航角的阈值。
    
    输出：
        构造并返回 `TrajectoryGenerationConfig` 实例。
    """

    dt: float = 0.1
    cruise_speed: float = 7.0
    max_climb_speed: float = 1.5
    max_acceleration: float = 2.0
    shortcut_resolution: float = 0.5
    control_point_spacing: float = 4.0
    collision_sample_spacing: float = 0.25
    arc_sample_spacing: float = 0.25
    smoothing_strengths: Sequence[float] = (1.5, 0.8, 0.35, 0.1, 0.0)
    control_spacing_scales: Sequence[float] = (1.0, 0.65, 0.4)
    low_speed_yaw_threshold: float = 0.15

    def __post_init__(self) -> None:
        """
        校验数据类构造参数是否合法，并在参数错误时抛出异常。
        
        输入：
            无。
        
        输出：
            无；参数不合法时抛出 ValueError。
        """
        positive = {
            "dt": self.dt,
            "cruise_speed": self.cruise_speed,
            "max_climb_speed": self.max_climb_speed,
            "max_acceleration": self.max_acceleration,
            "shortcut_resolution": self.shortcut_resolution,
            "control_point_spacing": self.control_point_spacing,
            "collision_sample_spacing": self.collision_sample_spacing,
            "arc_sample_spacing": self.arc_sample_spacing,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.low_speed_yaw_threshold < 0.0:
            raise ValueError("low_speed_yaw_threshold must be non-negative")
        if len(self.smoothing_strengths) == 0:
            raise ValueError("smoothing_strengths must not be empty")
        if len(self.control_spacing_scales) == 0:
            raise ValueError("control_spacing_scales must not be empty")
        if any(value < 0.0 for value in self.smoothing_strengths):
            raise ValueError("smoothing strengths must be non-negative")
        if any(value <= 0.0 for value in self.control_spacing_scales):
            raise ValueError("control spacing scales must be positive")


@dataclass(frozen=True)
class ReferenceTrajectory:
    """
    保存按控制周期采样的位置、速度、加速度和偏航参考。
    
    输入：
        time: 时间序列，单位为秒。
        raw_path: RRT* 生成的原始折线路径。
        shortcut_path: 经过碰撞安全 shortcut 简化后的路径。
        smooth_path: 密集采样的无碰撞 B 样条几何路径。
        arc_length: 各时刻对应的累计路径弧长。
        position_planning: 按控制周期采样的 [x, y, h] 规划坐标位置。
        position: 三维位置向量。
        velocity: 三维速度向量。
        acceleration: 动力学坐标系中的三维参考加速度。
        yaw: 经过连续展开处理的参考偏航角序列。
        yaw_rate: 参考偏航角速度序列。
        speed: 各弧长采样点的标量速度。
        spline_smoothing: 最终通过碰撞检测的 B 样条平滑强度。
        spline_control_spacing: 最终通过碰撞检测的 B 样条控制点间距。
    
    输出：
        构造并返回 `ReferenceTrajectory` 实例。
    """

    time: np.ndarray
    raw_path: np.ndarray
    shortcut_path: np.ndarray
    smooth_path: np.ndarray
    arc_length: np.ndarray
    position_planning: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: np.ndarray
    yaw_rate: np.ndarray
    speed: np.ndarray
    spline_smoothing: float
    spline_control_spacing: float

    @property # 这个是获取输出的方法，property意思是它可以像访问属性一样访问这个方法，而不需要加括号
    def outputs(self) -> np.ndarray:
        """
        组合位置、速度、零滚转俯仰和偏航角，生成九维 NMPC 参考。
        
        输入：
            无。
        
        输出：
            形状为 (N, 9) 的 NMPC 路径参考数组。
        """

        zeros = np.zeros((len(self.time), 2), dtype=float) # 代表了零滚转俯仰和偏航角
        return np.column_stack((self.position, self.velocity, zeros, self.yaw))

    def horizon_outputs(self, index: int, horizon: int) -> np.ndarray:
        """
        提取指定起始位置后的预测域参考，并在轨迹末端保持终值。
        
        输入：
            index: 轨迹采样起始索引。
            horizon: NMPC 预测步数。
        
        输出：
            形状为 (horizon + 1, 输出维数) 的参考数组。
        """

        if horizon <= 0:
            raise ValueError("horizon must be positive")
        indices = np.clip(
            np.arange(index, index + horizon + 1, dtype=int),
            0,
            len(self.time) - 1,
        )
        return self.outputs[indices]


@dataclass(frozen=True)
class _SplineCurve:
    """
    保存弧长参数化 B 样条的内部表示和采样数据。
    
    输入：
        tck: SciPy B 样条节点、系数和次数的内部表示。
        parameter: B 样条密集采样使用的参数数组。
        arc_length: 各时刻对应的累计路径弧长。
        sampled_path: B 样条密集采样得到的三维路径点。
        smoothing_strength: 当前 B 样条使用的平滑强度。
        control_spacing: 当前 B 样条控制点间距。
    
    输出：
        构造并返回 `_SplineCurve` 实例。
    """
    tck: tuple
    parameter: np.ndarray
    arc_length: np.ndarray
    sampled_path: np.ndarray
    smoothing_strength: float
    control_spacing: float

    @property
    def length(self) -> float:
        """
        返回当前 B 样条曲线的总弧长。
        
        输入：
            无。
        
        输出：
            B 样条总弧长。
        """
        return float(self.arc_length[-1])

    def parameter_at_arc(self, arc: np.ndarray) -> np.ndarray:
        """
        将弧长坐标插值转换为 B 样条参数。
        
        输入：
            arc: 一个或多个弧长坐标。
        
        输出：
            与输入弧长同形状的 B 样条参数数组。
        """
        return np.interp(arc, self.arc_length, self.parameter) # 线性插值：np.interp(想查询的值, 已知横坐标, 已知纵坐标)

    def evaluate_arc(self, arc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        按弧长计算 B 样条位置、单位切向量和切向量弧长导数。
        
        输入：
            arc: 一个或多个弧长坐标。
        
        输出：
            位置、单位切向量和切向量弧长导数组成的元组。
        """
        parameter = self.parameter_at_arc(np.asarray(arc, dtype=float)) # 位置参数
        position = np.asarray(splev(parameter, self.tck, der=0), dtype=float).T # 位置
        first = np.asarray(splev(parameter, self.tck, der=1), dtype=float).T # 位置的导
        second = np.asarray(splev(parameter, self.tck, der=2), dtype=float).T # 位置的二阶导

        parameter_speed = np.linalg.norm(first, axis=1) # 参数 u 每变化一点，对应弧长 s 变化多少
        parameter_speed = np.maximum(parameter_speed, 1e-9)
        tangent = first / parameter_speed[:, None] # 单位切线方向

        projection = np.sum(first * second, axis=1) # 一阶导数与二阶导数的点积
        tangent_parameter_derivative = (
            second / parameter_speed[:, None]
            - first * (projection / parameter_speed**3)[:, None]
        )
        tangent_arc_derivative = tangent_parameter_derivative / parameter_speed[:, None]  # 曲率向量
        return position, tangent, tangent_arc_derivative
        # position                轨迹位置 r(s)
        # tangent                 单位前进方向 T(s)
        # tangent_arc_derivative  曲率向量 dT/ds

def shortcut_path(
    path: Sequence[Sequence[float]],
    city_map: UrbanMap2_5D,
    resolution: float = 0.5,
) -> np.ndarray:
    """
    使用逐段碰撞检测贪心删除折线路径中的冗余节点。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        resolution: 碰撞检测沿线段采样的空间分辨率。
    
    输出：
        碰撞安全且节点更少的三维路径数组。
    """

    points = remove_redundant_points(np.asarray(path, dtype=float))
    if len(points) <= 2:
        return points.copy()
    if resolution <= 0.0: # 碰撞检测的采样间距
        raise ValueError("resolution must be positive")

    shortcut = [points[0]]
    current = 0
    while current < len(points) - 1:
        next_index = current + 1
        for candidate in range(len(points) - 1, current, -1): # 倒着检查最远可连接点
            if not city_map.edge_collides(
                points[current],
                points[candidate],
                resolution=resolution,
                inflated=True,
            ):
                next_index = candidate
                break
        shortcut.append(points[next_index])
        current = next_index

    return np.asarray(shortcut, dtype=float)


def generate_reference_trajectory(
    path: Sequence[Sequence[float]],
    city_map: UrbanMap2_5D,
    config: TrajectoryGenerationConfig | None = None,
) -> ReferenceTrajectory:
    """
    将无碰撞折线路径转换为平滑、带时间标记的 NMPC 参考轨迹。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        config: 当前算法或控制器配置对象。
    
    输出：
        包含位置、速度、加速度和偏航参考的 ReferenceTrajectory。
    """

    cfg = config or TrajectoryGenerationConfig()
    raw_path = remove_redundant_points(np.asarray(path, dtype=float))
    if raw_path.ndim != 2 or raw_path.shape[1] != 3 or len(raw_path) < 2:
        raise ValueError("path must have shape (N, 3) with N >= 2")
    if any(city_map.collides(point, inflated=True) for point in raw_path):
        raise ValueError("raw path contains a point inside an inflated obstacle")

    shortened = shortcut_path(raw_path, city_map, cfg.shortcut_resolution)
    curve = _fit_collision_free_spline(shortened, city_map, cfg)

    arc_grid = np.linspace(
        0.0,
        curve.length,
        max(2, int(np.ceil(curve.length / cfg.arc_sample_spacing)) + 1),
    )
    path_grid, tangent_grid, tangent_derivative_grid = curve.evaluate_arc(arc_grid)
    curvature = np.linalg.norm(tangent_derivative_grid, axis=1) # 曲率

    # 爬升速度约束
    speed_limit = np.full_like(arc_grid, cfg.cruise_speed)
    vertical_slope = np.abs(tangent_grid[:, 2])
    climb_limited_speed = cfg.max_climb_speed / np.maximum(vertical_slope, 1e-6)
    speed_limit = np.minimum(speed_limit, climb_limited_speed)

    # 曲率约束
    normal_acceleration_limit = 0.65 * cfg.max_acceleration
    curvature_limited_speed = np.sqrt(
        normal_acceleration_limit / np.maximum(curvature, 1e-8)
    )
    speed_limit = np.minimum(speed_limit, curvature_limited_speed)

    # 生成满足加速度约束的速度曲线
    tangential_acceleration_limit = 0.65 * cfg.max_acceleration
    speed_profile = _forward_backward_speed_profile(
        arc_grid,
        speed_limit,
        tangential_acceleration_limit,
    )

    # 得到：弧长—时间
    profile_time = _integrate_profile_time(arc_grid, speed_profile)

    steps = max(1, int(np.ceil(profile_time[-1] / cfg.dt)))
    time = np.arange(steps + 1, dtype=float) * cfg.dt
    arc_at_time = np.interp(
        np.minimum(time, profile_time[-1]),
        profile_time,
        arc_grid,
    )
    scalar_speed = np.interp(arc_at_time, arc_grid, speed_profile)
    scalar_speed[0] = 0.0
    scalar_speed[-1] = 0.0

    position_planning, tangent, tangent_arc_derivative = curve.evaluate_arc(arc_at_time)
    position_planning[0] = raw_path[0]
    position_planning[-1] = raw_path[-1] # 算出每一时刻的规划位置、切向量和切向量弧长导数

    scalar_acceleration = np.gradient(scalar_speed, cfg.dt, edge_order=1)
    velocity_planning = tangent * scalar_speed[:, None] # 计算速度向量
    acceleration_planning = (
        tangent * scalar_acceleration[:, None]
        + tangent_arc_derivative * scalar_speed[:, None] ** 2
    ) # 计算加速度向量
    velocity_planning[[0, -1]] = 0.0

    if not _polyline_is_collision_free( # 再次进行碰撞检测
        position_planning,
        city_map,
        cfg.collision_sample_spacing,
    ):
        raise RuntimeError("time-sampled B-spline trajectory failed collision validation")

    position = position_planning.copy()
    velocity = velocity_planning.copy()
    acceleration = acceleration_planning.copy()
    position[:, 2] *= -1.0 # z是向下为正的
    velocity[:, 2] *= -1.0
    acceleration[:, 2] *= -1.0

    yaw = _yaw_from_velocity(velocity, cfg.low_speed_yaw_threshold)
    # 计算航向角变化率
    yaw_rate = np.gradient(yaw, cfg.dt, edge_order=1)
    yaw_rate[[0, -1]] = 0.0

    return ReferenceTrajectory(
        time=time,
        raw_path=raw_path,
        shortcut_path=shortened,
        smooth_path=curve.sampled_path,
        arc_length=arc_at_time,
        position_planning=position_planning,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        yaw=yaw,
        yaw_rate=yaw_rate,
        speed=scalar_speed,
        spline_smoothing=curve.smoothing_strength,
        spline_control_spacing=curve.control_spacing,
    )


def _fit_collision_free_spline(
    path: np.ndarray,
    city_map: UrbanMap2_5D,
    config: TrajectoryGenerationConfig,
) -> _SplineCurve:
    """
    逐步降低平滑强度并增加控制点，寻找无碰撞三次 B 样条。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        config: 当前算法或控制器配置对象。
    
    输出：
        通过碰撞检测的内部 _SplineCurve 对象。
    """
    for spacing_scale in config.control_spacing_scales:
        spacing = config.control_point_spacing * spacing_scale
        controls = _densify_polyline(path, spacing)
        if len(controls) < 4:
            controls = _resample_polyline(path, 4)
        parameter = _normalized_distance(controls)
        degree = 3
        weights = np.ones(len(controls), dtype=float)
        weights[[0, -1]] = 1e4

        for strength in config.smoothing_strengths:
            smoothing = float(len(controls) * strength**2)
            try:
                tck, _ = splprep(
                    controls.T,
                    u=parameter,
                    w=weights,
                    k=degree,
                    s=smoothing,
                )
            except ValueError:
                continue

            estimate = np.asarray(
                splev(np.linspace(0.0, 1.0, 400), tck),
                dtype=float,
            ).T
            estimated_length = float(
                np.linalg.norm(np.diff(estimate, axis=0), axis=1).sum()
            )
            sample_count = max(
                200,
                int(np.ceil(estimated_length / config.collision_sample_spacing)) + 1,
            )
            dense_parameter = np.linspace(0.0, 1.0, sample_count)
            dense_path = np.asarray(splev(dense_parameter, tck), dtype=float).T
            dense_path[0] = path[0]
            dense_path[-1] = path[-1]

            if not _polyline_is_collision_free(
                dense_path,
                city_map,
                config.collision_sample_spacing,
            ):
                continue

            arc = _cumulative_distance(dense_path)
            valid = np.concatenate(([True], np.diff(arc) > 1e-9))
            if np.count_nonzero(valid) < 4:
                continue
            return _SplineCurve(
                tck=tck,
                parameter=dense_parameter[valid],
                arc_length=arc[valid],
                sampled_path=dense_path[valid],
                smoothing_strength=float(strength),
                control_spacing=float(spacing),
            )

    raise RuntimeError(
        "no collision-free cubic B-spline found; reduce smoothing or increase map clearance"
    )


def _densify_polyline(path: np.ndarray, max_spacing: float) -> np.ndarray:
    """
    按最大点间距在线段上插值，增加路径控制点。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
        max_spacing: 插值后相邻控制点允许的最大距离。
    
    输出：
        点间距不超过指定值的加密路径数组。
    """
    points = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        distance = float(np.linalg.norm(end - start))
        segments = max(1, int(np.ceil(distance / max_spacing)))
        for index in range(1, segments + 1):
            points.append(start + (index / segments) * (end - start))
    return remove_redundant_points(np.asarray(points, dtype=float))


def _resample_polyline(path: np.ndarray, sample_count: int) -> np.ndarray:
    """
    按累计弧长将折线路径重采样为指定数量的点。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
        sample_count: 目标重采样点数量。
    
    输出：
        包含指定数量等弧长采样点的路径数组。
    """
    distance = _cumulative_distance(path)
    sample_distance = np.linspace(0.0, distance[-1], sample_count)
    return np.column_stack(
        [np.interp(sample_distance, distance, path[:, axis]) for axis in range(3)]
    )


def _normalized_distance(path: np.ndarray) -> np.ndarray:
    """
    计算路径点归一化累计距离参数。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
    
    输出：
        从 0 到 1 单调递增的归一化累计距离数组。
    """
    distance = _cumulative_distance(path)
    if distance[-1] <= 0.0:
        raise ValueError("path length must be positive")
    return distance / distance[-1]


def _cumulative_distance(path: np.ndarray) -> np.ndarray:
    """
    计算路径各点对应的累计欧氏距离。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
    
    输出：
        与路径点一一对应的累计距离数组。
    """
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )


def _polyline_is_collision_free(
    path: np.ndarray,
    city_map: UrbanMap2_5D,
    resolution: float,
) -> bool:
    """
    检查路径点及相邻线段是否全部避开膨胀障碍。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        resolution: 碰撞检测沿线段采样的空间分辨率。
    
    输出：
        全部点和线段无碰撞时为 True，否则为 False。
    """
    if any(city_map.collides(point, inflated=True) for point in path):
        return False
    return all(
        not city_map.edge_collides(
            start,
            end,
            resolution=resolution,
            inflated=True,
        )
        for start, end in zip(path[:-1], path[1:])
    )


def _forward_backward_speed_profile(
    arc: np.ndarray,
    speed_limit: np.ndarray,
    acceleration_limit: float,
) -> np.ndarray:
    """
    通过前向加速和后向制动传播生成满足加速度约束的速度曲线。
    
    输入：
        arc: 一个或多个弧长坐标。
        speed_limit: 各弧长采样点允许的最大标量速度。
        acceleration_limit: 允许的最大切向加速度。
    
    输出：
        满足速度和加速度限制的标量速度数组。
    """
    speed = np.maximum(np.asarray(speed_limit, dtype=float), 0.0)
    speed[0] = 0.0
    speed[-1] = 0.0

    for index in range(1, len(speed)):
        distance = arc[index] - arc[index - 1]
        reachable = np.sqrt(max(0.0, speed[index - 1] ** 2 + 2.0 * acceleration_limit * distance))
        speed[index] = min(speed[index], reachable)

    for index in range(len(speed) - 2, -1, -1):
        distance = arc[index + 1] - arc[index]
        stoppable = np.sqrt(max(0.0, speed[index + 1] ** 2 + 2.0 * acceleration_limit * distance))
        speed[index] = min(speed[index], stoppable)

    return speed


def _integrate_profile_time(arc: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """
    根据弧长增量和速度曲线积分得到累计时间。
    
    输入：
        arc: 一个或多个弧长坐标。
        speed: 各弧长采样点的标量速度。
    
    输出：
        各弧长采样点对应的累计时间数组。
    """
    segment_distance = np.diff(arc)
    average_speed = 0.5 * (speed[:-1] + speed[1:])
    segment_time = segment_distance / np.maximum(average_speed, 1e-6)
    return np.concatenate(([0.0], np.cumsum(segment_time)))


def _yaw_from_velocity(velocity: np.ndarray, threshold: float) -> np.ndarray:
    """
    根据水平速度计算连续偏航角，并在低速区保持上一有效方向。
    
    输入：
        velocity: 三维速度向量。
        threshold: 水平速度低速判定阈值。
    
    输出：
        经过 unwrap 处理的连续偏航角数组。
    """
    horizontal_speed = np.linalg.norm(velocity[:, 0:2], axis=1)
    moving = np.flatnonzero(horizontal_speed >= threshold)
    if len(moving) == 0:
        return np.zeros(len(velocity), dtype=float)

    yaw = np.empty(len(velocity), dtype=float)
    first_yaw = float(np.arctan2(velocity[moving[0], 1], velocity[moving[0], 0]))
    previous = first_yaw
    for index in range(len(velocity)):
        if horizontal_speed[index] >= threshold:
            previous = float(np.arctan2(velocity[index, 1], velocity[index, 0]))
        yaw[index] = previous
    return np.unwrap(yaw)

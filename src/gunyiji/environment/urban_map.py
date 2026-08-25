"""A simple 2.5D city map for path-planning validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gunyiji.environment.obstacles import Building


@dataclass(frozen=True)
class MapBounds:
    """
    描述规划空间在 x、y 和高度 h 方向上的取值边界。
    
    输入：
        x_min: 规划地图 x 方向下边界，单位为米。
        x_max: 规划地图 x 方向上边界，单位为米。
        y_min: 规划地图 y 方向下边界，单位为米。
        y_max: 规划地图 y 方向上边界，单位为米。
        h_min: 规划允许的最小飞行高度，单位为米。
        h_max: 规划允许的最大飞行高度，单位为米。
    
    输出：
        构造并返回 `MapBounds` 实例。
    """

    x_min: float = 0.0
    x_max: float = 200.0
    y_min: float = 0.0
    y_max: float = 200.0
    h_min: float = 5.0
    h_max: float = 60.0

    def contains(self, point: Sequence[float]) -> bool:
        """
        判断规划坐标点是否位于地图边界内。
        
        输入：
            point: 规划坐标点 [x, y, h]。
        
        输出：
            点位于地图边界内时为 True，否则为 False。
        """
        x, y, height = np.asarray(point, dtype=float)
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.h_min <= height <= self.h_max
        )

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        将地图上下边界转换为便于采样的两个 NumPy 向量。
        
        输入：
            无。
        
        输出：
            三维下边界和三维上边界数组组成的元组。
        """
        lower = np.array([self.x_min, self.y_min, self.h_min], dtype=float)
        upper = np.array([self.x_max, self.y_max, self.h_max], dtype=float)
        return lower, upper


@dataclass(frozen=True)
class UrbanMap2_5D:
    """
    保存二维建筑物平面与高度信息，并提供膨胀避障检测。
    
    输入：
        bounds: 规划空间的 x、y 和高度边界对象。
        buildings: 城市地图中的建筑物元组。
        horizontal_clearance: 碰撞检测使用的建筑水平安全裕度，单位为米。
        vertical_clearance: 碰撞检测使用的楼顶垂直安全裕度，单位为米。
    
    输出：
        构造并返回 `UrbanMap2_5D` 实例。
    """

    bounds: MapBounds
    buildings: tuple[Building, ...]
    horizontal_clearance: float = 3.0
    vertical_clearance: float = 3.0

    @property
    def inflated_buildings(self) -> tuple[Building, ...]:
        """
        返回应用安全裕度后的全部建筑物。
        
        输入：
            无。
        
        输出：
            应用水平与垂直安全裕度后的建筑物元组。
        """
        return tuple(
            building.inflated(self.horizontal_clearance, self.vertical_clearance)
            for building in self.buildings
        )

    def in_bounds(self, point: Sequence[float]) -> bool:
        """
        判断规划点是否位于地图允许范围内。
        
        输入：
            point: 规划坐标点 [x, y, h]。
        
        输出：
            点位于规划边界内时为 True，否则为 False。
        """
        return self.bounds.contains(point)

    def collides(self, point: Sequence[float], inflated: bool = True) -> bool:
        """
        判断规划点是否越界或与建筑物发生碰撞。
        
        输入：
            point: 规划坐标点 [x, y, h]。
            inflated: 是否使用安全膨胀后的建筑物进行检测。
        
        输出：
            点越界或进入障碍物时为 True，否则为 False。
        """
        if not self.in_bounds(point):
            return True
        obstacles = self.inflated_buildings if inflated else self.buildings
        return any(building.contains_point(point) for building in obstacles)

    def edge_collides(
        self,
        start: Sequence[float],
        end: Sequence[float],
        resolution: float = 0.75,
        inflated: bool = True,
    ) -> bool:
        """
        沿线段采样，判断两规划点之间是否与障碍物碰撞。
        
        输入：
            start: 路径或线段起点 [x, y, h]。
            end: 路径或线段终点 [x, y, h]。
            resolution: 碰撞检测沿线段采样的空间分辨率。
            inflated: 是否使用安全膨胀后的建筑物进行检测。
        
        输出：
            线段越界或穿过障碍物时为 True，否则为 False。
        """
        start_arr = np.asarray(start, dtype=float)
        end_arr = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end_arr - start_arr))
        samples = max(2, int(np.ceil(distance / resolution)) + 1)
        for alpha in np.linspace(0.0, 1.0, samples):
            point = (1.0 - alpha) * start_arr + alpha * end_arr
            if self.collides(point, inflated=inflated):
                return True
        return False


def make_default_city_map() -> UrbanMap2_5D:
    """
    创建用于展示低楼飞越与高楼绕行取舍的默认城市地图。
    
    输入：
        无。
    
    输出：
        配置好边界、建筑物和安全裕度的 UrbanMap2_5D。
    """

    buildings = (
        Building(35.0, 55.0, 20.0, 70.0, 28.0, "B01-left-mid"),
        Building(35.0, 55.0, 90.0, 140.0, 18.0, "B02-left-low"),
        Building(85.0, 115.0, 75.0, 105.0, 20.0, "B03-center-low"),
        Building(120.0, 150.0, 95.0, 135.0, 55.0, "B04-center-high"),
        Building(155.0, 175.0, 40.0, 95.0, 32.0, "B05-right-mid"),
        Building(155.0, 175.0, 115.0, 165.0, 25.0, "B06-right-mid"),
        Building(75.0, 95.0, 130.0, 170.0, 35.0, "B07-upper-mid"),
        Building(105.0, 130.0, 30.0, 60.0, 45.0, "B08-lower-high"),
    )
    return UrbanMap2_5D(bounds=MapBounds(), buildings=buildings)

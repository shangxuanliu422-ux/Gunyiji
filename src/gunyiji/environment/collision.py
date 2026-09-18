"""2.5D 路径规划碰撞检测工具。"""

from __future__ import annotations

from typing import Sequence

from gunyiji.environment.urban_map import UrbanMap2_5D


def is_state_valid(city_map: UrbanMap2_5D, point: Sequence[float]) -> bool:
    """
    判断规划状态是否位于边界内且不碰撞膨胀障碍。
    
    输入：
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        point: 规划坐标点 [x, y, h]。
    
    输出：
        状态合法且无碰撞时为 True，否则为 False。
    """

    return not city_map.collides(point, inflated=True)


def is_edge_valid(
    city_map: UrbanMap2_5D,
    start: Sequence[float],
    end: Sequence[float],
    resolution: float = 0.75,
) -> bool:
    """
    判断两个规划状态之间的直线段是否无碰撞。
    
    输入：
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        start: 路径或线段起点 [x, y, h]。
        end: 路径或线段终点 [x, y, h]。
        resolution: 碰撞检测沿线段采样的空间分辨率。
    
    输出：
        线段无碰撞时为 True，否则为 False。
    """

    return not city_map.edge_collides(start, end, resolution=resolution, inflated=True)

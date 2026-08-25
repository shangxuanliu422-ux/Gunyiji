"""Obstacle primitives for 2.5D urban planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Building:
    """
    描述城市地图中轴对齐的长方体建筑物。
    
    输入：
        xmin: 建筑物在 x 方向的最小边界，单位为米。
        xmax: 建筑物在 x 方向的最大边界，单位为米。
        ymin: 建筑物在 y 方向的最小边界，单位为米。
        ymax: 建筑物在 y 方向的最大边界，单位为米。
        height: 建筑物高度，单位为米。
        name: 对象名称或校验字段名称。
    
    输出：
        构造并返回 `Building` 实例。
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    height: float
    name: str = ""

    def __post_init__(self) -> None:
        """
        校验数据类构造参数是否合法，并在参数错误时抛出异常。
        
        输入：
            无。
        
        输出：
            无；参数不合法时抛出 ValueError。
        """
        if self.xmin >= self.xmax:
            raise ValueError("xmin must be smaller than xmax")
        if self.ymin >= self.ymax:
            raise ValueError("ymin must be smaller than ymax")
        if self.height <= 0.0:
            raise ValueError("building height must be positive")

    @property
    def center(self) -> np.ndarray:
        """
        返回建筑物顶面中心点坐标。
        
        输入：
            无。
        
        输出：
            建筑物顶面中心 [x, y, height]。
        """
        return np.array(
            [0.5 * (self.xmin + self.xmax), 0.5 * (self.ymin + self.ymax), self.height],
            dtype=float,
        )

    @property
    def size(self) -> tuple[float, float, float]:
        """
        返回建筑物在 x、y 和高度方向上的尺寸。
        
        输入：
            无。
        
        输出：
            建筑物宽度、深度和高度组成的元组。
        """
        return self.xmax - self.xmin, self.ymax - self.ymin, self.height

    def inflated(self, horizontal_margin: float, vertical_margin: float) -> "Building":
        """
        按水平和垂直安全裕度生成膨胀后的建筑物。
        
        输入：
            horizontal_margin: 建筑物水平方向安全膨胀距离。
            vertical_margin: 建筑物顶部安全膨胀距离。
        
        输出：
            加入指定安全裕度后的新 Building 对象。
        """

        return Building(
            xmin=self.xmin - horizontal_margin,
            xmax=self.xmax + horizontal_margin,
            ymin=self.ymin - horizontal_margin,
            ymax=self.ymax + horizontal_margin,
            height=self.height + vertical_margin,
            name=self.name,
        )

    def contains_point(self, point: Sequence[float]) -> bool:
        """
        判断规划坐标点是否位于建筑物体积内部。
        
        输入：
            point: 规划坐标点 [x, y, h]。
        
        输出：
            点位于建筑体积内时为 True，否则为 False。
        """

        x, y, height = np.asarray(point, dtype=float)
        return (
            self.xmin <= x <= self.xmax
            and self.ymin <= y <= self.ymax
            and 0.0 <= height <= self.height
        )

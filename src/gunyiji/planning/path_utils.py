"""Path utility functions for [x, y, h] planning coordinates."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def path_length(path: np.ndarray) -> float:
    """
    计算三维折线路径的欧氏总长度。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
    
    输出：
        路径欧氏总长度，单位为米。
    """

    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def height_change(path: np.ndarray) -> float:
    """
    计算路径累计爬升与下降高度的绝对值之和。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
    
    输出：
        累计高度变化绝对值，单位为米。
    """

    if len(path) < 2:
        return 0.0
    return float(np.abs(np.diff(path[:, 2])).sum())


def path_cost(path: np.ndarray, height_weight: float = 2.0) -> float:
    """
    计算路径长度与高度变化惩罚组成的总代价。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
        height_weight: 路径代价中的高度变化惩罚权重。
    
    输出：
        加入高度变化惩罚后的标量路径代价。
    """

    return path_length(path) + height_weight * height_change(path)


def planning_to_dynamics_path(path: Sequence[Sequence[float]]) -> np.ndarray:
    """
    将高度向上的规划坐标转换为 z 轴向下的动力学坐标。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
    
    输出：
        z=-h 的动力学坐标路径数组。
    """

    converted = np.asarray(path, dtype=float).copy()
    converted[:, 2] *= -1.0
    return converted


def remove_redundant_points(path: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """
    删除路径中相邻且距离小于容差的重复点。
    
    输入：
        path: 由 [x, y, h] 路径点组成的二维数组。
        tolerance: 判断相邻点重复时使用的距离容差。
    
    输出：
        删除相邻重复点后的路径数组。
    """

    if len(path) < 2:
        return path
    keep = [0]
    for idx in range(1, len(path)):
        if np.linalg.norm(path[idx] - path[keep[-1]]) > tolerance:
            keep.append(idx)
    return path[keep]

"""RRT and RRT* planners for 2.5D urban maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from gunyiji.environment.urban_map import UrbanMap2_5D

PlannerMode = Literal["rrt", "rrt_star"]


@dataclass
class RRTNode:
    """
    保存 RRT 树中的单个节点、父节点索引和累计代价。
    
    输入：
        point: 规划坐标点 [x, y, h]。
        parent: 当前树节点的父节点索引；根节点使用 -1。
        cost: 从起点累计到当前节点或结果路径的总代价。
    
    输出：
        构造并返回 `RRTNode` 实例。
    """

    point: np.ndarray
    parent: int
    cost: float


@dataclass(frozen=True)
class PlannerConfig:
    """
    保存 RRT 与 RRT* 的采样、扩展、碰撞检测和代价参数。
    
    输入：
        mode: 轨迹线的绘制模式。
        max_iterations: 规划器允许执行的最大采样迭代次数。
        step_size: 采样树每次扩展允许的最大三维距离，单位为米。
        goal_sample_rate: 直接采样终点的概率。
        goal_tolerance: 允许尝试连接终点的距离阈值，单位为米。
        neighbor_radius: RRT* 选择父节点和重连时的邻域半径。
        collision_resolution: 规划器线段碰撞检测的采样分辨率。
        height_weight: 路径代价中的高度变化惩罚权重。
        random_seed: 随机数生成器种子，用于复现实验结果。
    
    输出：
        构造并返回 `PlannerConfig` 实例。
    """

    mode: PlannerMode = "rrt_star"
    max_iterations: int = 1800
    step_size: float = 5.0
    goal_sample_rate: float = 0.12
    goal_tolerance: float = 5.0
    neighbor_radius: float = 13.0
    collision_resolution: float = 0.75
    height_weight: float = 2.2
    random_seed: int = 7

    def __post_init__(self) -> None:
        """
        校验数据类构造参数是否合法，并在参数错误时抛出异常。
        
        输入：
            无。
        
        输出：
            无；参数不合法时抛出 ValueError。
        """
        if self.mode not in ("rrt", "rrt_star"):
            raise ValueError("mode must be 'rrt' or 'rrt_star'")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if not 0.0 <= self.goal_sample_rate <= 1.0:
            raise ValueError("goal_sample_rate must be in [0, 1]")
        if self.goal_tolerance <= 0.0:
            raise ValueError("goal_tolerance must be positive")
        if self.neighbor_radius <= 0.0:
            raise ValueError("neighbor_radius must be positive")
        if self.collision_resolution <= 0.0:
            raise ValueError("collision_resolution must be positive")
        if self.height_weight < 0.0:
            raise ValueError("height_weight must be non-negative")


@dataclass(frozen=True)
class TreeSnapshot:
    """
    保存某次迭代的规划树快照，供生长动画使用。
    
    输入：
        iteration: 当前规划迭代编号。
        points: 规划树所有节点的三维坐标数组。
        parents: 规划树所有节点的父节点索引数组。
        best_path: 当前累计代价最低的起点到终点路径。
    
    输出：
        构造并返回 `TreeSnapshot` 实例。
    """

    iteration: int
    points: np.ndarray
    parents: np.ndarray
    best_path: np.ndarray | None = None


@dataclass(frozen=True)
class PlannerResult:
    """
    汇总路径规划结果、树节点、动画历史和路径代价。
    
    输入：
        mode: 轨迹线的绘制模式。
        success: 算法或优化求解是否成功的标志。
        path: 由 [x, y, h] 路径点组成的二维数组。
        dynamics_path: 将高度 h 转换为 z=-h 后的动力学坐标路径。
        nodes: 当前规划树节点序列。
        snapshots: 规划过程中保存的树结构与最佳路径快照。
        iterations: 规划器实际执行的迭代次数。
        cost: 从起点累计到当前节点或结果路径的总代价。
        message: 描述规划结果状态的文本。
    
    输出：
        构造并返回 `PlannerResult` 实例。
    """

    mode: PlannerMode
    success: bool
    path: np.ndarray
    dynamics_path: np.ndarray
    nodes: tuple[RRTNode, ...]
    snapshots: tuple[TreeSnapshot, ...]
    iterations: int
    cost: float
    message: str


class RRTPlanner:
    """
    在 2.5D 膨胀城市障碍环境中执行 RRT 或 RRT* 路径规划。
    
    输入：
        city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
        config: 当前算法或控制器配置对象。
    
    输出：
        构造并返回 `RRTPlanner` 实例。
    """

    def __init__(self, city_map: UrbanMap2_5D, config: PlannerConfig | None = None) -> None:
        """
        初始化对象并保存运行所需的模型、配置或随机数生成器。
        
        输入：
            city_map: 包含边界、建筑物和安全膨胀信息的 2.5D 城市地图。
            config: 当前算法或控制器配置对象。
        
        输出：
            无；完成实例内部状态初始化。
        """
        self.city_map = city_map
        self.config = config or PlannerConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        snapshot_stride: int = 100,
    ) -> PlannerResult:
        """
        从起点向终点构建采样树，并返回 RRT 或 RRT* 规划结果。
        
        输入：
            start: 路径或线段起点 [x, y, h]。
            goal: 路径规划终点 [x, y, h]。
            snapshot_stride: 每隔多少次规划迭代保存一帧树快照。
        
        输出：
            包含路径、树节点、快照和代价的 PlannerResult。
        """

        if snapshot_stride <= 0:
            raise ValueError("snapshot_stride must be positive")

        start_arr = np.asarray(start, dtype=float)
        goal_arr = np.asarray(goal, dtype=float)
        self._validate_endpoint("start", start_arr)
        self._validate_endpoint("goal", goal_arr)

        nodes: list[RRTNode] = [RRTNode(point=start_arr, parent=-1, cost=0.0)]
        snapshots: list[TreeSnapshot] = [self._snapshot(nodes, iteration=0)]
        best_goal_idx: int | None = None
        best_goal_cost = np.inf

        for iteration in range(1, self.config.max_iterations + 1):
            sample = self._sample(goal_arr)
            nearest_idx = self._nearest(nodes, sample)
            new_point = self._steer(nodes[nearest_idx].point, sample)

            if np.linalg.norm(new_point - nodes[nearest_idx].point) <= 1e-9:
                if iteration % snapshot_stride == 0:
                    best_path = (
                        self._extract_path(nodes, best_goal_idx)
                        if best_goal_idx is not None
                        else None
                    )
                    snapshots.append(
                        self._snapshot(nodes, iteration=iteration, best_path=best_path)
                    )
                continue

            if self.city_map.edge_collides(
                nodes[nearest_idx].point,
                new_point,
                resolution=self.config.collision_resolution,
                inflated=True,
            ):
                if iteration % snapshot_stride == 0:
                    best_path = (
                        self._extract_path(nodes, best_goal_idx)
                        if best_goal_idx is not None
                        else None
                    )
                    snapshots.append(
                        self._snapshot(nodes, iteration=iteration, best_path=best_path)
                    )
                continue

            parent_idx = nearest_idx
            new_cost = nodes[nearest_idx].cost + self._edge_cost(nodes[nearest_idx].point, new_point)

            if self.config.mode == "rrt_star":
                parent_idx, new_cost = self._choose_parent(nodes, new_point, parent_idx, new_cost)

            nodes.append(RRTNode(point=new_point, parent=parent_idx, cost=new_cost))
            new_idx = len(nodes) - 1

            if self.config.mode == "rrt_star":
                self._rewire(nodes, new_idx)

            if np.linalg.norm(new_point - goal_arr) <= 1e-9:
                goal_idx = new_idx
            else:
                goal_idx = self._try_connect_goal(
                    nodes,
                    new_idx,
                    goal_arr,
                    existing_goal_idx=best_goal_idx,
                )
            if goal_idx is not None and nodes[goal_idx].cost < best_goal_cost:
                best_goal_idx = goal_idx
                best_goal_cost = nodes[goal_idx].cost
            elif best_goal_idx is not None:
                best_goal_cost = nodes[best_goal_idx].cost

            if iteration % snapshot_stride == 0:
                best_path = self._extract_path(nodes, best_goal_idx) if best_goal_idx is not None else None
                snapshots.append(
                    self._snapshot(nodes, iteration=iteration, best_path=best_path)
                )

            if self.config.mode == "rrt" and best_goal_idx is not None:
                break

        success = best_goal_idx is not None
        if success:
            path = self._extract_path(nodes, best_goal_idx)
            message = "path found"
        else:
            nearest_goal_idx = self._nearest(nodes, goal_arr)
            path = self._extract_path(nodes, nearest_goal_idx)
            message = "goal not reached; returning nearest partial path"

        if len(snapshots) == 0 or not np.array_equal(snapshots[-1].points, self._points(nodes)):
            snapshots.append(
                self._snapshot(
                    nodes,
                    iteration=iteration,
                    best_path=path if success else None,
                )
            )
        elif success and snapshots[-1].best_path is None:
            snapshots.append(self._snapshot(nodes, iteration=iteration, best_path=path))

        dynamics_path = path.copy()
        dynamics_path[:, 2] *= -1.0
        return PlannerResult(
            mode=self.config.mode,
            success=success,
            path=path,
            dynamics_path=dynamics_path,
            nodes=tuple(nodes),
            snapshots=tuple(snapshots),
            iterations=iteration,
            cost=float(nodes[best_goal_idx].cost if success else nodes[self._nearest(nodes, goal_arr)].cost),
            message=message,
        )

    def _validate_endpoint(self, name: str, point: np.ndarray) -> None:
        """
        校验起点或终点的维度、地图边界和碰撞状态。
        
        输入：
            name: 对象名称或校验字段名称。
            point: 规划坐标点 [x, y, h]。
        
        输出：
            无。
        """
        if point.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {point.shape}")
        if self.city_map.collides(point, inflated=True):
            raise ValueError(f"{name} is outside bounds or inside an inflated obstacle: {point}")

    def _sample(self, goal: np.ndarray) -> np.ndarray:
        """
        按目标偏置概率采样一个无碰撞规划点。
        
        输入：
            goal: 路径规划终点 [x, y, h]。
        
        输出：
            地图边界内且不碰撞的三维采样点。
        """
        if self.rng.random() < self.config.goal_sample_rate:
            return goal
        lower, upper = self.city_map.bounds.as_arrays()
        for _ in range(100):
            sample = self.rng.uniform(lower, upper)
            if not self.city_map.collides(sample, inflated=True):
                return sample
        return self.rng.uniform(lower, upper)

    def _nearest(self, nodes: Sequence[RRTNode], sample: np.ndarray) -> int:
        """
        根据水平距离和高度惩罚寻找距离采样点最近的树节点。
        
        输入：
            nodes: 当前规划树节点序列。
            sample: 随机采样的规划目标点。
        
        输出：
            距离采样点最近的树节点索引。
        """
        points = self._points(nodes)
        distances = np.linalg.norm(points[:, 0:2] - sample[0:2], axis=1)
        distances += self.config.height_weight * np.abs(points[:, 2] - sample[2])
        return int(np.argmin(distances))

    def _near_indices(self, nodes: Sequence[RRTNode], point: np.ndarray) -> np.ndarray:
        """
        查找位于 RRT* 重连邻域半径内的全部节点索引。
        
        输入：
            nodes: 当前规划树节点序列。
            point: 规划坐标点 [x, y, h]。
        
        输出：
            位于重连邻域内的节点索引数组。
        """
        points = self._points(nodes)
        distances = np.linalg.norm(points - point, axis=1)
        return np.where(distances <= self.config.neighbor_radius)[0]

    def _steer(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        从已有节点朝目标采样点按最大步长扩展新节点。
        
        输入：
            start: 路径或线段起点 [x, y, h]。
            target: 扩展方向的目标点。
        
        输出：
            经过步长和地图边界限制的新节点坐标。
        """
        delta = target - start
        distance = float(np.linalg.norm(delta))
        if distance <= self.config.step_size:
            candidate = target.copy()
        else:
            candidate = start + self.config.step_size * delta / distance

        lower, upper = self.city_map.bounds.as_arrays()
        return np.clip(candidate, lower, upper)

    def _edge_cost(self, start: np.ndarray, end: np.ndarray) -> float:
        """
        计算线段长度与高度变化惩罚组成的边代价。
        
        输入：
            start: 路径或线段起点 [x, y, h]。
            end: 路径或线段终点 [x, y, h]。
        
        输出：
            当前线段的长度与高度惩罚总代价。
        """
        delta = end - start
        euclidean = float(np.linalg.norm(delta))
        climb_penalty = self.config.height_weight * abs(float(delta[2]))
        return euclidean + climb_penalty

    def _choose_parent(
        self,
        nodes: Sequence[RRTNode],
        new_point: np.ndarray,
        default_parent: int,
        default_cost: float,
    ) -> tuple[int, float]:
        """
        在邻域节点中选择使新节点累计代价最小的无碰撞父节点。
        
        输入：
            nodes: 当前规划树节点序列。
            new_point: 准备加入规划树的新节点坐标。
            default_parent: 新节点默认父节点索引。
            default_cost: 使用默认父节点时的新节点累计代价。
        
        输出：
            最佳父节点索引及对应累计代价组成的元组。
        """
        best_parent = default_parent
        best_cost = default_cost
        for idx in self._near_indices(nodes, new_point):
            node = nodes[int(idx)]
            if self.city_map.edge_collides(
                node.point,
                new_point,
                resolution=self.config.collision_resolution,
                inflated=True,
            ):
                continue
            candidate_cost = node.cost + self._edge_cost(node.point, new_point)
            if candidate_cost < best_cost:
                best_parent = int(idx)
                best_cost = candidate_cost
        return best_parent, best_cost

    def _rewire(self, nodes: list[RRTNode], new_idx: int) -> None:
        """
        尝试通过新节点重连邻域节点以降低其累计代价。
        
        输入：
            nodes: 当前规划树节点序列。
            new_idx: 新加入节点在节点列表中的索引。
        
        输出：
            无。
        """
        new_node = nodes[new_idx]
        for idx in self._near_indices(nodes, new_node.point):
            idx = int(idx)
            if idx == new_idx or idx == new_node.parent:
                continue
            node = nodes[idx]
            candidate_cost = new_node.cost + self._edge_cost(new_node.point, node.point)
            if candidate_cost >= node.cost:
                continue
            if self.city_map.edge_collides(
                new_node.point,
                node.point,
                resolution=self.config.collision_resolution,
                inflated=True,
            ):
                continue
            nodes[idx] = RRTNode(point=node.point, parent=new_idx, cost=candidate_cost)
            self._propagate_cost_to_children(nodes, idx)

    def _propagate_cost_to_children(self, nodes: list[RRTNode], parent_idx: int) -> None:
        """
        父节点代价变化后递归更新所有后代节点代价。
        
        输入：
            nodes: 当前规划树节点序列。
            parent_idx: 父节点在节点列表中的索引。
        
        输出：
            无。
        """
        for idx, node in enumerate(nodes):
            if node.parent != parent_idx:
                continue
            parent = nodes[parent_idx]
            new_cost = parent.cost + self._edge_cost(parent.point, node.point)
            nodes[idx] = RRTNode(point=node.point, parent=parent_idx, cost=new_cost)
            self._propagate_cost_to_children(nodes, idx)

    def _try_connect_goal(
        self,
        nodes: list[RRTNode],
        from_idx: int,
        goal: np.ndarray,
        existing_goal_idx: int | None,
    ) -> int | None:
        """
        尝试将新节点无碰撞连接到唯一终点节点。
        
        输入：
            nodes: 当前规划树节点序列。
            from_idx: 尝试连接终点的起始节点索引。
            goal: 路径规划终点 [x, y, h]。
            existing_goal_idx: 当前唯一终点节点索引；尚未建立时为 None。
        
        输出：
            终点节点索引；无法连接时为 None。
        """
        from_point = nodes[from_idx].point
        if np.linalg.norm(from_point - goal) > self.config.goal_tolerance:
            return None
        if self.city_map.edge_collides(
            from_point,
            goal,
            resolution=self.config.collision_resolution,
            inflated=True,
        ):
            return None
        goal_cost = nodes[from_idx].cost + self._edge_cost(from_point, goal)
        if existing_goal_idx is not None:
            if goal_cost < nodes[existing_goal_idx].cost:
                nodes[existing_goal_idx] = RRTNode(
                    point=goal.copy(),
                    parent=from_idx,
                    cost=goal_cost,
                )
                self._propagate_cost_to_children(nodes, existing_goal_idx)
            return existing_goal_idx

        nodes.append(RRTNode(point=goal.copy(), parent=from_idx, cost=goal_cost))
        return len(nodes) - 1

    def _extract_path(self, nodes: Sequence[RRTNode], node_idx: int) -> np.ndarray:
        """
        沿父节点索引回溯并生成从起点到指定节点的路径。
        
        输入：
            nodes: 当前规划树节点序列。
            node_idx: 需要回溯路径的目标节点索引。
        
        输出：
            从根节点到指定节点的三维路径数组。
        """
        points = []
        idx = node_idx
        while idx >= 0:
            node = nodes[idx]
            points.append(node.point)
            idx = node.parent
        points.reverse()
        return np.asarray(points, dtype=float)

    @staticmethod
    def _points(nodes: Sequence[RRTNode]) -> np.ndarray:
        """
        将节点序列中的三维坐标提取为 NumPy 数组。
        
        输入：
            nodes: 当前规划树节点序列。
        
        输出：
            形状为 (N, 3) 的规划树节点坐标数组。
        """
        return np.asarray([node.point for node in nodes], dtype=float)

    @staticmethod
    def _parents(nodes: Sequence[RRTNode]) -> np.ndarray:
        """
        将节点序列中的父节点索引提取为整数数组。
        
        输入：
            nodes: 当前规划树节点序列。
        
        输出：
            长度为 N 的父节点索引数组。
        """
        return np.asarray([node.parent for node in nodes], dtype=int)

    def _snapshot(
        self,
        nodes: Sequence[RRTNode],
        iteration: int,
        best_path: np.ndarray | None = None,
    ) -> TreeSnapshot:
        """
        复制当前规划树和最佳路径，生成动画快照。
        
        输入：
            nodes: 当前规划树节点序列。
            iteration: 当前规划迭代编号。
            best_path: 当前累计代价最低的起点到终点路径。
        
        输出：
            复制当前树状态得到的 TreeSnapshot。
        """
        return TreeSnapshot(
            iteration=iteration,
            points=self._points(nodes),
            parents=self._parents(nodes),
            best_path=None if best_path is None else best_path.copy(),
        )

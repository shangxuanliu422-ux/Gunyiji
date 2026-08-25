import numpy as np

from gunyiji.environment import MapBounds, UrbanMap2_5D
from gunyiji.planning import PlannerConfig, RRTPlanner


def test_rrt_star_keeps_one_goal_node_and_no_zero_length_path_segments() -> None:
    """
    验证 RRT* 仅保留一个终点节点且最终路径不存在零长度线段。
    
    输入：
        无。
    
    输出：
        无；断言失败时由测试框架报告错误。
    """
    city_map = UrbanMap2_5D(
        bounds=MapBounds(
            x_min=0.0,
            x_max=20.0,
            y_min=0.0,
            y_max=20.0,
            h_min=1.0,
            h_max=10.0,
        ),
        buildings=(),
    )
    planner = RRTPlanner(
        city_map,
        PlannerConfig(
            mode="rrt_star",
            max_iterations=20,
            step_size=5.0,
            goal_sample_rate=1.0,
            goal_tolerance=5.0,
            neighbor_radius=8.0,
            collision_resolution=0.5,
            height_weight=2.0,
            random_seed=7,
        ),
    )

    goal = np.array([10.0, 0.0, 2.0])
    result = planner.plan([0.0, 0.0, 2.0], goal, snapshot_stride=5)

    assert result.success
    assert np.all(np.linalg.norm(np.diff(result.path, axis=0), axis=1) > 1e-9)
    goal_node_count = sum(np.allclose(node.point, goal) for node in result.nodes)
    assert goal_node_count == 1

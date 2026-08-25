import numpy as np

from gunyiji.environment import Building, MapBounds, UrbanMap2_5D
from gunyiji.planning import (
    TrajectoryGenerationConfig,
    generate_reference_trajectory,
    shortcut_path,
)


def test_generates_collision_free_timed_reference() -> None:
    """
    验证轨迹生成结果无碰撞、端点正确、起终点速度为零且坐标转换正确。
    
    输入：
        无。
    
    输出：
        无；断言失败时由测试框架报告错误。
    """
    city_map = UrbanMap2_5D(
        bounds=MapBounds(
            x_min=0.0,
            x_max=12.0,
            y_min=-5.0,
            y_max=5.0,
            h_min=1.0,
            h_max=10.0,
        ),
        buildings=(Building(4.0, 6.0, -1.0, 1.0, 3.0, "test"),),
        horizontal_clearance=0.5,
        vertical_clearance=0.5,
    )
    raw_path = np.array(
        [
            [0.0, 0.0, 2.0],
            [3.0, 0.0, 5.0],
            [7.0, 0.0, 5.0],
            [10.0, 0.0, 2.0],
        ]
    )
    config = TrajectoryGenerationConfig(
        dt=0.1,
        cruise_speed=3.0,
        max_climb_speed=1.0,
        max_acceleration=1.5,
        control_point_spacing=1.0,
        collision_sample_spacing=0.1,
        arc_sample_spacing=0.1,
    )

    shortened = shortcut_path(raw_path, city_map, resolution=0.1)
    trajectory = generate_reference_trajectory(raw_path, city_map, config)

    assert len(shortened) <= len(raw_path)
    assert trajectory.outputs.shape == (len(trajectory.time), 9)
    assert np.allclose(trajectory.position_planning[0], raw_path[0])
    assert np.allclose(trajectory.position_planning[-1], raw_path[-1])
    assert np.allclose(trajectory.velocity[[0, -1]], 0.0)
    assert np.allclose(trajectory.position[:, 2], -trajectory.position_planning[:, 2])
    assert np.all(np.isfinite(trajectory.yaw))
    assert all(
        not city_map.edge_collides(a, b, resolution=0.05, inflated=True)
        for a, b in zip(
            trajectory.position_planning[:-1],
            trajectory.position_planning[1:],
        )
    )

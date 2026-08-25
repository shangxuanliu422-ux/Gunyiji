from gunyiji.control import NMPCConfig, PathNMPCConfig


def test_default_and_path_reference_dimensions_are_compatible() -> None:
    """
    验证旧六维 NMPC 与路径九维 NMPC 的跟踪状态配置保持兼容。
    
    输入：
        无。
    
    输出：
        无；断言失败时由测试框架报告错误。
    """
    default = NMPCConfig()
    path = PathNMPCConfig()

    assert default.output_size == 6
    assert tuple(default.tracked_state_indices) == (0, 1, 2, 6, 7, 8)
    assert path.output_size == 9
    assert tuple(path.tracked_state_indices) == (0, 1, 2, 3, 4, 5, 6, 7, 8)

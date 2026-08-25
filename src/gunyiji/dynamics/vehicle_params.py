"""Vehicle parameters for the simplified rolling-wing dynamics model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True) # frozen=True 表示该数据类是不可变的，一旦实例化后，其属性值不能被修改。
# 只是实例化之后就不能改了，而不是说类本身不能改，类本身可以更改参数，改完再实例化就行
class VehicleParams:
    """
    保存小型滚翼飞行器的质量、惯量、重力和虚拟控制输入限幅。
    
    输入：
        mass: 飞行器质量，单位为千克。
        ix: 绕机体 x 轴的转动惯量，单位为千克平方米。
        iy: 绕机体 y 轴的转动惯量，单位为千克平方米。
        iz: 绕机体 z 轴的转动惯量，单位为千克平方米。
        gravity: 重力加速度，单位为米每二次方秒。
        min_fx: 前向虚拟力允许的最小值，单位为牛。
        max_fx: 前向虚拟力允许的最大值，单位为牛。
        min_fz: 垂向虚拟力允许的最小值，单位为牛。
        max_fz: 垂向虚拟力允许的最大值，单位为牛。
        max_tau_x: 绕机体 x 轴虚拟力矩的绝对值上限。
        max_tau_y: 绕机体 y 轴虚拟力矩的绝对值上限。
        max_tau_z: 绕机体 z 轴虚拟力矩的绝对值上限。
    
    输出：
        构造并返回 `VehicleParams` 实例。
    """

    mass: float = 1.8
    ix: float = 0.035
    iy: float = 0.045
    iz: float = 0.065
    gravity: float = 9.80665

    min_fx: float = -15.0
    max_fx: float = 15.0
    min_fz: float = -45.0
    max_fz: float = 8.0
    max_tau_x: float = 1.2
    max_tau_y: float = 1.2
    max_tau_z: float = 0.8

    def __post_init__(self) -> None:
        """
        校验数据类构造参数是否合法，并在参数错误时抛出异常。
        
        输入：
            无。
        
        输出：
            无；参数不合法时抛出 ValueError。
        """
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.ix <= 0.0 or self.iy <= 0.0 or self.iz <= 0.0:
            raise ValueError("all principal inertias must be positive")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if self.min_fx > self.max_fx:
            raise ValueError("min_fx must be <= max_fx")
        if self.min_fz > self.max_fz:
            raise ValueError("min_fz must be <= max_fz")
        if self.max_tau_x <= 0.0 or self.max_tau_y <= 0.0 or self.max_tau_z <= 0.0:
            raise ValueError("torque limits must be positive")

    @property
    # 使用@property，调用时：params.inertia
    # 不使用：params.inertia()
    def inertia(self) -> tuple[float, float, float]: # 类型注解，表示返回一个包含三个浮点数的元组
        """
        返回飞行器三个主惯量组成的元组。
        
        输入：
            无。
        
        输出：
            按 [Ix, Iy, Iz] 排列的三个主惯量元组。
        """
        return self.ix, self.iy, self.iz

    @property
    def hover_input(self) -> tuple[float, float, float, float, float]:
        """
        返回当前模型在水平姿态下抵消重力所需的虚拟输入。
        
        输入：
            无。
        
        输出：
            按 [fx, fz, tau_x, tau_y, tau_z] 排列的悬停输入。
        """

        return 0.0, -self.mass * self.gravity, 0.0, 0.0, 0.0

    @property
    def input_lower_bounds(self) -> tuple[float, float, float, float, float]:
        """
        返回五维虚拟控制输入的下限。
        
        输入：
            无。
        
        输出：
            五维虚拟控制输入下限元组。
        """
        return (
            self.min_fx,
            self.min_fz,
            -self.max_tau_x,
            -self.max_tau_y,
            -self.max_tau_z,
        )

    @property
    def input_upper_bounds(self) -> tuple[float, float, float, float, float]:
        """
        返回五维虚拟控制输入的上限。
        
        输入：
            无。
        
        输出：
            五维虚拟控制输入上限元组。
        """
        return (
            self.max_fx,
            self.max_fz,
            self.max_tau_x,
            self.max_tau_y,
            self.max_tau_z,
        )

DEFAULT_VEHICLE_PARAMS = VehicleParams()
""" DEFAULT_VEHICLE_PARAMS = VehicleParams(mass=1000)
print(DEFAULT_VEHICLE_PARAMS.mass) """
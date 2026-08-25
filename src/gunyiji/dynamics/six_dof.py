"""Simplified 12-state rolling-wing flight dynamics.

State order:
    [x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot]

Virtual input order:
    [fx, fz, tau_x, tau_y, tau_z]

The earth frame is NED-like: z is positive downward, so gravity enters as +g
in the z acceleration. At level hover, the virtual input is approximately
fx = 0 and fz = -mass * gravity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin
from typing import Sequence

import numpy as np

from gunyiji.dynamics.vehicle_params import DEFAULT_VEHICLE_PARAMS, VehicleParams

STATE_SIZE = 12
INPUT_SIZE = 5

STATE_NAMES = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "phi",
    "theta",
    "psi",
    "phi_dot",
    "theta_dot",
    "psi_dot",
)

INPUT_NAMES = ("fx", "fz", "tau_x", "tau_y", "tau_z")


def as_state_vector(state: Sequence[float]) -> np.ndarray:
    """
    将输入转换并校验为十二维浮点状态向量。
    
    输入：
        state: 十二维飞行器状态向量。
    
    输出：
        形状为 (12,) 的浮点状态向量。
    """

    vector = np.asarray(state, dtype=float)
    if vector.shape != (STATE_SIZE,):
        raise ValueError(f"state must have shape ({STATE_SIZE},), got {vector.shape}")
    return vector


def as_input_vector(virtual_input: Sequence[float]) -> np.ndarray:
    """
    将输入转换并校验为五维虚拟控制向量。
    
    输入：
        virtual_input: 五维虚拟控制输入 [fx, fz, tau_x, tau_y, tau_z]。
    
    输出：
        形状为 (5,) 的浮点虚拟控制向量。
    """

    vector = np.asarray(virtual_input, dtype=float)
    if vector.shape != (INPUT_SIZE,):
        raise ValueError(f"virtual_input must have shape ({INPUT_SIZE},), got {vector.shape}")
    return vector


def make_state(
    position: Sequence[float] = (0.0, 0.0, 0.0),
    velocity: Sequence[float] = (0.0, 0.0, 0.0),
    attitude: Sequence[float] = (0.0, 0.0, 0.0),
    attitude_rate: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray: # 这里直接初始化是因为默认参数是不可变的元组，所以不会有可变对象作为默认参数的问题。
    """
    由位置、速度、欧拉角和欧拉角速度拼接十二维状态。
    
    输入：
        position: 三维位置向量。
        velocity: 三维速度向量。
        attitude: 滚转、俯仰、偏航欧拉角向量，单位为弧度。
        attitude_rate: 欧拉角速度向量，单位为弧度每秒。
    
    输出：
        拼接完成的十二维状态向量。
    """

    parts = (
        np.asarray(position, dtype=float),
        np.asarray(velocity, dtype=float),
        np.asarray(attitude, dtype=float),
        np.asarray(attitude_rate, dtype=float),
    )
    for name, part in zip(
        ("position", "velocity", "attitude", "attitude_rate"), parts, strict=True
    ):
        if part.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {part.shape}")
    return np.concatenate(parts) # 主要是为了检查错误


def make_virtual_input(
    fx: float = 0.0,
    fz: float | None = None, # fz 可以为 None，表示使用悬停值
    tau_x: float = 0.0,
    tau_y: float = 0.0,
    tau_z: float = 0.0,
    params: VehicleParams = DEFAULT_VEHICLE_PARAMS,
) -> np.ndarray:
    """
    构造五维虚拟控制输入；未指定垂向力时默认使用悬停值。
    
    输入：
        fx: 机体系 x 轴方向虚拟力，单位为牛。
        fz: 机体系 z 轴方向虚拟力；未指定时使用悬停值，单位为牛。
        tau_x: 绕机体 x 轴的虚拟力矩，单位为牛米。
        tau_y: 绕机体 y 轴的虚拟力矩，单位为牛米。
        tau_z: 绕机体 z 轴的虚拟力矩，单位为牛米。
        params: 飞行器物理参数对象。
    
    输出：
        五维虚拟控制输入向量。
    """

    if fz is None:
        fz = -params.mass * params.gravity
    return np.array([fx, fz, tau_x, tau_y, tau_z], dtype=float)


def clip_virtual_input(
    virtual_input: Sequence[float],
    params: VehicleParams = DEFAULT_VEHICLE_PARAMS,
) -> np.ndarray:
    """
    依据飞行器参数对虚拟控制输入进行逐元素限幅。
    
    输入：
        virtual_input: 五维虚拟控制输入 [fx, fz, tau_x, tau_y, tau_z]。
        params: 飞行器物理参数对象。
    
    输出：
        满足飞行器输入上下限的五维控制向量。
    """

    lower = np.asarray(params.input_lower_bounds, dtype=float)
    upper = np.asarray(params.input_upper_bounds, dtype=float)
    return np.clip(as_input_vector(virtual_input), lower, upper)


def translational_acceleration(
    attitude: Sequence[float],
    virtual_force: Sequence[float],
    params: VehicleParams = DEFAULT_VEHICLE_PARAMS,
) -> np.ndarray:
    """
    根据姿态和虚拟力计算地固坐标系平动加速度。
    
    输入：
        attitude: 滚转、俯仰、偏航欧拉角向量，单位为弧度。
        virtual_force: 机体系前向力和垂向力 [fx, fz]。
        params: 飞行器物理参数对象。
    
    输出：
        地固坐标系三维平动加速度向量。
    """

    phi, theta, psi = np.asarray(attitude, dtype=float)
    fx, fz = np.asarray(virtual_force, dtype=float)
    mass = params.mass

    c_phi = cos(phi)
    s_phi = sin(phi)
    c_theta = cos(theta)
    s_theta = sin(theta)
    c_psi = cos(psi)
    s_psi = sin(psi)

    ax = (c_psi * c_theta) * fx / mass
    ax += (c_psi * s_theta * c_phi + s_psi * s_phi) * fz / mass

    ay = (s_psi * c_theta) * fx / mass
    ay += (s_psi * s_theta * c_phi - c_psi * s_phi) * fz / mass

    az = -s_theta * fx / mass
    az += (c_phi * c_theta) * fz / mass
    az += params.gravity

    return np.array([ax, ay, az], dtype=float)


def attitude_acceleration(
    attitude_rate: Sequence[float],
    virtual_torque: Sequence[float],
    params: VehicleParams = DEFAULT_VEHICLE_PARAMS,
) -> np.ndarray:
    """
    根据角速度、虚拟力矩和惯量计算欧拉角加速度。
    
    输入：
        attitude_rate: 欧拉角速度向量，单位为弧度每秒。
        virtual_torque: 三个虚拟控制力矩 [tau_x, tau_y, tau_z]。
        params: 飞行器物理参数对象。
    
    输出：
        滚转、俯仰和偏航角加速度向量。
    """

    phi_dot, theta_dot, psi_dot = np.asarray(attitude_rate, dtype=float)
    tau_x, tau_y, tau_z = np.asarray(virtual_torque, dtype=float)
    ix, iy, iz = params.inertia

    phi_ddot = -((iz - iy) / ix) * theta_dot * psi_dot + tau_x / ix
    theta_ddot = -((ix - iz) / iy) * phi_dot * psi_dot + tau_y / iy
    psi_ddot = -((iy - ix) / iz) * theta_dot * phi_dot + tau_z / iz

    return np.array([phi_ddot, theta_ddot, psi_ddot], dtype=float)


def state_derivative(
    t: float,
    state: Sequence[float],
    virtual_input: Sequence[float],
    params: VehicleParams = DEFAULT_VEHICLE_PARAMS,
) -> np.ndarray:
    """
    计算十二状态飞行器模型的连续时间状态导数。
    
    输入：
        t: 当前仿真时间，单位为秒。
        state: 十二维飞行器状态向量。
        virtual_input: 五维虚拟控制输入 [fx, fz, tau_x, tau_y, tau_z]。
        params: 飞行器物理参数对象。
    
    输出：
        十二维连续时间状态导数向量。
    """

    del t # del 是为了避免未使用参数的警告，t在这里不被使用。
    # 加上 t 是为了保持函数签名与导数函数的标准形式一致，方便在积分器中使用。
    x = as_state_vector(state)
    u = as_input_vector(virtual_input)

    velocity = x[3:6]
    attitude = x[6:9]
    attitude_rate = x[9:12]

    acceleration = translational_acceleration(attitude, u[0:2], params)
    angular_acceleration = attitude_acceleration(attitude_rate, u[2:5], params)

    dx = np.zeros(STATE_SIZE, dtype=float)
    dx[0:3] = velocity
    dx[3:6] = acceleration
    dx[6:9] = attitude_rate
    dx[9:12] = angular_acceleration
    return dx #  这个dx就是rk4_step函数中k1,k2,k3,k4的计算结果，最终用于更新状态向量。


def output_position_attitude(state: Sequence[float]) -> np.ndarray:
    """
    从十二维状态中提取位置和欧拉角六维输出。
    
    输入：
        state: 十二维飞行器状态向量。
    
    输出：
        按 [x, y, z, phi, theta, psi] 排列的六维输出。
    """

    x = as_state_vector(state)
    return np.concatenate((x[0:3], x[6:9]))


@dataclass(frozen=True)
class RollingWingSixDOF:
    """
    封装滚翼飞行器十二状态六自由度动力学模型。
    
    输入：
        params: 飞行器物理参数对象。
    
    输出：
        构造并返回 `RollingWingSixDOF` 实例。
    """

    params: VehicleParams = field(default_factory=lambda: DEFAULT_VEHICLE_PARAMS)
    # 表示参数用默认值 DEFAULT_VEHICLE_PARAMS 初始化，使用 lambda 是为了避免可变对象作为默认参数的问题。
    
    def derivative(
        self,
        t: float,
        state: Sequence[float],
        virtual_input: Sequence[float],
    ) -> np.ndarray:
        """
        调用当前飞行器参数计算连续时间状态导数。
        
        输入：
            t: 当前仿真时间，单位为秒。
            state: 十二维飞行器状态向量。
            virtual_input: 五维虚拟控制输入 [fx, fz, tau_x, tau_y, tau_z]。
        
        输出：
            十二维连续时间状态导数。
        """
        return state_derivative(t, state, virtual_input, self.params)

    def output(self, state: Sequence[float]) -> np.ndarray:
        """
        从飞行器状态中提取位置和姿态六维输出。
        
        输入：
            state: 十二维飞行器状态向量。
        
        输出：
            当前时刻或状态对应的参考输出向量。
        """
        return output_position_attitude(state)

    def hover_input(self) -> np.ndarray:
        """
        返回当前动力学模型对应的五维悬停虚拟输入。
        
        输入：
            无。
        
        输出：
            按 [fx, fz, tau_x, tau_y, tau_z] 排列的悬停输入。
        """
        return np.asarray(self.params.hover_input, dtype=float)

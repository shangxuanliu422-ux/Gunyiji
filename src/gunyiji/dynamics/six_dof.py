"""滚翼飞行器十二状态刚体动力学模型。

状态排列：
    [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

虚拟输入排列：
    [fx, fz, tau_x, tau_y, tau_z]

地固坐标系采用类似 NED 的约定，z 轴向下为正，因此重力以 +g 的形式
进入 z 轴加速度。水平悬停时，虚拟输入近似为 fx=0、fz=-mass*gravity。
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
    "p",
    "q",
    "r",
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
    body_rate: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray: # 这里直接初始化是因为默认参数是不可变的元组，所以不会有可变对象作为默认参数的问题。
    """
    由位置、速度、欧拉角和机体系角速度拼接十二维状态。
    
    输入：
        position: 三维位置向量。
        velocity: 三维速度向量。
        attitude: 滚转、俯仰、偏航欧拉角向量，单位为弧度。
        body_rate: 机体系角速度 [p, q, r]，单位为弧度每秒。
    
    输出：
        拼接完成的十二维状态向量。
    """

    parts = (
        np.asarray(position, dtype=float),
        np.asarray(velocity, dtype=float),
        np.asarray(attitude, dtype=float),
        np.asarray(body_rate, dtype=float),
    )
    for name, part in zip(
        ("position", "velocity", "attitude", "body_rate"), parts, strict=True
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


def euler_angle_rates(
    attitude: Sequence[float],
    body_rate: Sequence[float],
) -> np.ndarray:
    """
    将机体系角速度转换为 ZYX 欧拉角变化率。

    输入：
        attitude: 欧拉角 [phi, theta, psi]，单位为弧度。
        body_rate: 机体系角速度 [p, q, r]，单位为弧度每秒。

    输出：
        欧拉角变化率 [phi_dot, theta_dot, psi_dot]。

    说明：
        ZYX 欧拉角在 theta=+-90 度处存在坐标奇异性。
    """

    attitude_array = np.asarray(attitude, dtype=float)
    body_rate_array = np.asarray(body_rate, dtype=float)
    if attitude_array.shape != (3,):
        raise ValueError(f"attitude must have shape (3,), got {attitude_array.shape}")
    if body_rate_array.shape != (3,):
        raise ValueError(f"body_rate must have shape (3,), got {body_rate_array.shape}")

    phi, theta, _ = attitude_array
    p, q, r = body_rate_array
    c_phi = cos(phi)
    s_phi = sin(phi)
    c_theta = cos(theta)
    if abs(c_theta) < 1.0e-8:
        raise ValueError("ZYX Euler-angle kinematics are singular at theta=+-90 deg")

    tan_theta = sin(theta) / c_theta
    return np.array(
        [
            p + s_phi * tan_theta * q + c_phi * tan_theta * r,
            c_phi * q - s_phi * r,
            (s_phi * q + c_phi * r) / c_theta,
        ],
        dtype=float,
    )


def body_rate_from_euler_rate(
    attitude: Sequence[float],
    euler_rate: Sequence[float],
) -> np.ndarray:
    """
    将 ZYX 欧拉角变化率转换为机体系角速度。

    输入：
        attitude: 欧拉角 [phi, theta, psi]，单位为弧度。
        euler_rate: 欧拉角变化率 [phi_dot, theta_dot, psi_dot]。

    输出：
        机体系角速度 [p, q, r]。
    """

    attitude_array = np.asarray(attitude, dtype=float)
    euler_rate_array = np.asarray(euler_rate, dtype=float)
    if attitude_array.shape != (3,):
        raise ValueError(f"attitude must have shape (3,), got {attitude_array.shape}")
    if euler_rate_array.shape != (3,):
        raise ValueError(f"euler_rate must have shape (3,), got {euler_rate_array.shape}")

    phi, theta, _ = attitude_array
    phi_dot, theta_dot, psi_dot = euler_rate_array
    return np.array(
        [
            phi_dot - sin(theta) * psi_dot,
            cos(phi) * theta_dot + sin(phi) * cos(theta) * psi_dot,
            -sin(phi) * theta_dot + cos(phi) * cos(theta) * psi_dot,
        ],
        dtype=float,
    )


def body_angular_acceleration(
    body_rate: Sequence[float],
    virtual_torque: Sequence[float],
    params: VehicleParams = DEFAULT_VEHICLE_PARAMS,
) -> np.ndarray:
    """
    根据机体系角速度、虚拟力矩和主惯量计算机体系角加速度。
    
    输入：
        body_rate: 机体系角速度 [p, q, r]，单位为弧度每秒。
        virtual_torque: 三个虚拟控制力矩 [tau_x, tau_y, tau_z]。
        params: 飞行器物理参数对象。
    
    输出：
        机体系角加速度 [p_dot, q_dot, r_dot]。
    """

    p, q, r = np.asarray(body_rate, dtype=float)
    tau_x, tau_y, tau_z = np.asarray(virtual_torque, dtype=float)
    ix, iy, iz = params.inertia

    p_dot = ((iy - iz) / ix) * q * r + tau_x / ix
    q_dot = ((iz - ix) / iy) * p * r + tau_y / iy
    r_dot = ((ix - iy) / iz) * p * q + tau_z / iz

    return np.array([p_dot, q_dot, r_dot], dtype=float)


def attitude_acceleration(
    body_rate: Sequence[float],
    virtual_torque: Sequence[float],
    params: VehicleParams = DEFAULT_VEHICLE_PARAMS,
) -> np.ndarray:
    """兼容旧接口，返回机体系角加速度 [p_dot, q_dot, r_dot]。"""

    return body_angular_acceleration(body_rate, virtual_torque, params)


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
    body_rate = x[9:12]

    acceleration = translational_acceleration(attitude, u[0:2], params)
    attitude_rate = euler_angle_rates(attitude, body_rate)
    angular_acceleration = body_angular_acceleration(body_rate, u[2:5], params)

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

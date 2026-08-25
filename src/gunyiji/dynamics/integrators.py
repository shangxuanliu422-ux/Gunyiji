"""Small integration helpers for continuous-time dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Sequence

import numpy as np

# 以下内容可以删掉，表明类型只是方便人类、IDE 查看
State = Sequence[float] # 表示是一个浮点数序列
Control = Sequence[float]
Derivative = Callable[[float, State, Control], np.ndarray] # 表示是 (时间, 状态, 控制) → 状态导数 的函数
# ndarray 是 N 维数组
ControlLaw = Callable[[float, np.ndarray], Control] # (时间, 状态) → 控制输入 的函数
Method = Literal["euler", "rk4"] # 表示 Method 只能是两个指定字符串之一


def euler_step(
    derivative: Derivative,
    t: float,
    state: State,
    control: Control,
    dt: float,
) -> np.ndarray:
    """
    使用显式欧拉法推进一个仿真时间步。
    
    输入：
        derivative: 连续动力学导数函数。
        t: 当前仿真时间，单位为秒。
        state: 十二维飞行器状态向量。
        control: 当前五维虚拟控制输入或控制律。
        dt: 离散控制或积分时间步长，单位为秒。
    
    输出：
        推进一个时间步后的状态向量。
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    x = np.asarray(state, dtype=float) # 不管传入列表、元组还是数组，都把它统一看作浮点 NumPy 数组，并尽量避免不必要的复制。
    return x + dt * derivative(t, x, control)


def rk4_step(
    derivative: Derivative,
    t: float,
    state: State,
    control: Control,
    dt: float,
) -> np.ndarray:
    """
    使用四阶 Runge-Kutta 方法推进一个仿真时间步。
    
    输入：
        derivative: 连续动力学导数函数。
        t: 当前仿真时间，单位为秒。
        state: 十二维飞行器状态向量。
        control: 当前五维虚拟控制输入或控制律。
        dt: 离散控制或积分时间步长，单位为秒。
    
    输出：
        推进一个时间步后的状态向量。
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    x = np.asarray(state, dtype=float)
    k1 = derivative(t, x, control)
    k2 = derivative(t + 0.5 * dt, x + 0.5 * dt * k1, control)
    k3 = derivative(t + 0.5 * dt, x + 0.5 * dt * k2, control)
    k4 = derivative(t + dt, x + dt * k3, control)
    # 这里的 t 其实没有用到，主要是为了保持函数签名与导数函数的标准形式一致，方便在积分器中使用。
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def step(
    derivative: Derivative,
    t: float,
    state: State,
    control: Control,
    dt: float,
    method: Method = "rk4",
) -> np.ndarray:
    """
    根据指定积分方法选择欧拉法或四阶 Runge-Kutta 法推进一步。
    
    输入：
        derivative: 连续动力学导数函数。
        t: 当前仿真时间，单位为秒。
        state: 十二维飞行器状态向量。
        control: 当前五维虚拟控制输入或控制律。
        dt: 离散控制或积分时间步长，单位为秒。
        method: 数值积分方法名称。
    
    输出：
        采用指定方法推进一个时间步后的状态向量。
    """

    if method == "euler":
        return euler_step(derivative, t, state, control, dt)
    if method == "rk4":
        return rk4_step(derivative, t, state, control, dt)
    raise ValueError(f"unknown integration method: {method}")


def simulate(
    derivative: Derivative,
    initial_state: State,
    control: Control | ControlLaw,
    dt: float,
    steps: int,
    method: Method = "rk4",
) -> tuple[np.ndarray, np.ndarray]:
    """
    使用恒定控制量或控制律执行多步动力学仿真。
    
    输入：
        derivative: 连续动力学导数函数。
        initial_state: 仿真初始十二维状态。
        control: 当前五维虚拟控制输入或控制律。
        dt: 离散控制或积分时间步长，单位为秒。
        steps: 需要执行的离散仿真步数。
        method: 数值积分方法名称。
    
    输出：
        时间序列与对应状态历史数组组成的元组。
    """

    if steps < 0:
        raise ValueError("steps must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    states = np.zeros((steps + 1, len(initial_state)), dtype=float)
    # steps + 1 是因为我们要存储初始状态和每一步的状态
    time = np.arange(steps + 1, dtype=float) * dt
    states[0] = np.asarray(initial_state, dtype=float)

    for k in range(steps):
        t = time[k]
        x = states[k]
        u = control(t, x) if callable(control) else control
        # callable 判断 control 是否是可调用对象（函数或方法），如果是，则调用它获取控制输入；否则，直接使用 control 作为控制输入。
        # 意思就是如果control是个固定值，就直接用它；如果control是个函数，就调用它来计算当前的控制输入。
        states[k + 1] = step(derivative, t, x, u, dt, method)

    return time, states

""" print(np.zeros(3)) """
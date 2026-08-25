import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1] # 获取项目根目录的路径，即 GUNYIJI 目录的路径
SRC_DIR = PROJECT_ROOT / "src" # 获取 src 目录的路径

if str(SRC_DIR) not in sys.path: # 检查 src 目录是否已经在 sys.path 中，如果不在，则将其添加到 sys.path 的开头
    sys.path.insert(0, str(SRC_DIR))

import gunyiji.dynamics as gd


state = gd.make_state(
    position=[1, 2, 3],
    velocity=[4, 5, 6],
    attitude=[0.1, 0.2, 0.3],
    attitude_rate=[0.4, 0.5, 0.6],
)

assert state.shape == (12,) # assert 语句用于检查条件是否为真，如果条件为假，则会抛出 AssertionError 异常。这里检查 state 的形状是否为 (12,)，即长度为 12 的一维数组。
assert np.allclose(state[0:3], [1, 2, 3]) # allclose 函数用于判断两个数组是否在一定的容差范围内相等，这里检查 state 的前 3 个元素是否接近 [1, 2, 3]。
assert np.allclose(state[3:6], [4, 5, 6])

model = gd.RollingWingSixDOF()
state = np.zeros(12)
dx = model.derivative(0.0, state, model.hover_input())
print("dx:", dx)
assert np.allclose(dx, 0.0, atol=1e-12)  # atol 是绝对误差容差，表示允许的最大绝对误差。这里检查 dx 是否接近 0.0，允许的最大绝对误差为 1e-12.

aaa = gd.simulate(
    derivative=model.derivative,
    initial_state=state,
    # control=model.hover_input(), # 只给一个升力，其他全为0
    control=[0, 0, 0.0, 0.0, 0.0],
    dt=0.01,
    steps=100
)
print(aaa[1][20])

acc = gd.translational_acceleration(
    attitude=[0, 0, 0],
    virtual_force=[3.6, 0],
)
print("acc:", acc)

acc = gd.translational_acceleration(
    attitude=[0, 0, np.pi / 2],
    virtual_force=[3.6, -1.8 * 9.80665],
)

assert np.isclose(acc[0], 0.0, atol=1e-12)
assert np.isclose(acc[1], 2.0)

acc = gd.attitude_acceleration(
    attitude_rate=[0, 0, 0],
    virtual_torque=gd.DEFAULT_VEHICLE_PARAMS.inertia, # 使用默认的飞行器参数中的惯量作为虚拟力矩输入
)
print("attitude_acceleration:", acc)

time, states = gd.simulate(
    model.derivative,
    initial_state=np.zeros(12),
    control=model.hover_input(),
    dt=0.01,
    steps=5000,
)

assert np.max(np.abs(states)) < 1e-9
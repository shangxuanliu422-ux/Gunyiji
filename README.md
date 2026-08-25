# Gunyiji：滚翼飞行器路径规划与 NMPC 仿真平台

Gunyiji 是一个使用 Python 搭建的滚翼飞行器算法验证项目，覆盖简化六自由度动力学、非线性模型预测控制（NMPC）、四滚翼控制分配、执行机构动态、城市低空路径规划、轨迹平滑及时标化等环节。

项目当前已经跑通两条主要验证链路：

```text
螺旋线控制验证：
参考轨迹 → NMPC → 非线性控制分配 → 一阶执行机构 → 六自由度动力学

城市低空自主飞行：
2.5D 城市地图 → RRT/RRT* → Shortcut → 三次 B 样条
→ 碰撞复检 → 弧长/时间参数化 → NMPC → 六自由度动力学
```

> 本项目定位为算法与数值仿真验证平台，当前结果不代表真机飞行精度或嵌入式实时性能。

## 主要功能

### 1. 滚翼飞行器动力学

- 十二状态简化六自由度模型；
- 状态为位置、速度、欧拉角及欧拉角变化率；
- 控制量为前向力、垂向力和三轴力矩；
- 采用类似 NED 的坐标约定，动力学坐标中 `z` 向下为正，高度满足 `h = -z`；
- 支持 Euler 和四阶 Runge-Kutta（RK4）数值积分；
- 包含悬停、自由落体、坐标变换及单轴力矩等测试。

状态和控制量分别为：

```text
x = [x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot]
u = [fx, fz, tau_x, tau_y, tau_z]
```

### 2. 非线性模型预测控制

- 使用 CasADi 建模、IPOPT 求解非线性规划；
- 在预测模型中使用 RK4 离散动力学；
- 支持预测域跟踪误差、终端误差、控制量及控制增量惩罚；
- 使用上一时刻最优解移位构造热启动；
- 每个控制周期只执行最优控制序列的第一项，再根据新状态滚动求解；
- 支持螺旋线六维参考和城市轨迹九维参考。

### 3. 四滚翼控制分配与执行机构

- 使用 CasADi/IPOPT 求解非线性控制分配；
- 将 `[fx, fz, tau_x, tau_y, tau_z]` 映射为四组滚翼转速 `Omega_i` 和偏转角 `beta_i`；
- 约束滚翼转速、偏转角及变化率；
- 使用上一拍分配结果进行热启动；
- 建立电机和偏转机构一阶动态模型；
- 记录期望控制量、分配可实现控制量和执行机构实际控制量。

当前执行机构时间常数为算法验证暂定值：

```text
电机时间常数：      0.025 s
偏转机构时间常数：  0.035 s
```

### 4. 噪声与扰动验证

螺旋线实验支持以下复合扰动：

- 质量偏差 `+5%`；
- 三轴惯量偏差 `+10% / -8% / +12%`；
- 位置、速度、姿态和角速度测量噪声；
- 低通随机外力与三轴随机外力矩；
- 控制分配误差和执行机构动态滞后。

扰动使用固定随机种子，便于重复实验。当前尚未加入 EKF，带噪状态会直接作为 NMPC 的状态反馈。

### 5. 2.5D 城市环境

- 地图水平范围约为 `200 m × 200 m`；
- 建筑物使用带高度的长方体表示；
- 支持飞行高度上下限；
- 支持建筑物水平和垂向安全膨胀；
- 提供点碰撞和三维线段碰撞检测；
- 规划坐标使用 `[x, y, h]`，其中高度 `h` 向上为正。

### 6. RRT 与 RRT*

- RRT 和 RRT* 共用同一规划器，通过配置选择算法；
- 支持目标偏置采样、最近节点搜索和定步长扩展；
- RRT* 支持邻域父节点选择、重连和子节点代价传播；
- 代价函数同时考虑路径长度和高度变化，避免不必要的大幅爬升；
- 规划过程可保存树快照并生成三维生长动画。

### 7. 路径后处理与时标轨迹生成

RRT* 输出的三维折线路径不能直接用于飞行器跟踪，项目采用以下处理流程：

```text
删除重复点
→ Shortcut 简化（每次连接均进行碰撞检测）
→ 三次 B 样条平滑
→ 对膨胀障碍物进行密集碰撞复检
→ 弧长参数化
→ 根据巡航速度、爬升速度、曲率和加速度限制生成速度曲线
→ 前向/后向速度规划，满足起终点零速度
→ 按 NMPC 周期进行时间采样
→ 生成位置、速度、加速度、偏航角及偏航角速度
```

B 样条使用 SciPy 的 `splprep` 和 `splev`。如果平滑曲线发生碰撞，程序会降低平滑强度或增加控制点密度后重新拟合。

## 项目结构

```text
Gunyiji/
├── configs/                         # 参数配置预留
├── docs/                            # 架构说明与推导笔记
├── experiments/                     # 可直接运行的实验脚本
│   ├── run_mpc_tracking.py          # 理想虚拟控制量下的螺旋线跟踪
│   ├── run_mpc_tracking_allocation.py
│   │                                # 螺旋线 + 控制分配 + 执行机构
│   ├── run_mpc_tracking_disturbance.py
│   │                                # 螺旋线复合扰动验证
│   ├── run_rrt_star.py              # RRT/RRT* 规划及树生长动画
│   ├── run_rrt_mpc_tracking.py      # 城市规划、轨迹生成与 NMPC 跟踪
│   └── plot_rrt_mpc_result_plotly.py
│                                    # Plotly 交互式结果页面
├── src/gunyiji/
│   ├── control/
│   │   ├── mpc.py                   # CasADi NMPC
│   │   └── control_allocation.py    # 非线性控制分配与一阶执行机构
│   ├── dynamics/
│   │   ├── six_dof.py               # 十二状态六自由度动力学
│   │   ├── integrators.py           # Euler/RK4 积分
│   │   └── vehicle_params.py        # 飞行器参数和输入约束
│   ├── environment/
│   │   ├── urban_map.py             # 2.5D 城市地图
│   │   ├── obstacles.py             # 长方体建筑物
│   │   └── collision.py             # 碰撞检测接口
│   └── planning/
│       ├── rrt_star.py              # RRT/RRT* 规划器
│       ├── path_utils.py            # 路径代价和坐标转换
│       └── trajectory_generation.py # Shortcut、B 样条及时标轨迹
├── tests/                            # 单元测试与回归测试
├── requirements.txt
└── pyproject.toml
```

`simulation/`、`visualization/`、`utils/` 和部分配置文件目前主要作为后续模块化重构的预留目录。

## 环境安装

建议使用 Python 3.10 或更高版本。

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 运行实验

所有实验都应从项目根目录运行。脚本默认打开 Matplotlib 动画或结果窗口；服务器环境可添加 `--no-show`。

### 1. 理想螺旋线 NMPC 跟踪

```bash
python experiments/run_mpc_tracking.py
```

### 2. 螺旋线控制分配与执行机构验证

```bash
python experiments/run_mpc_tracking_allocation.py
```

使用原实验中的大初始状态偏差：

```bash
python experiments/run_mpc_tracking_allocation.py --perturbed-start
```

### 3. 噪声、模型失配与随机外扰验证

```bash
python experiments/run_mpc_tracking_disturbance.py
```

指定随机种子或无界面运行：

```bash
python experiments/run_mpc_tracking_disturbance.py --seed 21 --no-show
```

### 4. RRT 或 RRT* 路径规划

```bash
python experiments/run_rrt_star.py --planner rrt
python experiments/run_rrt_star.py --planner rrt_star
```

加快树生长动画可增大快照间隔：

```bash
python experiments/run_rrt_star.py --planner rrt_star --snapshot-stride 100
```

### 5. RRT* 轨迹生成与 NMPC 跟踪

```bash
python experiments/run_rrt_mpc_tracking.py
```

无界面运行：

```bash
python experiments/run_rrt_mpc_tracking.py --no-show
```

### 6. 生成 Plotly 交互式结果

需要先运行 `run_rrt_mpc_tracking.py` 生成 NPZ 数据：

```bash
python experiments/plot_rrt_mpc_result_plotly.py
```

输出文件统一写入 `outputs/`。该目录已加入 `.gitignore`，不会提交大体积仿真数据和 HTML 文件。

## 测试

```bash
pytest -q
```

当前测试覆盖：

- 六自由度模型和数值积分；
- NMPC 配置与参考维度；
- RRT* 终点节点及零长度路径检查；
- Shortcut、B 样条、时标轨迹和碰撞安全性；
- 控制分配悬停映射、代表性指令和执行机构动态。

## 已验证结果

以下数据均为数值仿真结果。

### 螺旋线：控制分配与执行机构

在模型匹配、无外扰条件下进行 24 s 仿真：

```text
NMPC 求解：           240 / 240
控制分配求解：       240 / 240
平均位置误差：       0.00017 m
最大位置误差：       0.00879 m
平均姿态误差：       2.91 deg
```

### 螺旋线：复合扰动

加入质量/惯量失配、状态测量噪声及低通随机外力/力矩后进行 24 s 仿真：

```text
NMPC 求解：           240 / 240
控制分配求解：       240 / 240
平均位置误差：       0.0447 m
最大位置误差：       0.0804 m
最终位置误差：       0.0375 m
平均姿态误差：       3.21 deg
```

### 城市路径规划与 NMPC 跟踪

在模型匹配、状态精确可测且无外扰的条件下：

```text
RRT* 原始节点：      18
Shortcut 节点：      6
B 样条采样点：       1008
轨迹总弧长：         约 251.7 m
轨迹持续时间：       44.0 s
NMPC 求解：           440 / 440
最大参考速度：       7.0 m/s
最大参考加速度：     1.835 m/s^2
平均位置误差：       0.00131 m
最大位置误差：       0.0736 m
最终位置误差：       0.0125 m
```

轨迹在仿真过程中未侵入膨胀后的建筑物区域。

## 当前局限与后续工作

- 动力学使用简化欧拉角模型，大姿态机动时应改用机体系角速度和四元数；
- 气动力参数、推力系数及执行机构参数仍是算法验证值，需要 CFD、台架试验和系统辨识；
- 当前控制分配未考虑滚翼间气动干扰、反扭矩、陀螺力矩及执行器故障；
- 城市规划全流程尚未接入控制分配、执行机构动态和随机扰动；
- 噪声实验尚未加入 EKF，也未开展多随机种子的 Monte Carlo 统计；
- NMPC 使用通用 IPOPT 求解器，尚未在目标飞控硬件上验证最坏求解时间；
- 底层 PID、硬件在环和真机实验仍待完成；
- 后续可实现在线线性化 LTV-MPC 或实时迭代 NMPC，并与当前方案比较实时性。

## 说明

本仓库用于控制、规划和轨迹生成算法的学习与验证。仿真指标受模型匹配程度、求解器配置和实验参数影响，不应直接解释为真机性能。

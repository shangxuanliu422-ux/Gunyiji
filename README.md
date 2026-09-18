# Gunyiji：滚翼飞行器规划与控制仿真

基于 Python 的算法验证项目，包含十二状态六自由度模型、城市低空路径规划、轨迹生成、EKF、NMPC、串级 MPC/PID、LQR/LQT、非线性控制分配和执行机构动态。

项目用于数值仿真与方法比较。质量、惯量、推力系数及执行机构参数采用算法验证值，尚未完成实机参数辨识、硬件在环或真机验证。

## 验证流程

```text
螺旋线直接NMPC：
位置/姿态参考 + 状态反馈 → 六自由度NMPC → 垂向积分补偿（扰动实验）
→ 非线性控制分配 → 执行机构 → 飞行器 → 带噪测量/EKF

螺旋线串级控制：
位置/速度/加速度参考 + 平动状态反馈 → 位置MPC / LQR / LQT / 前馈PD
→ 期望加速度 → 力和姿态转换 → 姿态P + 角速度PID
→ 非线性控制分配 → 执行机构 → 飞行器 → 带噪测量/EKF

城市路径：
2.5D地图 → RRT/RRT* → Shortcut → 三次B样条 → 碰撞复检
→ 时间参数化 → 位置/速度/姿态参考 → 六自由度NMPC → 飞行器
```

城市路径实验目前直接施加虚拟力/力矩，尚未接入控制分配、执行机构动态和EKF。不能将螺旋线实验的完整链路视为城市实验已实现的功能。

## 模型与控制器

### 六自由度模型

```text
状态：[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
输入：[fx, fz, tau_x, tau_y, tau_z]
```

- 动力学坐标采用类似NED的约定，z向下为正；规划坐标为 `[x,y,h]`，高度 `h=-z`。
- 使用ZYX欧拉角运动学、刚体平动与转动方程，支持Euler/RK4积分。
- 标称质量1.8 kg，主惯量为0.035、0.045、0.065 kg·m²。

### 直接NMPC

- 使用CasADi建模、IPOPT求解，预测采用RK4离散六自由度动力学。
- 目标包括跟踪误差、偏离标称悬停输入、控制增量和终端误差。
- 约束包括初始状态、离散动力学、虚拟输入限幅及俯仰角±85°限制；没有将障碍物写入NMPC约束。
- 移位热启动，每拍执行控制序列第一项。
- 螺旋线直接NMPC跟踪位置和姿态；城市路径NMPC跟踪位置、速度和姿态。轨迹生成器虽计算加速度，城市NMPC目标并不直接跟踪加速度。

### 串级外环与姿态控制

- 外环10 Hz，姿态P和角速度PID内环默认100 Hz。
- 位置MPC使用六维双积分模型，优化三轴加速度，包含位置、速度、参考加速度和加速度增量代价。
- 外环加速度逐轴限制为x/y方向±2 m/s²、z方向±3 m/s²；增量是软惩罚，没有变化率硬约束。
- 前馈LQR和前馈PD可替换位置MPC，保留下游控制环节。
- 增量预瞄LQT将上一拍加速度加入状态，离线计算Riccati增益，在线计算未来参考产生的仿射项；它不是带约束MPC。
- LQR、LQT、PD对最终加速度做限幅，不保证整段预测满足限制。
- 期望姿态由加速度和参考航向转换得到：滚转用于产生侧向分力，期望俯仰设为零，偏航来自参考轨迹。

### EKF与扰动

- NumPy实现十二状态EKF，不依赖FilterPy；使用RK4预测、中心差分离散雅可比和Joseph协方差更新。
- 当前观测是对全部真实状态添加噪声，观测矩阵为单位阵；尚未实现真实IMU/GNSS原始数据融合。
- 直接NMPC扰动脚本默认质量偏差+15%，惯量偏差依次为+15%、−15%、+15%，并加入测量噪声和低通随机外力/力矩。
- 扰动脚本在NMPC输出后叠加垂向积分补偿，默认使用EKF；可通过 `--no-ekf` 改用原始带噪反馈。
- 串级及外环对比实验使用标称质量、惯量，保留噪声、外扰和共享高度积分，不能当作质量失配对比实验。

### 四滚翼控制分配

- 将五维虚拟控制量映射为四组转速和推力方向角，共八个执行机构指令。
- 推力模型为 `T_i=k_T*Omega_i²`，单翼作用力为 `[T_i*cos(beta_i),0,-T_i*sin(beta_i)]`，力矩由安装位置与力的叉乘计算。
- 非线性优化兼顾力/力矩匹配、指令变化和转速均衡，支持幅值及指令变化率限制。
- 电机和方向角机构采用一阶动态，时间常数暂定0.025 s和0.035 s，并有限幅及速率限制。
- 当前串级对比关闭分配器的指令变化率约束，执行机构模型仍保留速率限制。
- 未建模反扭矩、陀螺力矩及滚翼间气动干扰。

### 城市规划与轨迹生成

- 2.5D建筑环境，支持高度限制、障碍膨胀、点和线段碰撞检测。
- RRT/RRT*支持目标偏置采样；RRT*包含择优父节点、重连和代价传播。
- Shortcut简化后使用SciPy三次B样条平滑，再检查碰撞；必要时调整平滑参数重新拟合。
- 基于弧长、曲率、巡航/爬升速度和加速度限制构造速度曲线，再按控制周期采样。
- 支持Matplotlib绘图和Plotly交互结果页面。

## 安装

建议Python 3.10及以上。从仓库根目录创建环境并安装依赖：

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

下面的 `python` 指已安装依赖的虚拟环境解释器；Windows未激活环境时可替换为 `.\.venv\Scripts\python.exe`。

## 运行实验

所有命令从仓库根目录执行。去掉 `--no-show` 可打开相应结果窗口。

```bash
# 原始螺旋线NMPC
python experiments/run_mpc_tracking.py --no-show

# 加入控制分配与执行机构
python experiments/run_mpc_tracking_allocation.py --no-show

# 参数失配、噪声、随机外扰、EKF和高度积分
python experiments/run_mpc_tracking_disturbance.py --seed 21 --no-show
python experiments/run_mpc_tracking_disturbance.py --seed 21 --no-ekf --no-show

# 直接NMPC与位置MPC/姿态PID架构比较
python experiments/run_mpc_tracking_cascade_pid.py --no-show

# 同一串级架构：EKF与原始带噪反馈比较
python experiments/run_mpc_pid_ekf_comparison.py --noise-scale 10 --no-show

# 统一内环下比较MPC、LQR、增量预瞄LQT、前馈PD
python experiments/run_outer_loop_comparison.py --no-show --sweep

# RRT/RRT*与城市轨迹跟踪
python experiments/run_rrt_star.py --planner rrt --no-show
python experiments/run_rrt_star.py --planner rrt_star --no-show
python experiments/run_rrt_mpc_tracking.py --no-show

# 先运行城市轨迹跟踪，再生成交互页面
python experiments/plot_rrt_mpc_result_plotly.py
```

外环对比支持 `--duration`、`--horizon`、`--seeds`、`--noise-scale`、`--initial-offset`、`--pd-wn`、`--pd-zeta`、`--lqr-r-scale`、`--delta-scale` 和 `--terminal dare|mpc`。使用各脚本的 `--help` 查看完整选项。

结果写入 `outputs/`，包括PNG、NPZ、CSV和HTML。外环比较单独写入 `outputs/outer_loop_comparison/`：

- `metrics.csv`：每种控制器、每个随机种子的指标。
- `histories.npz`：状态、参考、加速度指令、控制量和耗时。
- `config.json`：实验配置、反馈增益、Riccati矩阵及软件版本信息。
- `comparison.png`：第一个随机种子的误差、指令和耗时曲线。
- `tuning_sweep.csv`：启用 `--sweep` 后生成的参数扫描结果。

同一脚本再次运行会覆盖对应同名结果文件。EKF比较脚本的结果文件名仍含 `threefold_noise`，实际噪声倍数以命令参数及NPZ内的 `measurement_noise_scale` 为准。

## 外环对比结果

2026-09-18本地运行：24秒螺旋线、预测60步、外环10 Hz、内环100 Hz，随机种子21/22/23。所有方案使用相同的内环实现、每个种子对应的噪声和扰动数组，保留EKF和高度积分；初始状态在参考轨迹上，质量/惯量无失配。

平均误差、平均耗时取三次运行平均；最大位置误差取三次运行的最坏值。耗时为外环控制器调用，不包含初始化、EKF、内环PID或控制分配。

| 外环 | 平均位置误差 | 最大位置误差 | 平均控制器耗时 |
|---|---:|---:|---:|
| 位置MPC / IPOPT | 2.37 cm | 6.94 cm | 8.270 ms |
| 前馈LQR，沿用原Q/R | 4.24 cm | 11.26 cm | 0.016 ms |
| 增量预瞄LQT，Riccati终端权重 | 2.37 cm | 6.94 cm | 0.329 ms |
| 前馈PD，Kp=4、Kv=4 | 3.10 cm | 10.62 cm | 0.015 ms |

此工况中MPC和增量LQT实际加速度输出未触边，指令最大差约2.03e-6 m/s²，外环计算约加速25倍。完整仿真的平均墙钟耗时从8.28秒降至6.42秒，因此不能将外环加速倍数解释为完整系统加速倍数。有约束激活时，裁剪LQT输出不等价于求解带约束MPC。

普通LQR的默认权重并非独立调优所得，其指令更激进、触边比例约20.7%；上述结果不代表LQR通常劣于PD。所有数据只代表这些仿真工况及本机计时，不代表嵌入式实时性能。

## 外环参数调节

- **PD**：从 `Kp=wn², Kv=2*zeta*wn` 出发。固定 `zeta=1`，逐渐增加 `wn`；再检查不同阻尼下的误差、指令变化及饱和。默认 `wn=2,zeta=1`。
- **普通LQR**：固定状态权重Q，调整输入权重R。增大R通常减小动作强度；过大会使纠偏变慢。反馈增益在初始化时计算。
- **增量LQT**：扩展状态为 `[位置,速度,上一拍加速度]`，决策量为加速度增量，正确处理输入代价展开后的交叉项。增量权重增大通常使指令更平滑，但可能牺牲响应速度。
- **终端权重**：`dare` 使用扩展模型的离散Riccati解；`mpc` 使用原MPC的位置、速度终端权重。普通无限时域LQR不需要另外手调有限时域终端权重。

`--sweep` 采用独立种子7、12秒、初始位置偏差 `[0.5,-0.3,0.2] m`，不自动用测试种子选择增益。局部扫描中，增量权重0.1/1/10倍的平均位置误差分别为8.35/6.84/7.99 cm；PD的 `wn=1/2/3,zeta=1` 分别为17.38/7.93/7.73 cm，但从2提高到3使加速度增量RMS约增大4倍。6秒预测窗口下，原MPC终端权重改为0.1/1/10倍几乎不影响结果。这些数据是敏感性检查，不是全局寻优或独立测试集上的调优结论。

## 项目结构

```text
configs/                  配置文件
experiments/              实验入口、EKF与串级控制对比实现
src/gunyiji/control/      六自由度NMPC、非线性控制分配和执行机构
src/gunyiji/dynamics/     刚体模型、参数、积分器
src/gunyiji/environment/  2.5D地图和碰撞检测
src/gunyiji/planning/     RRT/RRT*、路径后处理和时间参数化
tests/                    数学、接口与回归检查
requirements.txt          运行与测试依赖
pyproject.toml            包配置
```

部分 `simulation/`、`visualization/`、`utils/` 文件仍是模块化预留。个人求职材料、参考论文和本地 `docs/` 笔记不作为仓库交付内容。

## 测试

安装 `requirements.txt` 中的测试依赖后运行：

```bash
python -m pytest -q
```

现有检查覆盖动力学特殊工况、欧拉角运动学、参考维度、规划和轨迹后处理、控制分配及执行机构等。部分动力学工况目前为输出检查，不能等同于完整自动断言覆盖。

新增 `tests/test_outer_loop_comparison.py` 检查：

- 无约束最优序列未触边且终端代价一致时，增量LQT与现有MPC第一拍一致。
- 普通LQR和扩展LQR反馈使相应理想离散模型稳定。
- 前馈与加速度限幅正确。

外环对比开发时已直接调用这三个测试函数并通过断言；这不是全部测试套件的通过声明。

## 当前边界

- 欧拉角在俯仰±90°附近存在奇异性；大姿态场景需要替换姿态表示。
- 标称模型和简化力映射需要独立实测数据验证；仿真跟踪准确不能证明实机模型准确。
- EKF使用带噪全状态观测，尚未建模实际传感器偏置、异步测量及丢帧。
- 外环加速度限幅不能保证完整执行机构可行性；城市避障采用规划、膨胀和复检，没有闭环安全证明。
- 求解器成功标志不等于在控制截止时间内完成；求解失败和连续失败降级处理仍需完善。
- 高度积分为优化器外补偿，没有完整覆盖下游控制分配饱和。
- 直接NMPC与串级PID旧对比同时改变更新频率，不能仅将性能差异归因于PID；新外环对比统一了这些环节。
- 当前三种子外环结果不是跨载荷、强约束和大姿态的全面鲁棒性验证。

## 提交范围

`.gitignore` 排除虚拟环境、缓存、输出数据、日志、本地代理配置、`docs/`、求职材料和根目录临时 `test.py`。保留源码、实验脚本、测试、依赖与README，运行后可重新生成结果。

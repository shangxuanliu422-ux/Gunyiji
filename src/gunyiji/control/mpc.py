"""基于 CasADi 的滚翼飞行器虚拟输入非线性模型预测控制器。"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Sequence

import casadi as ca
import numpy as np

from gunyiji.dynamics.six_dof import (
    INPUT_SIZE,
    STATE_SIZE,
    as_input_vector,
    as_state_vector,
    body_rate_from_euler_rate,
    make_state,
)
from gunyiji.dynamics.vehicle_params import DEFAULT_VEHICLE_PARAMS, VehicleParams

OUTPUT_SIZE = 6
PATH_OUTPUT_SIZE = 9


@dataclass(frozen=True)
class NMPCConfig:
    """
    保存直接非线性模型预测控制器的预测域、权重和求解器参数。
    
    输入：
        dt: 离散控制或积分时间步长，单位为秒。
        horizon: NMPC 预测步数。
        output_weight: 预测域各跟踪输出的阶段误差权重。
        terminal_weight: 预测域末端各跟踪输出的误差权重。
        input_weight: 虚拟控制输入偏离悬停值的惩罚权重。
        input_delta_weight: 相邻控制周期虚拟输入变化量的惩罚权重。
        tracked_state_indices: 从十二维状态中选取为控制器输出的索引序列。
        ipopt_max_iter: IPOPT 单次求解允许的最大迭代次数。
        ipopt_tol: IPOPT 求解收敛容差。
    
    输出：
        构造并返回 `NMPCConfig` 实例。
    """

    dt: float = 0.1
    horizon: int = 15
    output_weight: Sequence[float] = (35.0, 35.0, 55.0, 1.0, 1.0, 3.0)
    terminal_weight: Sequence[float] = (90.0, 90.0, 130.0, 2.0, 2.0, 8.0)
    input_weight: Sequence[float] = (0.015, 0.008, 0.02, 0.02, 0.015)
    input_delta_weight: Sequence[float] = (0.35, 0.25, 0.05, 0.05, 0.04)
    tracked_state_indices: Sequence[int] = (0, 1, 2, 6, 7, 8)
    ipopt_max_iter: int = 80
    ipopt_tol: float = 1e-4

    def __post_init__(self) -> None:
        """
        校验数据类构造参数是否合法，并在参数错误时抛出异常。
        
        输入：
            无。
        
        输出：
            无；参数不合法时抛出 ValueError。
        """
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        tracked_indices = np.asarray(self.tracked_state_indices, dtype=int)
        if tracked_indices.ndim != 1 or len(tracked_indices) == 0:
            raise ValueError("tracked_state_indices must be a non-empty one-dimensional sequence")
        if np.any(tracked_indices < 0) or np.any(tracked_indices >= STATE_SIZE):
            raise ValueError(f"tracked_state_indices entries must be in [0, {STATE_SIZE})")
        if len(np.unique(tracked_indices)) != len(tracked_indices):
            raise ValueError("tracked_state_indices entries must be unique")
        output_size = len(tracked_indices)
        _check_weight("output_weight", self.output_weight, output_size)
        _check_weight("terminal_weight", self.terminal_weight, output_size)
        _check_weight("input_weight", self.input_weight, INPUT_SIZE)
        _check_weight("input_delta_weight", self.input_delta_weight, INPUT_SIZE)

    @property
    def output_size(self) -> int:
        """
        返回控制器实际跟踪的状态分量数量。
        
        输入：
            无。
        
        输出：
            控制器参考输出维数。
        """
        return len(self.tracked_state_indices)


@dataclass(frozen=True)
class PathNMPCConfig(NMPCConfig): # 继承自 NMPCConfig，增加了对路径跟踪的支持
    """
    提供面向规划路径的位置、速度和姿态九维 NMPC 默认权重。
    
    输入：
        horizon: NMPC 预测步数。
        output_weight: 预测域各跟踪输出的阶段误差权重。
        terminal_weight: 预测域末端各跟踪输出的误差权重。
        input_weight: 虚拟控制输入偏离悬停值的惩罚权重。
        input_delta_weight: 相邻控制周期虚拟输入变化量的惩罚权重。
        tracked_state_indices: 从十二维状态中选取为控制器输出的索引序列。
    
    输出：
        构造并返回 `PathNMPCConfig` 实例。
    """

    horizon: int = 25
    output_weight: Sequence[float] = ( # 这几个权重是针对位置、速度和姿态的误差权重，分别对应 x, y, z, vx, vy, vz, roll, pitch, yaw
        32.0,
        32.0,
        48.0,
        6.0,
        6.0,
        9.0,
        0.15,
        0.15,
        4.0,
    )
    terminal_weight: Sequence[float] = (
        85.0,
        85.0,
        125.0,
        12.0,
        12.0,
        18.0,
        0.3,
        0.3,
        10.0,
    )
    input_weight: Sequence[float] = (0.012, 0.008, 0.02, 0.02, 0.012)
    input_delta_weight: Sequence[float] = (0.3, 0.25, 0.05, 0.05, 0.035)
    tracked_state_indices: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True)
class NMPCResult:
    """
    保存一次 NMPC 优化得到的控制量、预测轨迹和求解状态。
    
    输入：
        control: 当前五维虚拟控制输入或控制律。
        predicted_states: NMPC 在预测域内得到的状态序列。
        predicted_controls: NMPC 在预测域内得到的虚拟控制序列。
        objective: 本次 NMPC 优化的目标函数值。
        status: 求解器返回的状态文本。
        success: 算法或优化求解是否成功的标志。
    
    输出：
        构造并返回 `NMPCResult` 实例。
    """

    control: np.ndarray
    predicted_states: np.ndarray
    predicted_controls: np.ndarray
    objective: float
    status: str
    success: bool


class CasadiNMPC:
    """
    使用 CasADi 和 IPOPT 求解虚拟控制输入 NMPC 问题。
    
    输入：
        params: 飞行器物理参数对象。
        config: 当前算法或控制器配置对象。
    
    输出：
        构造并返回 `CasadiNMPC` 实例。
    """

    def __init__(
        self,
        params: VehicleParams = DEFAULT_VEHICLE_PARAMS,
        config: NMPCConfig | None = None, # config 可以为 None，表示使用默认的 NMPCConfig 配置
        # NMPCConfig | None 是类型。相当于默认 config = None
    ) -> None:
        """
        初始化对象并保存运行所需的模型、配置或随机数生成器。
        
        输入：
            params: 飞行器物理参数对象。
            config: 当前算法或控制器配置对象。
        
        输出：
            无；完成实例内部状态初始化。
        """
        self.params = params
        self.config = config or NMPCConfig() # 如果 config 为 None，则使用默认的 NMPCConfig 配置
        self._last_solution: np.ndarray | None = None # _ 表示这是一个私有属性，通常不应该在类的外部直接访问。_last_solution 用于存储上一次求解器的最优解，以便在下一次求解时作为初始猜测，提高求解效率。
        self._build_solver() # 表示直接调用 _build_solver 方法来构建求解器，而不需要在外部调用它。

    def solve(
        self,
        state: Sequence[float],
        previous_input: Sequence[float],
        reference_outputs: np.ndarray,
    ) -> NMPCResult:
        """
        根据当前状态、上一控制量和预测域参考求解一次 NMPC。
        
        输入：
            state: 十二维飞行器状态向量。
            previous_input: 上一时刻施加的五维虚拟控制输入。
            reference_outputs: 预测域内的控制器参考输出数组。
        
        输出：
            包含最优控制量、预测状态、目标函数值和求解状态的 NMPCResult。
        """

        x0 = as_state_vector(state)
        u_prev = as_input_vector(previous_input)
        y_ref = self._format_reference(reference_outputs) # 检查 reference_outputs 的形状是否符合要求，并将其转换为适合求解器使用的格式。

        p = np.concatenate((x0, u_prev, y_ref.T.reshape(-1, order="F")))
        # p 是求解器的参数向量，包含当前状态、上一控制输入和参考输出。y_ref.T.reshape(-1, order="F") 将参考输出矩阵按列优先顺序展平为一维数组，以便与 x0 和 u_prev 连接成一个长向量。
        w0 = self._initial_guess(x0, u_prev) # 产生一个初始猜测

        try:
            solution = self._solver( # _solver 是 CasADi 的求解器对象，调用它会执行优化求解。传入的参数包括初始猜测 w0、变量上下界 lbx 和 ubx、约束上下界 lbg 和 ubg，以及参数向量 p。
                x0=w0,
                lbx=self._lbx,
                ubx=self._ubx,
                lbg=self._lbg,
                ubg=self._ubg,
                p=p,
            )
            w_opt = np.asarray(solution["x"], dtype=float).reshape(-1) # 将求解器返回的最优解转换为一维 NumPy 数组。solution["x"] 是 CasADi 返回的最优解，可能是一个 CasADi 矩阵或其他类型，通过 np.asarray 转换为 NumPy 数组，并使用 reshape(-1) 将其展平为一维数组。
            self._last_solution = w_opt
            status = self._solver.stats().get("return_status", "unknown") 
            # self._solver.stats() 返回一个字典，包含求解器的状态信息。get("return_status", "unknown") 尝试获取 "return_status" 键对应的值，如果该键不存在，则返回默认值 "unknown"。status 变量用于记录求解器的返回状态，以便在 NMPCResult 中报告求解结果。
            success = bool(self._solver.stats().get("success", False))
            objective = float(solution["f"])
            
        except RuntimeError as exc:
            w_opt = w0
            status = f"solver_error: {exc}"
            success = False
            objective = float("nan")

        states, controls = self._unpack_decision(w_opt)
        return NMPCResult(
            control=controls[0].copy(),
            predicted_states=states,
            predicted_controls=controls,
            objective=objective,
            status=status,
            success=success,
        )

    # _build_solver() 只在初始化时执行一次。它负责把 NMPC 的数学问题搭出来。
    def _build_solver(self) -> None:
        """
        构建 CasADi 非线性规划变量、代价函数、约束和 IPOPT 求解器。
        
        输入：
            无。
        
        输出：
            无；求解器及变量上下界保存到实例。
        """
        cfg = self.config
        nx = STATE_SIZE
        nu = INPUT_SIZE
        ny = cfg.output_size
        horizon = cfg.horizon # 预测时域

        # ca.MX.sym(...)：创建符号变量，这都是形状，不是数值。MX 是 CasADi 的符号矩阵类型，适用于非线性优化问题。sym(...) 方法用于创建符号变量，参数包括变量名称、行数和列数。
        x_var = ca.MX.sym("X", nx, horizon + 1) # 状态变量
        u_var = ca.MX.sym("U", nu, horizon) # 控制变量

        # 这仨是参数，不是决策变量。每次求解时都要传入不同的参数值，但不参与优化。
        x0_par = ca.MX.sym("X0", nx) # 初始状态参数
        u_prev_par = ca.MX.sym("U_PREV", nu) # 上一控制输入参数
        y_ref_par = ca.MX.sym("Y_REF", ny, horizon + 1) # 参考输出参数

        # ca.DM(...)：创建数值矩阵，是已知常数
        # ca.diag(...)：把向量变成对角矩阵，用于二次型
        q = ca.diag(ca.DM(cfg.output_weight))
        q_terminal = ca.diag(ca.DM(cfg.terminal_weight))
        r = ca.diag(ca.DM(cfg.input_weight))
        r_delta = ca.diag(ca.DM(cfg.input_delta_weight))
        hover = ca.DM(self.params.hover_input)

        constraints = [x_var[:, 0] - x0_par]
        objective = 0

        for k in range(horizon):
            y_error = self._output_expr(x_var[:, k]) - y_ref_par[:, k]
            u_error = u_var[:, k] - hover
            delta_u = u_var[:, k] - (u_prev_par if k == 0 else u_var[:, k - 1])

            objective += ca.mtimes([y_error.T, q, y_error]) # mtimes 是矩阵乘法
            objective += ca.mtimes([u_error.T, r, u_error])
            objective += ca.mtimes([delta_u.T, r_delta, delta_u])

            x_next = self._rk4_expr(x_var[:, k], u_var[:, k])
            constraints.append(x_var[:, k + 1] - x_next)

        terminal_error = self._output_expr(x_var[:, horizon]) - y_ref_par[:, horizon]
        objective += ca.mtimes([terminal_error.T, q_terminal, terminal_error])

        decision = ca.vertcat(ca.vec(x_var), ca.vec(u_var)) # 决策变量向量，包含所有预测状态和控制输入，便于求解器处理
        
        # 火箭里“没有 p”，不是因为火箭没有参数，而是因为你多数时候做的是离线 OCP，把初始状态、目标轨道、故障工况、发动机参数这些都直接写成常数了。
        # 火箭 OCP：常常一次性搭一个固定问题。
        # MPC：同一个问题反复求，每次只换当前状态、上一拍控制和参考轨迹。
        # p 不是神秘东西，它就是“每次求解时换题目条件，但不想重新写题”的那个题目条件向量。
        parameters = ca.vertcat(x0_par, u_prev_par, ca.vec(y_ref_par)) # parameters = 当前状态 + 上一拍输入 + 整个预测窗口的参考轨迹
        constraint_vec = ca.vertcat(*constraints) # 将约束列表展开为一个向量，便于求解器处理

        nlp = {"x": decision, "f": objective, "g": constraint_vec, "p": parameters}
        options = {
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": cfg.ipopt_max_iter,
                "tol": cfg.ipopt_tol,
                "acceptable_tol": 5.0 * cfg.ipopt_tol,
            },
        }
        self._solver = ca.nlpsol("virtual_input_nmpc", "ipopt", nlp, options)
        # virtual_input_nmpc 是求解器的名字，可以在日志中看到。ipopt 是求解器类型，表示使用 IPOPT 求解器。nlp 是定义的非线性规划问题，包括决策变量、目标函数、约束和参数。options 是求解器的配置选项，包括打印设置、最大迭代次数和收敛容差等。
        
        state_lower = np.full(nx * (horizon + 1), -np.inf, dtype=float) # np.full 创建一个指定形状的数组，并用指定的值填充。这里创建了一个大小为 nx * (horizon + 1) 的数组，所有元素都初始化为负无穷大，表示状态变量没有下界限制。
        state_upper = np.full(nx * (horizon + 1), np.inf, dtype=float)
        # ZYX 欧拉角在俯仰 +-90 度处奇异，限制预测状态远离该坐标奇异点。
        pitch_limit = np.deg2rad(85.0)
        state_lower[7::nx] = -pitch_limit
        state_upper[7::nx] = pitch_limit
        input_lower = np.tile(np.asarray(self.params.input_lower_bounds), horizon) # np.tile 将输入下界数组重复 horizon 次，形成一个长度为 nu * horizon 的数组，表示每个预测步的控制输入下界。
        input_upper = np.tile(np.asarray(self.params.input_upper_bounds), horizon)
        self._lbx = np.concatenate((state_lower, input_lower))
        self._ubx = np.concatenate((state_upper, input_upper))
        self._lbg = np.zeros(nx * (horizon + 1), dtype=float) # _lbg 和 _ubg 分别表示约束的下界和上界，这里设置为零向量，表示所有约束都必须等于零，即状态转移方程必须严格满足。
        self._ubg = np.zeros(nx * (horizon + 1), dtype=float)

        # constraint_vec 只是把约束表达式拼起来；每个约束到底是等式还是不等式，由对应位置的 lbg/ubg 决定。
    
    def _dynamics_expr(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """
        构造 CasADi 符号形式的十二状态连续动力学方程。
        
        输入：
            x: CasADi 符号状态向量。
            u: CasADi 符号虚拟控制向量。
        
        输出：
            十二维 CasADi 连续状态导数符号向量。
        """
        phi = x[6]
        theta = x[7]
        psi = x[8]
        p = x[9]
        q = x[10]
        r_body = x[11]

        fx = u[0]
        fz = u[1]
        tau_x = u[2]
        tau_y = u[3]
        tau_z = u[4]

        mass = self.params.mass
        ix, iy, iz = self.params.inertia

        c_phi = ca.cos(phi)
        s_phi = ca.sin(phi)
        c_theta = ca.cos(theta)
        s_theta = ca.sin(theta)
        c_psi = ca.cos(psi)
        s_psi = ca.sin(psi)

        tan_theta = s_theta / c_theta
        phi_dot = p + s_phi * tan_theta * q + c_phi * tan_theta * r_body
        theta_dot = c_phi * q - s_phi * r_body
        psi_dot = (s_phi * q + c_phi * r_body) / c_theta

        ax = (c_psi * c_theta) * fx / mass
        ax += (c_psi * s_theta * c_phi + s_psi * s_phi) * fz / mass

        ay = (s_psi * c_theta) * fx / mass
        ay += (s_psi * s_theta * c_phi - c_psi * s_phi) * fz / mass

        az = -s_theta * fx / mass
        az += (c_phi * c_theta) * fz / mass
        az += self.params.gravity

        p_dot = ((iy - iz) / ix) * q * r_body + tau_x / ix
        q_dot = ((iz - ix) / iy) * p * r_body + tau_y / iy
        r_dot = ((ix - iy) / iz) * p * q + tau_z / iz

        return ca.vertcat(
            x[3],
            x[4],
            x[5],
            ax,
            ay,
            az,
            phi_dot,
            theta_dot,
            psi_dot,
            p_dot,
            q_dot,
            r_dot,
        ) # 相当于给状态量都求了一导

    def _rk4_expr(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """
        构造 CasADi 符号形式的四阶 Runge-Kutta 离散动力学。
        
        输入：
            x: CasADi 符号状态向量。
            u: CasADi 符号虚拟控制向量。
        
        输出：
            经过一个控制周期后的十二维 CasADi 状态符号向量。
        """
        dt = self.config.dt
        k1 = self._dynamics_expr(x, u)
        k2 = self._dynamics_expr(x + 0.5 * dt * k1, u)
        k3 = self._dynamics_expr(x + 0.5 * dt * k2, u)
        k4 = self._dynamics_expr(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _output_expr(self, x: ca.MX) -> ca.MX:
        """
        按照配置的状态索引提取 NMPC 跟踪输出。
        
        输入：
            x: CasADi 符号状态向量。
        
        输出：
            按配置选取的 CasADi 跟踪输出符号向量。
        """
        return ca.vertcat(*[x[int(index)] for index in self.config.tracked_state_indices])
        # *是解包操作符，将列表中的元素作为单独的参数传递给 ca.vertcat 函数，从而将选定的状态分量垂直拼接成一个新的 CasADi 符号向量。

    def _initial_guess(self, state: np.ndarray, previous_input: np.ndarray) -> np.ndarray:
        """
        根据上一时刻解进行移位，生成当前 NMPC 决策变量初值。
        
        输入：
            state: 十二维飞行器状态向量。
            previous_input: 上一时刻施加的五维虚拟控制输入。
        
        输出：
            展平后的 NMPC 决策变量初始猜测向量。
        """
        nx = STATE_SIZE
        nu = INPUT_SIZE
        horizon = self.config.horizon

        if self._last_solution is None:
            states = np.tile(state, (horizon + 1, 1)) # 形状是 (horizon + 1, STATE_SIZE)，表示在预测域内每个时间步都使用当前状态作为初始猜测。加一是因为预测域包含当前时刻和未来的 horizon 个时刻。
            controls = np.tile(previous_input, (horizon, 1)) # 形状是 (horizon, INPUT_SIZE)，表示在预测域内每个时间步都使用上一时刻的控制输入作为初始猜测。
        else:
            # 初值移位
            # 旧: [X0, X1, X2, X3, X4] [U0, U1, U2, U3]
            # 新: [X_real, X2, X3, X4, X4] [U1, U2, U3, U3]
            states, controls = self._unpack_decision(self._last_solution)
            states[:-1] = states[1:]
            states[-1] = states[-2]
            states[0] = state
            controls[:-1] = controls[1:]
            controls[-1] = controls[-2]

        return np.concatenate(
            (
                # states原先是二维数组，行是时间步，列是状态分量。reshape(-1) 会按行优先展平为一维数组，而 states.T.reshape(-1, order="F") 会按列优先展平为一维数组。这里使用 order="F" 是为了确保在优化器中变量的排列顺序与约束和目标函数的定义一致。
                # states = np.array([
                #   [10, 11, 12],   # 第 0 个时刻的状态 x0
                #   [20, 21, 22],   # 第 1 个时刻的状态 x1
                #   [30, 31, 32],   # 第 2 个时刻的状态 x2
                #   [40, 41, 42],   # 第 3 个时刻的状态 x3
                #   [50, 51, 52]    # 第 4 个时刻的状态 x4)
                states.T.reshape(nx * (horizon + 1), order="F"), # F：按列优先展平
                # 得到的结果是 [X0, X1, X2, X3, X4]，即横着排成一行
                controls.T.reshape(nu * horizon, order="F"),
            )
        )

    def _unpack_decision(self, decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        将一维优化决策向量拆分为预测状态和预测控制序列。
        
        输入：
            decision: 一维 NMPC 优化决策向量。
        
        输出：
            预测状态数组和预测控制数组组成的元组。
        """
        nx = STATE_SIZE
        nu = INPUT_SIZE
        horizon = self.config.horizon
        state_count = nx * (horizon + 1)

        # 前 state_count 个数是状态量，但是是一维的，需要重新组成矩阵形式
        states = decision[:state_count].reshape((nx, horizon + 1), order="F").T
        controls = decision[state_count:].reshape((nu, horizon), order="F").T
        return states, controls

    def _format_reference(self, reference_outputs: np.ndarray) -> np.ndarray:
        """
        将参考数据转换为浮点数组并校验预测域形状。
        
        输入：
            reference_outputs: 预测域内的控制器参考输出数组。
        
        输出：
            完成类型转换和形状校验的参考数组。
        """
        refs = np.asarray(reference_outputs, dtype=float)
        expected = (self.config.horizon + 1, self.config.output_size)
        if refs.shape != expected:
            raise ValueError(f"reference_outputs must have shape {expected}, got {refs.shape}")
        return refs


def _check_weight(name: str, value: Sequence[float], expected_size: int) -> None:
    """
    校验权重向量长度和非负性。
    
    输入：
        name: 对象名称或校验字段名称。
        value: 需要校验的数值序列。
        expected_size: 期望的向量元素数量。
    
    输出：
        无；权重不合法时抛出 ValueError。
    """
    array = np.asarray(value, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {array.shape}")
    if np.any(array < 0.0):
        raise ValueError(f"{name} entries must be non-negative")

@dataclass(frozen=True)
class CircleTrajectory:
    """
    生成用于控制器验证的水平圆形参考轨迹。
    
    输入：
        radius: 圆形或螺旋参考轨迹的水平半径，单位为米。
        angular_rate: 参考轨迹绕中心旋转的角速度，单位为弧度每秒。
        center: 圆形或螺旋参考轨迹的三维中心坐标。
        roll: 人为设置的参考滚转角，单位为弧度。
        pitch: 人为设置的参考俯仰角，单位为弧度。
        yaw_offset: 相对于轨迹参数角的偏航角偏置，单位为弧度。
    
    输出：
        构造并返回 `CircleTrajectory` 实例。
    """

    radius: float = 2.0
    angular_rate: float = 0.25
    center: Sequence[float] = (0.0, 0.0, -1.5)
    roll: float = 0.0
    pitch: float = 0.0
    yaw_offset: float = np.pi / 2.0

    def output(self, t: float) -> np.ndarray:
        """
        计算给定时刻的轨迹位置与姿态参考输出。
        
        输入：
            t: 当前仿真时间，单位为秒。
        
        输出：
            当前时刻或状态对应的参考输出向量。
        """
        angle = self.angular_rate * t
        cx, cy, cz = self.center
        return np.array(
            [
                cx + self.radius * cos(angle),
                cy + self.radius * sin(angle),
                cz,
                self.roll,
                self.pitch,
                angle + self.yaw_offset,
            ],
            dtype=float,
        )

    def velocity(self, t: float) -> np.ndarray:
        """
        计算给定时刻的三维参考速度。
        
        输入：
            t: 当前仿真时间，单位为秒。
        
        输出：
            三维参考速度向量。
        """
        angle = self.angular_rate * t
        return np.array(
            [
                -self.radius * self.angular_rate * sin(angle),
                self.radius * self.angular_rate * cos(angle),
                0.0,
            ],
            dtype=float,
        )

    def state(self, t: float) -> np.ndarray:
        """
        生成给定时刻的完整十二维参考状态。
        
        输入：
            t: 当前仿真时间，单位为秒。
        
        输出：
            十二维参考状态向量。
        """
        output = self.output(t)
        return make_state(
            position=output[0:3],
            velocity=self.velocity(t),
            attitude=output[3:6],
            body_rate=body_rate_from_euler_rate(
                output[3:6],
                (0.0, 0.0, self.angular_rate),
            ),
        )

    def horizon_outputs(self, start_time: float, dt: float, horizon: int) -> np.ndarray:
        """
        按起始时间和控制周期生成预测域内的位置与姿态参考。
        
        输入：
            start_time: 预测域起始时间，单位为秒。
            dt: 离散控制或积分时间步长，单位为秒。
            horizon: NMPC 预测步数。
        
        输出：
            形状为 (horizon + 1, 输出维数) 的参考数组。
        """
        refs = [self.output(start_time + k * dt) for k in range(horizon + 1)]
        return np.asarray(refs, dtype=float)


@dataclass(frozen=True)
class HelicalTrajectory:
    """
    生成符合 z 轴向下约定的螺旋上升参考轨迹。
    
    输入：
        radius: 圆形或螺旋参考轨迹的水平半径，单位为米。
        angular_rate: 参考轨迹绕中心旋转的角速度，单位为弧度每秒。
        climb_rate: 参考轨迹向上的爬升速度，单位为米每秒。
        center: 圆形或螺旋参考轨迹的三维中心坐标。
        roll: 人为设置的参考滚转角，单位为弧度。
        pitch: 人为设置的参考俯仰角，单位为弧度。
        yaw_offset: 相对于轨迹参数角的偏航角偏置，单位为弧度。
    
    输出：
        构造并返回 `HelicalTrajectory` 实例。
    """

    radius: float = 2.0
    angular_rate: float = 0.35
    climb_rate: float = 0.15
    center: Sequence[float] = (0.0, 0.0, -1.0)
    roll: float = 0.0
    pitch: float = 0.0
    yaw_offset: float = np.pi / 2.0

    def output(self, t: float) -> np.ndarray:
        """
        计算给定时刻的轨迹位置与姿态参考输出。
        
        输入：
            t: 当前仿真时间，单位为秒。
        
        输出：
            当前时刻或状态对应的参考输出向量。
        """
        angle = self.angular_rate * t
        cx, cy, cz = self.center
        return np.array(
            [
                cx + self.radius * cos(angle),
                cy + self.radius * sin(angle),
                cz - self.climb_rate * t,
                self.roll,
                self.pitch,
                angle + self.yaw_offset,
            ],
            dtype=float,
        )

    def velocity(self, t: float) -> np.ndarray:
        """
        计算给定时刻的三维参考速度。
        
        输入：
            t: 当前仿真时间，单位为秒。
        
        输出：
            三维参考速度向量。
        """
        angle = self.angular_rate * t
        return np.array(
            [
                -self.radius * self.angular_rate * sin(angle),
                self.radius * self.angular_rate * cos(angle),
                -self.climb_rate,
            ],
            dtype=float,
        )

    def state(self, t: float) -> np.ndarray:
        """
        生成给定时刻的完整十二维参考状态。
        
        输入：
            t: 当前仿真时间，单位为秒。
        
        输出：
            十二维参考状态向量。
        """
        output = self.output(t)
        return make_state(
            position=output[0:3],
            velocity=self.velocity(t),
            attitude=output[3:6],
            body_rate=body_rate_from_euler_rate(
                output[3:6],
                (0.0, 0.0, self.angular_rate),
            ),
        )

    def horizon_outputs(self, start_time: float, dt: float, horizon: int) -> np.ndarray:
        """
        按起始时间和控制周期生成预测域内的位置与姿态参考。
        
        输入：
            start_time: 预测域起始时间，单位为秒。
            dt: 离散控制或积分时间步长，单位为秒。
            horizon: NMPC 预测步数。
        
        输出：
            形状为 (horizon + 1, 输出维数) 的参考数组。
        """
        refs = [self.output(start_time + k * dt) for k in range(horizon + 1)]
        return np.asarray(refs, dtype=float)

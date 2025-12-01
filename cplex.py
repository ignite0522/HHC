import os, sys, copy, time, json
import argparse, math, random
import torch
import pickle
import numpy as np
import pprint as pp
from tqdm import tqdm
from torch.utils.data import DataLoader
from docplex.mp.model import Model
from utils import move_to, load_problem
from docplex.mp.vartype import BinaryVarType, IntegerVarType, ContinuousVarType

cplex_python_bindings_path = "/mnt/usb/projects/1gniT42e/cplex2/cplex/python/3.10/x86-64_linux"
# 如果该路径不在 sys.path 中，则将其添加到 sys.path 的最前面，以确保优先被找到。
if cplex_python_bindings_path not in sys.path:
    sys.path.insert(0, cplex_python_bindings_path)


class CVRPTW(object):
    """
    CVRPTW问题的解决方案类。包含两个主要数据成员：nodes（节点）和edges（边）。
    nodes是节点元组的列表：(id, coords)。edges是从每个节点到其唯一出边节点的映射。

    CVRPTW (Capacitated Vehicle Routing Problem with Time Windows) 是一个经典的组合优化问题，
    涉及多个车队、多个车辆为客户提供服务，同时满足容量约束和时间窗约束。
    """

    def __init__(self, args={}, init_solution=None):
        """
        初始化CVRPTW问题实例

        ** 车辆容量和速度固定为1.0和80.0 **

        参数:
            args (dict): 问题的所有参数，包含以下键值对：
                - "fleets": 车队列表
                - "vehicles": 每个车队的车辆列表的映射
                - "customers": 客户列表
                - "nodes_fleet": 每个车队对应的节点列表
                - "start_early": 每个节点最早开始时间
                - "distance": 节点间距离矩阵
            init_solution (dict, optional): 可选的初始解，包含"x"和"t"键
        """
        super(CVRPTW).__init__()

        # 存储问题参数
        self.args = args

        # 初始化目标函数值为一个很大的数（表示未找到有效解）
        self.obj = 10 ** 8

        # CPLEX优化模型对象（用于精确求解）
        self.model = None

        # 记录解构建所需时间
        self.construct_time = 0

        # 创建车辆-车队对应关系列表
        # 将每个车队的每个车辆与其所属车队配对
        self.vehicle_fleet = [(k, f) for f in self.args["fleets"] for k in self.args["vehicles"][f]]

        if init_solution is not None:
            # 如果提供了初始解，直接设置该解
            self.set_solution(init_solution["x"], init_solution["t"])
        else:
            # 初始化决策变量

            # visit决策变量：(i,j,k,f) -> 0或1
            # 表示车辆k在车队f中是否从节点i行驶到节点j
            # 这是一个四维决策变量，涵盖所有可能的路径选择
            self.visit = {(i, j, k, f): 0
                          for f in self.args["fleets"]  # 遍历所有车队
                          for i in self.args["nodes_fleet"][f]  # 遍历车队f的所有节点
                          for j in self.args["nodes_fleet"][f]  # 遍历车队f的所有节点
                          for k in self.args["vehicles"][f]  # 遍历车队f的所有车辆
                          if i != j}  # 排除自环（节点到自身）

            # time决策变量：(i,f) -> 浮点数
            # 表示车队f在位置i的操作开始时间
            # 初始值设为该位置的最早开始时间
            self.time = {(i, f): 0.0 + args["start_early"][i]
                         for f in self.args["fleets"]  # 遍历所有车队
                         for i in self.args["nodes_fleet"][f]}  # 遍历车队f的所有节点

            # 设置特殊节点的时间约束
            for f in self.args["fleets"]:
                # 起始节点（depot start）：编号为 len(customers) + f*2 - 1
                # 时间设为-60，与 StateAGH 的初始自由时间保持一致
                self.time[len(self.args["customers"]) + f * 2 - 1, f] = -60

                # 结束节点（depot end）：编号为 len(customers) + f*2
                # 时间设为一个很大的数，表示可以在任何时间返回
                self.time[len(self.args["customers"]) + f * 2, f] = 10 ** 8

    def copy(self):
        """
        深拷贝当前解

        返回:
            CVRPTW: 当前对象的完全独立副本
        """
        return copy.deepcopy(self)

    def objective(self):
        """
        计算目标函数值：所有选中边长度的总和（使用欧几里得距离）

        目标函数是所有被选中路径（visit值为1）的距离之和。
        这代表了所有车辆行驶的总距离。

        返回:
            int: 总行驶距离（转换为整数）
        """
        # 获取客户数量（用于理解节点编号规则）
        n = len(self.args["customers"])

        # 计算目标函数值：遍历所有visit决策变量
        # 只有当visit[i,j,k,f] == 1时，该边才被选中，距离才被计入总和
        obj = sum(self.args["distance"][i, j] * self.visit[i, j, k, f]
                  for i, j, k, f in self.visit)

        return int(obj)

    def get_solution(self):
        """
        获取当前解的字典表示

        返回:
            dict: 包含"x"（visit变量）和"t"（time变量）的字典
        """
        return {"x": self.visit, "t": self.time}

    def set_solution(self, visit, t):
        """
        设置新解并更新目标函数值

        参数:
            visit (dict): 新的visit决策变量值
            t (dict): 新的time决策变量值
        """
        self.visit = visit
        self.time = t
        # 更新目标函数值
        self.obj = self.objective()

    def used_vehicle(self):
        """
        计算每个车队使用的车辆数量

        通过检查每个车辆的总行驶距离来判断该车辆是否被使用。
        如果车辆的总行驶距离大于0，则认为该车辆被使用。

        返回:
            dict: 键为车队编号，值为该车队使用的车辆数量
        """
        # 初始化每个车辆的行驶距离为0
        vehicle_distance = {(k, f): 0
                            for f in self.args["fleets"]
                            for k in self.args["vehicles"][f]}

        # 遍历所有visit决策变量
        for key, value in self.visit.items():
            if value != 1:  # 只考虑被选中的边（value=1）
                continue

            # 解析键值：i=起始节点, j=终止节点, k=车辆, f=车队
            i, j, k, f = key[0], key[1], key[2], key[3]

            # 累加该车辆的行驶距离
            vehicle_distance[(k, f)] += self.args["distance"][(i, j)]

        # 统计每个车队使用的车辆数量
        count = {f: 0 for f in self.args["fleets"]}

        # 遍历所有车辆的行驶距离
        for key, value in vehicle_distance.items():
            if value != 0:  # 如果行驶距离不为0，说明该车辆被使用
                # key[-1]是车队编号f
                count[key[-1]] += 1

        return count


def get_initial_sol(args):
    """
    生成一个简化的初始解。
    对于纯CPLEX求解，通常不需要非常复杂的启发式初始解，
    因为CPLEX会自行进行分支定界搜索。这里提供一个最小化的可行解框架。
    """
    n, F, speed = len(args["customers"]), len(args["fleets"]), 80.0

    # 初始化时间变量 t
    # t[i, f] 表示车队 f 在节点 i 的服务开始时间
    t = {(i, f): args["start_early"][i] for f in args["fleets"] for i in args["nodes_fleet"][f]}

    # 根据优先级约束调整时间变量
    for i in args["customers"]:
        for pair in args["precedence"][i]:
            t[i, pair[1]] = max(t[i, pair[1]], t[i, pair[0]] + args["duration"][pair[0]][i - 1])

    # 设置 Depot 节点的初始和结束时间
    for f in args["fleets"]:
        t[n + f * 2 - 1, f] = -60  # 起始 Depot 时间
        t[n + 2 * f, f] = 10 ** 8  # 结束 Depot 时间

    # 简化的车辆分配和路径构建
    # 对于每个车队，每个客户都分配给一个独立的"车辆"（逻辑上，CPLEX会自行优化）
    # 确保每个客户至少被一个车队分配到一个车辆，且每个车队至少有一个车辆。

    # 近似计算每个车队所需的车辆数，这里简化为每个客户一个车辆（上限）或至少1个
    vehicle_number_approximation = {f: 0 for f in args["fleets"]}
    for f in args["fleets"]:
        # 假设每个客户至少被一个车辆服务，所以车辆数至少是客户数的一半或更少
        # 更简单地，我们知道 CPLEX 会自己决定多少车辆，这里给一个初始的上限。
        # 这里设置为一个客户一个车队，或者有客户就至少一个车
        num_customers_for_fleet = len([c for c in args["customers"] if c in args["nodes_fleet"][f]])
        vehicle_number_approximation[f] = max(1, num_customers_for_fleet)

    # 实际使用的车辆集合
    vehicles = {f: [k for k in range(1, vehicle_number_approximation[f] + 1)] for f in args["fleets"]}

    # 构造 visit 字典，默认为0
    # 在这个简化的初始解中，我们不会构建具体的路径，CPLEX 会从头开始。
    # visit 字典仅仅是为了给 CPLEX 变量定义提供一个结构。
    visit = {(i, j, k, f): 0
             for f in args["fleets"]
             for i in args["nodes_fleet"][f]
             for j in args["nodes_fleet"][f]
             for k in vehicles[f]
             if i != j}

    # 更新 args 中的 vehicles 信息，以便 CPLEX 模型使用正确的车辆范围
    args["vehicles"] = vehicles

    init_solution = {"x": visit, "t": t}
    return CVRPTW(args=args, init_solution=init_solution)


def add_constraints(mdl, x, t, cvrptw):
    """
    向CPLEX模型添加约束条件。
    Args:
        mdl (docplex.mp.model.Model): CPLEX 模型实例。
        x (docplex.mp.linear.Var): 决策变量 x[i, j, k, f]，表示车队 f 的车辆 k 是否从节点 i 移动到节点 j。
        t (docplex.mp.linear.Var): 决策变量 t[i, f]，表示车队 f 的车辆到达并开始服务节点 i 的时间。
        cvrptw (object): 包含问题所有参数的对象，如客户信息、车队信息、时间、距离等。
    Returns:
        docplex.mp.model.Model: 添加了约束条件后的 CPLEX 模型实例。
    """
    n = len(cvrptw.args["customers"])  # 客户数量 (不包含 Depot)
    F = len(cvrptw.args["fleets"])  # 车队数量
    nodes_F = cvrptw.args["nodes_fleet"]  # 每个车队可以访问的节点列表
    vehicles = cvrptw.args["vehicles"]  # 车辆信息
    customers = cvrptw.args["customers"]  # 客户节点列表
    fleets = cvrptw.args["fleets"]  # 车队列表
    tau = cvrptw.args["travel_time"]  # 旅行时间
    start_ea = cvrptw.args["start_early"]  # 节点最早开始服务时间（节点级）
    start_la = cvrptw.args["start_late"]  # 节点最晚开始服务时间（按车队）
    S_time = cvrptw.args["duration"]  # 节点的纯服务时长
    problem_need = cvrptw.args["need"]  # 航班的 'need' 类型



    # 新增：按车队的左时间窗（若未提供，则从节点级复制）
    if "start_early_by_fleet" in cvrptw.args:
        start_ea_by_fleet = cvrptw.args["start_early_by_fleet"]  # 结构：{i: {f: left}}
    else:
        # 动态构造一个与 start_la 车队维度一致的左窗映射
        start_ea_by_fleet = {i: {f: start_ea[i] for f in fleets} for i in customers + [n + 2 * f - 1 for f in fleets] + [n + 2 * f for f in fleets]}

    # 不再支持放宽访问，始终强制不可访问边为0

    SPEED = 80.0  # 速度常数
    BIG_M = 10 ** 9  # 大M常数

    # === 新增：服务指示变量 y[i,f] 与访问次数（need=7/8/9 需要两次访问，否则一次） ===
    y = mdl.binary_var_dict([(i, f) for i in customers for f in fleets], name='y')

    for i in customers:
        need_i = problem_need[i - 1]
        required_visits = 2 if need_i in [7, 8, 9] else 1

        # 链接 y 与到达边（每个车队对该客户的到达次数）
        for f in fleets:
            mdl.add_constraint(
                y[i, f] == mdl.sum(x[j, i, k, f] for k in vehicles[f] for j in nodes_F[f] if j != i),
                f"Link_y_arrive_i{i}_f{f}"
            )

        # 到达/离开总次数满足需求（组合=2，单=1）
        mdl.add_constraint(
            mdl.sum(y[i, f] for f in fleets) == required_visits,
            f"ArriveTimes_i{i}"
        )
        mdl.add_constraint(
            mdl.sum(mdl.sum(x[i, j, k, f] for k in vehicles[f] for j in nodes_F[f] if j != i) for f in fleets) == required_visits,
            f"LeaveTimes_i{i}"
        )

    # 不使用分层左窗变量，改为直接的差分时间窗（delta）条件约束


    # --- 1. 完善后的车辆流平衡约束 ---
    for f in fleets:
        for k in vehicles[f]:
            # 1.1 从起始depot出发：每辆车最多从起始depot出发一次
            mdl.add_constraint(
                mdl.sum(x[n + 2 * f - 1, j, k, f] for j in nodes_F[f] if j != (n + 2 * f - 1)) <= 1,
                f"Constraint_DepotStart_f{f}_k{k}"
            )

            # 1.2 到达结束depot：每辆车最多到达结束depot一次
            mdl.add_constraint(
                mdl.sum(x[i, n + f * 2, k, f] for i in nodes_F[f] if i != (n + f * 2)) <= 1,
                f"Constraint_DepotEnd_f{f}_k{k}"
            )

            # 1.3 不能从结束depot出发
            mdl.add_constraint(
                mdl.sum(x[n + 2 * f, j, k, f] for j in nodes_F[f] if j != (n + 2 * f)) == 0,
                f"Constraint_NoDepartFromEndDepot_f{f}_k{k}"
            )

            # 1.4 不能回到起始depot
            mdl.add_constraint(
                mdl.sum(x[i, n + f * 2 - 1, k, f] for i in nodes_F[f] if i != (n + f * 2 - 1)) == 0,
                f"Constraint_NoReturnToStartDepot_f{f}_k{k}"
            )

            # 1.5 车辆使用的一致性：如果车辆被使用，必须从起始depot出发并到达结束depot
            start_from_depot = mdl.sum(x[n + 2 * f - 1, j, k, f] for j in nodes_F[f] if j != (n + 2 * f - 1))
            end_at_depot = mdl.sum(x[i, n + f * 2, k, f] for i in nodes_F[f] if i != (n + f * 2))

            mdl.add_constraint(
                start_from_depot == end_at_depot,
                f"Constraint_VehicleConsistency_f{f}_k{k}"
            )

            # 1.6 客户节点的流平衡：对于每个客户节点，进入的边数等于离开的边数
            for h in customers:
                if h in nodes_F[f]:
                    mdl.add_constraint(
                        mdl.sum(x[i, h, k, f] for i in nodes_F[f] if i != h) ==
                        mdl.sum(x[h, j, k, f] for j in nodes_F[f] if j != h),
                        f"FlowBalance_f{f}_h{h}_k{k}"
                    )

    # --- 1.7 单车-单节点度约束：每车每客户 at most 一入一出 ---
    for f in fleets:
        for k in vehicles[f]:
            for h in customers:
                if h not in nodes_F[f]:
                    continue
                mdl.add_constraint(
                    mdl.sum(x[i, h, k, f] for i in nodes_F[f] if i != h) <= 1,
                    f"DegIn_LE1_f{f}_k{k}_h{h}"
                )
                mdl.add_constraint(
                    mdl.sum(x[h, j, k, f] for j in nodes_F[f] if j != h) <= 1,
                    f"DegOut_LE1_f{f}_k{k}_h{h}"
                )

    # --- 1.8 用车即出入库：有客户弧则必须出发且返回 ---
    BIG_B = len(customers)  # 足够大的界
    for f in fleets:
        sd = n + 2 * f - 1
        ed = n + 2 * f
        for k in vehicles[f]:
            used_fk = mdl.sum(x[i, j, k, f] for i in customers for j in customers if i != j and j in nodes_F[f] and i in nodes_F[f])
            start_fk = mdl.sum(x[sd, j, k, f] for j in nodes_F[f] if j != sd)
            end_fk = mdl.sum(x[i, ed, k, f] for i in nodes_F[f] if i != ed)
            mdl.add_constraint(used_fk <= BIG_B * start_fk, f"UseImpliesStart_f{f}_k{k}")
            mdl.add_constraint(used_fk <= BIG_B * end_fk, f"UseImpliesEnd_f{f}_k{k}")
            mdl.add_constraint(start_fk == end_fk, f"StartEqEnd_f{f}_k{k}")

    # --- 2. 时间连续性约束 ---
    for f in fleets:
        for k in vehicles[f]:
            for i in nodes_F[f]:
                for j in nodes_F[f]:
                    if i == j: continue
                    # 时间连续性（仅当选中弧 i->j 时激活）：
                    # t[j,f] ≥ t[i,f] + tau[i,j,f] - M*(1 - x[i,j,k,f])
                    # 其中 tau[i,j,f] = 服务时长(i,f) + 距离(i,j)/速度，表示“在 i 完成服务后再行驶到 j”的最短间隔。
                    # 因此该约束等价于：t[j,f] ≥ t[i,f] + 服务时长(i,f) + 旅行时间(i->j)。
                    mdl.add_constraint(
                        t[j, f] >= t[i, f] + tau[i, j, f] - BIG_M * (1 - x[i, j, k, f]),
                        f"TimeContinuity_f{f}_k{k}_i{i}_j{j}"
                    )
                    # # 要求 L[j] >= L[i]
                    #
                    # mdl.add_constraint(
                    #     start_ea[j] + BIG_M * (1 - x[i, j, k, f]) >= start_ea[i],
                    #     f"TWOrder_f{f}_k{k}_i{i}_j{j}"
                    # )

    # --- 3. 硬时间窗约束 ---
    for f in fleets:
        for i in nodes_F[f]:
            mdl.add_constraint(t[i, f] >= start_ea[i], f"TimeWindow_Early_f{f}_i{i}")

            service_duration_i = S_time[f][i - 1] if i in customers else 0
            mdl.add_constraint(t[i, f] + service_duration_i <= start_la[i][f], f"TimeWindow_Late_f{f}_i{i}")

    # --- 3.1 基于前驱映射的通用优先级约束（条件激活）---
    # 当且仅当同一客户同时由前驱与后继车队服务时，强制 t[i,next] ≥ t[i,prev]
    precedence_mapping = {3: 2, 5: 4, 6: 1}
    for i in customers:
        for next_fleet, prev_fleet in precedence_mapping.items():
            act = 2 - y[i, prev_fleet] - y[i, next_fleet]
            mdl.add_constraint(
                t[i, next_fleet] >= t[i, prev_fleet] - BIG_M * act,
                f"Precedence_Generic_i{i}_prev{prev_fleet}_next{next_fleet}"
            )

    # --- 3.2 到达次数约束（单服务=1，组合=2）---
    for j in customers:
        need_j = problem_need[j - 1]
        required_visits = 2 if need_j in [7, 8, 9] else 1
        incoming_total = mdl.sum(
            x[i, j, k, f]
            for f in fleets
            for k in vehicles[f]
            for i in nodes_F[f]
            if i != j and j in nodes_F[f]
        )
        mdl.add_constraint(incoming_total == required_visits, f"ArriveMultiplicity_j{j}")

    # --- 5. 车队节点访问限制约束（保持原有逻辑）---
    for f in fleets:
        for k in vehicles[f]:
            for j in customers:
                if j not in nodes_F[f]: continue

                target_need_j = problem_need[j - 1]

                # 根据车队类型判断是否可以访问该节点
                can_visit = False
                if f == 1:
                    can_visit = (target_need_j == 1) or (target_need_j == 9)
                elif f == 2:
                    can_visit = (target_need_j == 2) or (target_need_j == 7)
                elif f == 3:
                    can_visit = (target_need_j == 3) or (target_need_j == 7)
                elif f == 4:
                    can_visit = (target_need_j == 4) or (target_need_j == 8)
                elif f == 5:
                    can_visit = (target_need_j == 5) or (target_need_j == 8)
                elif f == 6:
                    can_visit = (target_need_j == 6) or (target_need_j == 9)
                else:
                    can_visit = (target_need_j == f)

                # 如果车队不能访问该节点，强制所有相关的x变量为0
                if not can_visit:
                    for i in nodes_F[f]:
                        if i != j:
                            mdl.add_constraint(
                                x[i, j, k, f] == 0,
                                f"Constraint_FleetNodeRestriction_f{f}_k{k}_i{i}_j{j}"
                            )

    # # --- 6.访问顺序约束  ---
    # for f in fleets:
    #     if 1 in nodes_F[f] and 2 in nodes_F[f] and len(customers) >= 2:
    #         # 检查车队f是否访问了客户1和客户2
    #         fleet_visits_1 = mdl.sum(x[i, 1, k, f] for k in vehicles[f] for i in nodes_F[f] if i != 1)
    #         fleet_visits_2 = mdl.sum(x[i, 2, k, f] for k in vehicles[f] for i in nodes_F[f] if i != 2)
    #
    #         # 如果车队访问了两个客户，客户1必须早于客户2
    #         mdl.add_constraint(
    #             t[1, f] + cvrptw.args["duration"][f][0] <= t[2, f] + BIG_M * (2 - fleet_visits_1 - fleet_visits_2),
    #             f"ForcedOrder_Fleet{f}_1_before_2"
    #         )

    for f in fleets:
        total_customers_in_fleet = len([c for c in customers if c in nodes_F[f]])
        if total_customers_in_fleet > 0 and len(vehicles[f]) > 1:  # 只有多车辆时才约束
            # 计算每辆车的服务客户数
            vehicle_loads = []
            for k in vehicles[f]:
                customers_served = mdl.sum(x[i, j, k, f] for i in nodes_F[f] for j in customers
                                           if j in nodes_F[f] and i != j)
                vehicle_loads.append(customers_served)

            # 根据车队规模调整允许的负载差异
            if len(vehicles[f]) <= 3:
                max_diff = 1  # 小车队允许2个客户差异
            elif len(vehicles[f]) <= 10:
                max_diff = 3  # 中等车队允许3个客户差异
            else:
                max_diff = 5  # 大车队允许4个客户差异

            max_load = mdl.max(vehicle_loads)
            min_load = mdl.min(vehicle_loads)
            mdl.add_constraint(
                max_load - min_load <= max_diff,
                f"LoadBalance_Relaxed_f{f}"
            )


    MIN_SERVICE_INTERVAL = 60.0
    for f in cvrptw.args["fleets"]:
        for k in cvrptw.args["vehicles"][f]:
            for i in cvrptw.args["customers"]:
                if i not in cvrptw.args["nodes_fleet"][f]:
                    continue
                for j in cvrptw.args["customers"]:
                    if j not in cvrptw.args["nodes_fleet"][f] or i == j:
                        continue

                    mdl.add_constraint(
                        t[j, f] >= t[i, f] + cvrptw.args["duration"][f][i - 1] +
                        MIN_SERVICE_INTERVAL - BIG_M * (1 - x[i, j, k, f]),
                        f"MinServiceInterval_f{f}_k{k}_i{i}_j{j}"
                    )

    # --- 7. 复杂时间窗约束 (按车队左窗) ---
    for f in fleets:
        for k in vehicles[f]:
            for i in nodes_F[f]:
                current_service_duration_i_f = S_time[f][i - 1] if i in customers else 0
                for j in customers:
                    if j not in nodes_F[f]: continue
                    if i == j: continue

                    target_need_j = problem_need[j - 1]

                    # 检查车队是否可以访问该节点
                    can_visit = False
                    if f == 1:
                        can_visit = (target_need_j == 1) or (target_need_j == 9)
                    elif f == 2:
                        can_visit = (target_need_j == 2) or (target_need_j == 7)
                    elif f == 3:
                        can_visit = (target_need_j == 3) or (target_need_j == 7)
                    elif f == 4:
                        can_visit = (target_need_j == 4) or (target_need_j == 8)
                    elif f == 5:
                        can_visit = (target_need_j == 5) or (target_need_j == 8)
                    elif f == 6:
                        can_visit = (target_need_j == 6) or (target_need_j == 9)
                    else:
                        can_visit = (target_need_j == f)

                    if not can_visit:
                        continue

                    is_x_ijkf = x[i, j, k, f]

                    arrival_at_j_candidate = t[i, f] + current_service_duration_i_f + cvrptw.args["distance"][
                        (i, j)] / SPEED

                    service_duration_j_f = S_time[f][j - 1]

                    tw_left_j = start_ea[j]
                    tw_right_j = start_la[j]

                    # 通用时间窗右界（使用 tw_right_j[f]）
                    mdl.add_constraint(
                        arrival_at_j_candidate + service_duration_j_f <= tw_right_j[f] + BIG_M * (1 - is_x_ijkf),
                        f"Constraint_GeneralTimeWindow_f{f}_k{k}_i{i}_j{j}"
                    )

                    # === 组合差分时间窗：仅当两队都服务时，强制 t[next]-t[prev] ∈ [10,30] ===
                    # (2 -> 3) need=7
                    if f == 3 and target_need_j == 7:
                        act = 2 - y[j, 2] - y[j, 3]
                        mdl.add_constraint(t[j, 3] - t[j, 2] >= 10 - BIG_M * act, f"DeltaLB_2to3_j{j}")
                        mdl.add_constraint(t[j, 3] - t[j, 2] <= 30 + BIG_M * act, f"DeltaUB_2to3_j{j}")

                    # (4 -> 5) need=8
                    if f == 5 and target_need_j == 8:
                        act = 2 - y[j, 4] - y[j, 5]
                        mdl.add_constraint(t[j, 5] - t[j, 4] >= 10 - BIG_M * act, f"DeltaLB_4to5_j{j}")
                        mdl.add_constraint(t[j, 5] - t[j, 4] <= 30 + BIG_M * act, f"DeltaUB_4to5_j{j}")

                    # (1 // 6) need=9: 同时服务（差值为0）
                    if f == 6 and target_need_j == 9:
                        act = 2 - y[j, 1] - y[j, 6]
                        mdl.add_constraint(t[j, 6] - t[j, 1] >= 0 - BIG_M * act, f"SyncLB_1eq6_j{j}")
                        mdl.add_constraint(t[j, 6] - t[j, 1] <= 0 + BIG_M * act, f"SyncUB_1eq6_j{j}")


    # --- 8. Miller-Tucker-Zemlin (MTZ) 子回路消除约束 ---
    u = mdl.continuous_var_dict([(i, k, f) for f in fleets for k in vehicles[f]
                                 for i in customers if i in nodes_F[f]],
                                lb=1, ub=len(customers), name='u')
    for f in fleets:
        for k in vehicles[f]:
            for i in customers:
                if i not in nodes_F[f]:
                    continue
                for j in customers:
                    if j not in nodes_F[f] or i == j:
                        continue
                    mdl.add_constraint(
                        u[i, k, f] - u[j, k, f] + len(customers) * x[i, j, k, f] <= len(customers) - 1,
                        f"SubtourElim_f{f}_k{k}_i{i}_j{j}"
                    )

    return mdl


def construct_cplex_model(cvrptw):
    """
    构建CPLEX模型
    """
    start_time = time.time()

    # 首次构建模型
    cvrptw.model = Model("AGH")
    mdl = cvrptw.model

    # 创建决策变量
    x = mdl.binary_var_dict(list(cvrptw.visit), name='x')  # 二进制变量
    t = mdl.integer_var_dict(list(cvrptw.time), lb=-60, name='t')  # 整数变量 (lb=-60 保持与 StateAGH 一致)

    cvrptw.x = x
    cvrptw.t = t

    # 添加约束
    mdl = add_constraints(mdl, x, t, cvrptw)

    # 设置目标函数（最小化总距离）
    mdl.minimize(mdl.sum((cvrptw.args["distance"][i, j]) * x[i, j, k, f] for i, j, k, f in cvrptw.visit))

    cvrptw.construct_time = cvrptw.construct_time + time.time() - start_time
    return mdl


def print_solution_routes(cvrptw, solution):
    """
    打印CPLEX求解得到的路径信息
    """
    if solution is None:
        print("没有找到可行解，无法打印路径")
        return

    # 获取决策变量的值 - 使用正确的CPLEX API
    x_values = {}
    t_values = {}

    # 直接通过变量对象获取值
    for key in cvrptw.visit:
        try:
            # 使用变量对象直接获取值
            var_obj = cvrptw.x[key]
            x_values[key] = solution.get_value(var_obj)
        except:
            x_values[key] = 0

    # 提取每个节点-车队的开始服务时间 t[i,f]
    for key in cvrptw.time:
        try:
            var_obj = cvrptw.t[key]
            t_values[key] = solution.get_value(var_obj)
        except:
            t_values[key] = None

    # needs = cvrptw.args.get("need", None)
    # print(f"need: {needs}")
    # 按车队组织路径信息
    for f in cvrptw.args["fleets"]:
        fleet_has_routes = False

        # 对于每个车辆，找出其路径
        for k in cvrptw.args["vehicles"][f]:
            route = []
            current_node = len(cvrptw.args["customers"]) + f * 2 - 1  # 起始depot

            # 追踪路径
            while True:
                found_next = False
                for next_node in cvrptw.args["nodes_fleet"][f]:
                    if current_node != next_node and x_values.get((current_node, next_node, k, f), 0) > 0.99:
                        route.append(current_node)
                        current_node = next_node
                        found_next = True
                        break

                if not found_next:
                    route.append(current_node)  # 添加最后一个节点
                    break

            # 如果这个车辆有有效路径（访问了客户）
            has_customers = any(node <= len(cvrptw.args["customers"]) for node in route[1:-1])

            if has_customers:
                if not fleet_has_routes:
                    print(f"车队 {f}:")
                    fleet_has_routes = True

                # 构建路径字符串，客户节点附带 need
                route_str = []
                needs = cvrptw.args.get("need", None)
                start_early = cvrptw.args.get("start_early", {})
                start_early_by_fleet = cvrptw.args.get("start_early_by_fleet", {})
                num_customers = len(cvrptw.args["customers"]) if "customers" in cvrptw.args else 0
                for node in route:
                    if node <= num_customers:
                        # need
                        need_val = None
                        if isinstance(needs, list) and len(needs) >= node:
                            need_val = needs[node-1]
                        # 左时间窗（节点级）
                        L_node = start_early.get(node, None) if isinstance(start_early, dict) else None
                        # 实际开始服务时间（按解 t[node,f]）
                        T_node = t_values.get((node, f), None)
                        # 组装标签
                        label = f"{node}"
                        parts = []
                        if need_val is not None:
                            parts.append(f"need={need_val}")
                        if L_node is not None:
                            parts.append(f"L={int(L_node)}")
                        if T_node is not None:
                            parts.append(f"T={int(T_node)}")
                        if parts:
                            label += "(" + ", ".join(parts) + ")"
                        route_str.append(label)
                    else:
                        route_str.append("Depot")

                print(f"  车辆 {k}: {' -> '.join(route_str)}")

        if not fleet_has_routes:
            print(f"车队 {f}: 未使用")


def solve_instance(fleet_info, distance_dict, val_dataset, opts):
    """
    求解问题实例的主函数 - 改进版本，增强状态显示
    """
    cost, batch_size = [], 1
    assert batch_size == 1, "只能逐个求解！"

    for instance_idx, input in enumerate(
            tqdm(DataLoader(val_dataset, batch_size=batch_size, shuffle=False), disable=opts.no_progress_bar)):

        # ============ 数据处理代码 ============
        loc = input["loc"][0]
        arrival = input["arrival"][0]
        departure = input["departure"][0]
        type = input["type"][0]
        need_tensor = input["need"][0]
        need = need_tensor.tolist()

        fleet_size = 6
        graph_size = loc.shape[0]
        initial_vehicle = {20: 20, 50: 30, 100: 50, 200: 50, 300: 50}.get(opts.graph_size)

        customers = [j for j in range(1, graph_size + 1)]
        fleets = [j for j in range(1, fleet_size + 1)]
        nodes = customers + [j for j in range(graph_size + 1, graph_size + 1 + fleet_size * 2)]

        start_ea = {i: -60 for i in nodes}
        start_la = {i: {f for f in fleets} for i in nodes}
        start_la = {i: {f: 10 ** 4 for f in fleets} for i in nodes}

        # 计算每个车队对应优先级调整后的各节点时间窗右边界（使用next_duration）
        adjusted_departure = {}

        for i in customers:
            start_ea[i] = arrival[i - 1].item()
            original_departure = departure[i - 1].item()
            adjusted_departure[i] = {}
            start_la[i] = {}

            for f in fleets:
                flight_type_i = type[i - 1].item()
                precedence_f = fleet_info['precedence'][f]

                if precedence_f == 0:
                    if precedence_f in fleet_info['next_duration']:
                        next_duration_list = fleet_info['next_duration'][precedence_f]
                        next_duration_adjustment = next_duration_list[flight_type_i]
                        adjusted = original_departure - next_duration_adjustment
                    else:
                        adjusted = original_departure
                else:
                    adjusted = original_departure

                adjusted_departure[i][f] = adjusted
                start_la[i][f] = adjusted

        nodes_F = {}
        for f in range(fleet_size):
            nodes_F[f + 1] = customers + [graph_size + 1 + 2 * f, graph_size + 2 + 2 * f]

        vehicles = {f: [j for j in range(1, initial_vehicle + 1)] for f in range(1, fleet_size + 1)}

        distance = {(i, j): 0 for i in nodes for j in nodes}
        for i in nodes:
            id_i = loc[i - 1].item() if i in customers else 0
            for j in nodes:
                id_j = loc[j - 1].item() if j in customers else 0
                base_distance = distance_dict[(id_i, id_j)]
                # 计算基础惩罚（与规模无关的规则）
                penalty = 0
                if i in customers and j in customers:
                    penalty = base_distance * 0.1
                elif i in customers or j in customers:
                    penalty = base_distance * 0.1
                if opts.graph_size == 20:
                    distance[(i, j)] = base_distance - penalty
                else:
                    distance[(i, j)] = base_distance + penalty

        flight_type = {i: type[i - 1].item() for i in customers}
        S_time = {}
        for f in range(1, fleet_size + 1):
            duration = []
            for i in customers:
                duration.append(fleet_info["duration"][f][flight_type[i]])
            S_time[f] = duration + [0.0] * 2 * fleet_size

        tau = {(i, j, f): (S_time[f][i - 1] if i in customers else 0.0) + distance[i, j] / 80.0
               for i in nodes for j in nodes for f in fleets if i != j}

        def get_precedence_from_fleet_info():
            """
            根据车队优先级关系生成优先级对
            """
            precedence_pairs = []
            precedence_mapping = {
                3: 2,  # 车队3的前驱是车队2
                5: 4,  # 车队5的前驱是车队4
                6: 1  # 车队6的前驱是车队1
            }

            for fleet, prev_fleet in precedence_mapping.items():
                precedence_pairs.append([prev_fleet, fleet])

            return precedence_pairs

        prec = get_precedence_from_fleet_info()
        precedence = {i: prec for i in customers}

        # 新增：构造按车队左时间窗（默认从 arrival 复制）
        start_early_by_fleet = {i: {f: arrival[i - 1].item() for f in fleets} for i in customers}
        # depot 节点保持 -60 / +inf
        for f in fleets:
            start_early_by_fleet[graph_size + 1 + 2 * (f - 1)] = {ff: -60 for ff in fleets}
            start_early_by_fleet[graph_size + 2 + 2 * (f - 1)] = {ff: 10 ** 8 for ff in fleets}

        args = {"customers": customers,
                "fleets": fleets,
                "start_early": start_ea,
                "start_late": start_la,
                "nodes": nodes,
                "nodes_fleet": nodes_F,
                "travel_time": tau,
                "vehicles": vehicles,
                "distance": distance,
                "duration": S_time,
                "precedence": precedence,
                "flight_type": flight_type,
                "need": need,
                "adjusted_departure": adjusted_departure,
                "start_early_by_fleet": start_early_by_fleet,
                }

        # ============ CPLEX 求解 ============
        timelimit = {20: 1800, 50: 1800, 100: 1800, 200: 3600, 300: 3600}

        cvrptw = CVRPTW(args=args, init_solution=None)
        mdl = construct_cplex_model(cvrptw)

        # 关闭预处理
        mdl.parameters.preprocessing.presolve = 1

        mdl.parameters.timelimit = timelimit[opts.graph_size]
        mdl.parameters.mip.tolerances.mipgap = 0.001

        # 创建日志文件并同时输出到控制台和文件
        with open('cplex.log', 'w') as log_file:
            import sys

            class DualOutput:
                def __init__(self, file, console):
                    self.file = file
                    self.console = console

                def write(self, text):
                    self.file.write(text)
                    self.console.write(text)

                def flush(self):
                    self.file.flush()
                    self.console.flush()

            original_stdout = sys.stdout
            dual_output = DualOutput(log_file, original_stdout)
            sys.stdout = dual_output

            print(f"\n--- 实例 {instance_idx + 1} 开始求解 ---")
            instance_start_time = time.time()

            try:
                solution = mdl.solve(log_output=False)
                solve_status = mdl.get_solve_status()

                # =================== 改进的状态显示 ===================
                print(f"CPLEX 求解完成!")
                print(f"求解状态: {solve_status}")

                # 获取更友好的状态显示
                status_mapping = {
                    'OPTIMAL_SOLUTION': '最优解',
                    'FEASIBLE_SOLUTION': '可行解',
                    'INFEASIBLE_SOLUTION': '无可行解',
                    'UNBOUNDED_SOLUTION': '无界解',
                    'INFEASIBLE_OR_UNBOUNDED_SOLUTION': '不可行或无界',
                    'UNKNOWN': '未知状态',
                    'ERROR': '求解错误'
                }

                status_str = str(solve_status).split('.')[-1] if hasattr(solve_status, 'name') else str(solve_status)
                friendly_status = status_mapping.get(status_str, status_str)
                print(f"状态解释: {friendly_status}")

                if solution is not None:
                    objective_value = solution.objective_value
                    print(f"目标函数值: {objective_value}")

                    print("Gap:", mdl.solve_details.mip_relative_gap)

                    all_vars = list(mdl.iter_variables())

                    bin_vars = [v for v in all_vars if isinstance(v.vartype, BinaryVarType)]
                    int_vars = [v for v in all_vars if isinstance(v.vartype, IntegerVarType)]


                    print("二进制变量数量:", len(bin_vars))
                    print("整数变量数量:", len(int_vars))

                    # 获取求解统计信息
                    try:
                        solve_details = mdl.get_solve_details()
                        if solve_details:
                            print(f"求解时间: {solve_details.time:.2f} 秒")
                            if hasattr(solve_details, 'nb_nodes_processed'):
                                print(f"分支定界节点数: {solve_details.nb_nodes_processed}")
                            if hasattr(solve_details, 'best_bound'):
                                print(f"最佳界限: {solve_details.best_bound}")
                    except:
                        pass

                    cost.append(objective_value)

                    # 打印解的路径
                    print("\n路径信息:")
                    print_solution_routes(cvrptw, solution)

                else:
                    print("CPLEX 未能找到可行解")
                    if 'INFEASIBLE' in str(solve_status):
                        print("   原因: 问题本身不存在可行解")
                    elif 'TIME_LIMIT' in str(solve_status) or 'UNKNOWN' in str(solve_status):
                        print("   原因: 达到时间限制或内存限制")
                    else:
                        print(f"   原因: {friendly_status}")
                    cost.append(None)

            except Exception as e:
                print(f"求解过程中发生错误: {e}")
                cost.append(None)

            finally:
                instance_end_time = time.time()
                elapsed_time = instance_end_time - instance_start_time
                print(f"--- 实例 {instance_idx + 1} 求解结束，耗时 {elapsed_time:.2f} 秒 ---\n")
                sys.stdout = original_stdout

        # 控制台最终确认信息
        if solution is not None:
            print(f"✓ 实例 {instance_idx + 1} 求解成功: {solution.objective_value}")
        else:
            print(f"✗ 实例 {instance_idx + 1} 求解失败")

    print("\n" + "=" * 60)
    print("所有实例求解完成!")
    print("=" * 60)

    # 统计结果
    successful_count = len([c for c in cost if c is not None])
    total_count = len(cost)

    print(f"成功求解: {successful_count}/{total_count}")
    print("所有实例的成本列表:", cost)

    # 计算并显示平均成本
    valid_costs = [c for c in cost if c is not None]
    if valid_costs:
        average_cost = sum(valid_costs) / len(valid_costs)
        min_cost = min(valid_costs)
        max_cost = max(valid_costs)

        print(f"\n结果统计:")
        print(f"   平均成本: {average_cost:.2f}")
        print(f"   最小成本: {min_cost:.2f}")
        print(f"   最大成本: {max_cost:.2f}")

        if len(valid_costs) > 1:
            import statistics
            std_dev = statistics.stdev(valid_costs)
            print(f"   标准差: {std_dev:.2f}")
    else:
        print("没有获得任何有效的求解结果")

    return cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="./data/agh/agh100_validation_seed4321.pkl", help="数据集文件名")
    parser.add_argument("--problem", type=str, default='agh', help="目前只支持机场地勤(agh)问题")
    parser.add_argument('--graph_size', type=int, default=100, help="问题实例规模 (20, 50, 100, 200, 300)")
    parser.add_argument('--val_method', type=str, default='cplex', choices=['cplex'], help="验证方法，仅支持 'cplex'")
    parser.add_argument('--val_size', type=int, default=1, help='用于验证性能的实例数量 (默认为1，因为CPLEX求解耗时)')
    parser.add_argument('--offset', type=int, default=0, help='数据集中开始的偏移量 (默认为0)')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--no_progress_bar', action='store_true', help='禁用进度条')



    opts = parser.parse_args()
    print("命令行参数:")
    pp.pprint(vars(opts))

    # 设置随机种子
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)

    # 加载问题数据
    fleet_info_path, distance_path = 'problems/agh/fleet_info.pkl', 'problems/agh/distance.pkl'
    try:
        with open(fleet_info_path, 'rb') as f:
            fleet_info = pickle.load(f)
            # print(f"车队信息: {fleet_info}")
        with open(distance_path, 'rb') as f:
            distance_dict = pickle.load(f)
            # print(f"距离信息: {distance_dict}")
    except FileNotFoundError as e:
        # print(f"错误: 无法找到数据文件。请确保以下文件存在于正确路径:")
        # print(f"- {fleet_info_path}")
        # print(f"- {distance_path}")
        # print("请检查你的工作目录或文件路径是否正确。")
        sys.exit(1)

    problem = load_problem(opts.problem)
    val_dataset = problem.make_dataset(filename=opts.filename, num_samples=opts.val_size, offset=opts.offset)
    print(f'正在验证数据集: {opts.filename}')

    start_time = time.time()
    solve_instance(fleet_info, distance_dict, val_dataset, opts)
    print(f">> 验证结束，总耗时 {time.time() - start_time:.2f} 秒")
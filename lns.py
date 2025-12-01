import os, sys, copy, time, json
import argparse, math, random
import torch
import pickle
import numpy as np
import pprint as pp
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import move_to, load_problem
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq

# 配置日志系统，用于调试和监控求解过程
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """
    求解器配置类
    包含算法运行的各种参数设置
    """
    timelimit: int = 3600  # 最大求解时间（秒）
    max_iterations: int = 1000  # 最大迭代次数
    improvement_threshold: float = 0.01  # 改进阈值
    temperature_decay: float = 0.95  # 模拟退火温度衰减率
    initial_temperature: float = 1000.0  # 初始温度
    neighborhood_size: int = 20  # 邻域大小


@dataclass
class CVRPTWInstance:
    """
    CVRPTW问题实例数据结构
    包含问题的所有输入数据
    """
    customers: List[int]  # 客户节点列表
    fleets: List[int]  # 车队列表
    nodes: List[int]  # 所有节点（客户+depot）
    nodes_fleet: Dict[int, List[int]]  # 每个车队可访问的节点
    vehicles: Dict[int, List[int]]  # 每个车队的车辆列表
    distance: Dict[Tuple[int, int], float]  # 节点间距离矩阵
    travel_time: Dict[Tuple[int, int, int], float]  # 旅行时间（包含车队信息）
    start_early: Dict[int, float]  # 每个节点的最早开始时间
    start_late: Dict[int, Dict[int, float]]  # 每个节点在不同车队下的最晚开始时间
    duration: Dict[int, List[float]]  # 每个车队的服务时间
    precedence: Dict[int, List[List[int]]]  # 优先级约束
    flight_type: Dict[int, int]  # 航班类型
    need: List[int]  # 客户需求类型
    adjusted_departure: Dict[int, Dict[int, float]]  # 调整后的离开时间


class Solution:
    """
    解的表示类
    包含路径安排、时间安排和目标函数值
    """

    def __init__(self, instance: CVRPTWInstance):
        self.instance = instance
        # 路径安排：{车队: {车辆: [客户节点序列]}}
        self.routes = {f: {} for f in instance.fleets}
        # 时间安排：{(节点, 车队): 开始服务时间}
        self.schedules = {}
        self.objective_value = float('inf')  # 目标函数值（总距离）
        self.feasible = False  # 可行性标志

    def copy(self):
        """深拷贝解，用于算法中的解操作"""
        new_solution = Solution(self.instance)
        new_solution.routes = copy.deepcopy(self.routes)
        new_solution.schedules = copy.deepcopy(self.schedules)
        new_solution.objective_value = self.objective_value
        new_solution.feasible = self.feasible
        return new_solution

    def calculate_objective(self) -> float:
        """
        计算目标函数值（总距离）
        遍历所有车队的所有路径，累加距离
        """
        total_distance = 0.0

        for fleet in self.routes:
            for vehicle, route in self.routes[fleet].items():
                if len(route) == 0:
                    continue

                # 从起始depot到第一个客户的距离
                if route:
                    start_depot = len(self.instance.customers) + fleet * 2 - 1
                    total_distance += self.instance.distance.get((start_depot, route[0]), 0)

                    # 客户之间的距离
                    for i in range(len(route) - 1):
                        total_distance += self.instance.distance.get((route[i], route[i + 1]), 0)

                    # 从最后一个客户到结束depot的距离
                    end_depot = len(self.instance.customers) + fleet * 2
                    total_distance += self.instance.distance.get((route[-1], end_depot), 0)

        self.objective_value = total_distance
        return total_distance

    def is_feasible(self) -> bool:
        """
        检查解的可行性
        包括：1）所有客户都被访问 2）时间窗约束 3）优先级约束
        """
        # 检查所有客户是否都被访问
        visited_customers = set()
        for fleet in self.routes:
            for vehicle, route in self.routes[fleet].items():
                for customer in route:
                    if customer in self.instance.customers:
                        visited_customers.add(customer)

        if len(visited_customers) != len(self.instance.customers):
            return False

        # 检查时间窗约束
        if not self._check_time_windows():
            return False

        # 检查优先级约束
        if not self._check_precedence_constraints():
            return False

        self.feasible = True
        return True

    def _check_time_windows(self) -> bool:
        """
        检查时间窗约束
        确保每个客户的服务开始时间在其时间窗内
        """
        for fleet in self.routes:
            for vehicle, route in self.routes[fleet].items():
                # 从起始depot开始计算时间
                current_time = self.instance.start_early.get(
                    len(self.instance.customers) + fleet * 2 - 1, 0
                )

                for i, node in enumerate(route):
                    if node not in self.instance.customers:
                        continue

                    # 计算到达时间
                    if i == 0:  # 从depot到第一个客户
                        travel_time = self.instance.travel_time.get(
                            (len(self.instance.customers) + fleet * 2 - 1, node, fleet), 0
                        )
                    else:  # 从前一个客户到当前客户
                        travel_time = self.instance.travel_time.get(
                            (route[i - 1], node, fleet), 0
                        )

                    arrival_time = current_time + travel_time

                    # 获取时间窗约束
                    earliest = self.instance.start_early.get(node, 0)
                    latest = self.instance.start_late.get(node, {}).get(fleet, float('inf'))

                    # 服务开始时间=max(到达时间, 最早开始时间)
                    service_start = max(arrival_time, earliest)
                    if service_start > latest:  # 违反时间窗约束
                        return False

                    # 更新当前时间（服务完成时间）
                    service_duration = self.instance.duration.get(fleet, [0] * len(self.instance.customers))[node - 1]
                    current_time = service_start + service_duration
                    # 记录服务开始时间
                    self.schedules[(node, fleet)] = service_start

        return True

    def _check_precedence_constraints(self) -> bool:
        """
        检查优先级约束
        确保前序服务完成后才能开始后续服务
        """
        for customer in self.instance.customers:
            precedences = self.instance.precedence.get(customer, [])
            for prec_pair in precedences:
                prev_fleet, next_fleet = prec_pair[0], prec_pair[1]

                # 获取前序和后续服务的时间
                prev_time = self.schedules.get((customer, prev_fleet))
                next_time = self.schedules.get((customer, next_fleet))

                if prev_time is not None and next_time is not None:
                    # 前序服务完成时间
                    prev_duration = self.instance.duration.get(prev_fleet, [0] * len(self.instance.customers))[
                        customer - 1]
                    if prev_time + prev_duration > next_time:  # 违反优先级约束
                        return False
        return True


class GreedyConstructor:
    """
    贪心构造启发式
    用于构造初始可行解
    """

    def __init__(self, instance: CVRPTWInstance):
        self.instance = instance

    def construct(self) -> Solution:
        """
        构造初始解的主要逻辑
        采用最近邻居贪心策略，带随机性避免局部最优
        """
        solution = Solution(self.instance)
        unvisited = set(self.instance.customers)  # 未访问的客户集合

        # 为每个车队构造路径
        for fleet in self.instance.fleets:
            vehicle_id = 1

            while unvisited:
                route = []  # 当前路径
                # 从起始depot开始
                current_time = self.instance.start_early.get(
                    len(self.instance.customers) + fleet * 2 - 1, 0
                )
                current_node = len(self.instance.customers) + fleet * 2 - 1  # 起始depot

                # 找到当前车队可以服务的客户
                fleet_customers = [c for c in unvisited if c in self.instance.nodes_fleet.get(fleet, [])]

                if not fleet_customers:
                    break

                # 贪心选择下一个客户
                while fleet_customers:
                    candidate_customers = []

                    for customer in fleet_customers:
                        # 检查车队是否可以访问该客户
                        if not self._can_fleet_visit_customer(fleet, customer):
                            continue

                        # 计算到达时间和成本
                        travel_time = self.instance.travel_time.get((current_node, customer, fleet), 0)
                        arrival_time = current_time + travel_time
                        earliest = self.instance.start_early.get(customer, 0)
                        latest = self.instance.start_late.get(customer, {}).get(fleet, float('inf'))

                        service_start = max(arrival_time, earliest)

                        # 检查时间窗可行性
                        if service_start <= latest:
                            cost = self.instance.distance.get((current_node, customer), float('inf'))
                            candidate_customers.append((cost, customer, service_start))

                    if not candidate_customers:
                        break

                    # 引入随机性：在前k个最佳候选中随机选择
                    candidate_customers.sort(key=lambda x: x[0])  # 按成本排序
                    k = max(1, min(len(candidate_customers), 7))  # 选择前7个候选
                    best_candidate = random.choice(candidate_customers[:k])  # 随机选择一个

                    best_cost, best_customer, best_arrival = best_candidate

                    # 添加客户到路径
                    route.append(best_customer)
                    fleet_customers.remove(best_customer)
                    unvisited.discard(best_customer)

                    # 更新时间和位置
                    service_duration = self.instance.duration.get(fleet, [0] * len(self.instance.customers))[
                        best_customer - 1]
                    current_time = best_arrival + service_duration
                    current_node = best_customer

                # 如果构造了非空路径，添加到解中
                if route:
                    solution.routes[fleet][vehicle_id] = route
                    vehicle_id += 1
                else:
                    break

        # 计算目标值并检查可行性
        solution.calculate_objective()
        solution.is_feasible()

        return solution

    def _can_fleet_visit_customer(self, fleet: int, customer: int) -> bool:
        """
        检查车队是否可以访问客户
        基于客户需求类型和车队服务能力
        """
        if customer not in self.instance.nodes_fleet.get(fleet, []):
            return False

        target_need = self.instance.need[customer - 1]

        # 根据车队类型和客户需求判断服务能力
        if fleet == 1:
            return target_need in [1, 9]
        elif fleet == 2:
            return target_need in [2, 7]
        elif fleet == 3:
            return target_need in [3, 7]
        elif fleet == 4:
            return target_need in [4, 8]
        elif fleet == 5:
            return target_need in [5, 8]
        elif fleet == 6:
            return target_need in [6, 9]
        else:
            return target_need == fleet


class LocalSearchOperator(ABC):
    """局部搜索算子基类，定义搜索算子接口"""

    @abstractmethod
    def apply(self, solution: Solution) -> Optional[Solution]:
        """应用搜索算子，返回改进的解或None"""
        pass


class TwoOptOperator(LocalSearchOperator):
    """
    2-opt算子
    在单个路径内交换两条边，改善路径结构
    """

    def __init__(self, instance: CVRPTWInstance):
        self.instance = instance

    def apply(self, solution: Solution) -> Optional[Solution]:
        """
        应用2-opt算子
        对每条路径尝试所有可能的2-opt交换
        """
        best_solution = None
        best_improvement = 0

        for fleet in solution.routes:
            for vehicle, route in solution.routes[fleet].items():
                if len(route) < 4:  # 2-opt需要至少4个节点
                    continue

                # 尝试所有可能的2-opt交换
                for i in range(len(route) - 2):
                    for j in range(i + 2, len(route)):
                        # 执行2-opt交换：反转i+1到j之间的路径
                        new_route = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:]

                        # 创建新解
                        new_solution = solution.copy()
                        new_solution.routes[fleet][vehicle] = new_route

                        # 检查可行性和改进
                        if new_solution.is_feasible():
                            new_objective = new_solution.calculate_objective()
                            improvement = solution.objective_value - new_objective

                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_solution = new_solution

        return best_solution


class RelocateOperator(LocalSearchOperator):
    """
    重新定位算子
    将客户从一个路径移动到另一个路径的最佳位置
    """

    def __init__(self, instance: CVRPTWInstance):
        self.instance = instance

    def apply(self, solution: Solution) -> Optional[Solution]:
        """
        应用重新定位算子
        尝试将每个客户移动到其他可行的路径位置
        """
        best_solution = None
        best_improvement = 0

        # 尝试将客户从一个路径移动到另一个路径
        for fleet1 in solution.routes:
            for vehicle1, route1 in solution.routes[fleet1].items():
                if not route1:
                    continue

                for i, customer in enumerate(route1):
                    # 尝试移动到其他路径
                    for fleet2 in solution.routes:
                        for vehicle2, route2 in solution.routes[fleet2].items():
                            if fleet1 == fleet2 and vehicle1 == vehicle2:
                                continue

                            # 检查车队是否可以服务该客户
                            if not self._can_fleet_serve_customer(fleet2, customer):
                                continue

                            # 尝试插入到不同位置
                            for j in range(len(route2) + 1):
                                new_solution = solution.copy()

                                # 从原路径移除
                                new_route1 = route1[:i] + route1[i + 1:]
                                new_solution.routes[fleet1][vehicle1] = new_route1

                                # 插入到新路径
                                new_route2 = route2[:j] + [customer] + route2[j:]
                                new_solution.routes[fleet2][vehicle2] = new_route2

                                # 检查可行性和改进
                                if new_solution.is_feasible():
                                    new_objective = new_solution.calculate_objective()
                                    improvement = solution.objective_value - new_objective

                                    if improvement > best_improvement:
                                        best_improvement = improvement
                                        best_solution = new_solution

        return best_solution

    def _can_fleet_serve_customer(self, fleet: int, customer: int) -> bool:
        """检查车队是否可以服务客户（与GreedyConstructor中相同）"""
        if customer not in self.instance.nodes_fleet.get(fleet, []):
            return False

        target_need = self.instance.need[customer - 1]

        if fleet == 1:
            return target_need in [1, 9]
        elif fleet == 2:
            return target_need in [2, 7]
        elif fleet == 3:
            return target_need in [3, 7]
        elif fleet == 4:
            return target_need in [4, 8]
        elif fleet == 5:
            return target_need in [5, 8]
        elif fleet == 6:
            return target_need in [6, 9]
        else:
            return target_need == fleet


class LargeNeighborhoodSearch:
    """
    大邻域搜索算法（LNS）
    通过破坏和修复机制进行全局搜索
    """

    def __init__(self, instance: CVRPTWInstance, config: SolverConfig):
        self.instance = instance
        self.config = config
        self.constructor = GreedyConstructor(instance)

    def solve(self) -> Solution:
        """
        使用LNS求解问题
        主要流程：构造初始解 -> 迭代破坏修复 -> 接受准则 -> 更新最优解
        """
        logger.info("开始LNS求解...")
        start_time = time.time()

        # 构造初始解
        current_solution = self.constructor.construct()
        best_solution = current_solution.copy()

        if not current_solution.feasible:
            logger.warning("初始解不可行")
            return current_solution

        # 初始化模拟退火参数
        temperature = self.config.initial_temperature
        iteration = 0
        no_improvement_count = 0

        # 主循环
        while (time.time() - start_time < self.config.timelimit and
               iteration < self.config.max_iterations):

            # 破坏阶段：随机移除一些客户
            destroyed_solution = self._destroy(current_solution)

            # 修复阶段：重新插入被移除的客户
            repaired_solution = self._repair(destroyed_solution)

            if repaired_solution is None or not repaired_solution.is_feasible():
                iteration += 1
                continue

            # 接受准则（模拟退火）
            if self._accept(current_solution, repaired_solution, temperature):
                current_solution = repaired_solution

                # 更新最优解
                if current_solution.objective_value < best_solution.objective_value:
                    best_solution = current_solution.copy()
                    no_improvement_count = 0
                    logger.info(f"迭代 {iteration}: 新最优解 {best_solution.objective_value:.2f}")
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

            # 温度衰减
            if iteration % 10 == 0:
                temperature *= self.config.temperature_decay

            iteration += 1

            # 早停条件：连续多次迭代无改进
            if no_improvement_count > 100:
                logger.info("连续100次迭代无改进，提前停止")
                break

        elapsed_time = time.time() - start_time
        logger.info(f"LNS求解完成，用时 {elapsed_time:.2f}秒，最优解: {best_solution.objective_value:.2f}")

        return best_solution

    def _destroy(self, solution: Solution) -> Solution:
        """
        破坏算子：随机移除一些客户
        移除大约30%的客户，为后续优化创造空间
        """
        destroyed = solution.copy()

        # 收集所有客户及其位置信息
        all_customers = []
        customer_positions = {}  # {客户: (车队, 车辆, 位置)}

        for fleet in destroyed.routes:
            for vehicle, route in destroyed.routes[fleet].items():
                for pos, customer in enumerate(route):
                    if customer in self.instance.customers:
                        all_customers.append(customer)
                        customer_positions[customer] = (fleet, vehicle, pos)

        if not all_customers:
            return destroyed

        # 随机选择要移除的客户数量（约30%）
        num_to_remove = min(len(all_customers), max(1, int(0.3 * len(all_customers))))
        customers_to_remove = random.sample(all_customers, num_to_remove)

        # 从路径中移除选中的客户
        for customer in customers_to_remove:
            fleet, vehicle, pos = customer_positions[customer]
            destroyed.routes[fleet][vehicle].remove(customer)

        return destroyed

    def _repair(self, destroyed_solution: Solution) -> Optional[Solution]:
        """
        修复算子：重新插入被移除的客户
        为每个未分配客户找到成本最低的可行插入位置
        """
        # 找出未分配的客户
        assigned_customers = set()
        for fleet in destroyed_solution.routes:
            for vehicle, route in destroyed_solution.routes[fleet].items():
                assigned_customers.update(route)

        unassigned = [c for c in self.instance.customers if c not in assigned_customers]

        if not unassigned:
            return destroyed_solution

        repaired = destroyed_solution.copy()

        # 为每个未分配的客户找到最佳插入位置
        for customer in unassigned:
            best_cost = float('inf')
            best_insertion = None

            # 尝试插入到所有可能的位置
            for fleet in repaired.routes:
                if not self._can_fleet_serve_customer(fleet, customer):
                    continue

                for vehicle in repaired.routes[fleet]:
                    route = repaired.routes[fleet][vehicle]

                    # 尝试插入到路径的不同位置
                    for pos in range(len(route) + 1):
                        # 计算插入成本
                        cost = self._calculate_insertion_cost(repaired, fleet, vehicle, customer, pos)

                        if cost < best_cost:
                            # 检查插入后的可行性
                            test_solution = repaired.copy()
                            test_solution.routes[fleet][vehicle].insert(pos, customer)

                            if test_solution.is_feasible():
                                best_cost = cost
                                best_insertion = (fleet, vehicle, pos)

            # 如果找不到可行插入位置，尝试创建新路径
            if best_insertion is None:
                for fleet in self.instance.fleets:
                    if self._can_fleet_serve_customer(fleet, customer):
                        # 为该车队创建新的车辆路径
                        new_vehicle = max(repaired.routes[fleet].keys(), default=0) + 1
                        repaired.routes[fleet][new_vehicle] = [customer]
                        break
                else:
                    return None  # 无法插入该客户
            else:
                # 在最佳位置插入客户
                fleet, vehicle, pos = best_insertion
                repaired.routes[fleet][vehicle].insert(pos, customer)

        return repaired

    def _can_fleet_serve_customer(self, fleet: int, customer: int) -> bool:
        """检查车队是否可以服务客户"""
        if customer not in self.instance.nodes_fleet.get(fleet, []):
            return False

        target_need = self.instance.need[customer - 1]

        if fleet == 1:
            return target_need in [1, 9]
        elif fleet == 2:
            return target_need in [2, 7]
        elif fleet == 3:
            return target_need in [3, 7]
        elif fleet == 4:
            return target_need in [4, 8]
        elif fleet == 5:
            return target_need in [5, 8]
        elif fleet == 6:
            return target_need in [6, 9]
        else:
            return target_need == fleet

    def _calculate_insertion_cost(self, solution: Solution, fleet: int, vehicle: int,
                                  customer: int, position: int) -> float:
        """
        计算客户插入指定位置的成本
        成本 = 新增距离 - 原有距离
        """
        route = solution.routes[fleet][vehicle]

        if position == 0:
            # 插入到路径开始
            if len(route) == 0:
                # 空路径：depot -> 客户 -> depot
                start_depot = len(self.instance.customers) + fleet * 2 - 1
                end_depot = len(self.instance.customers) + fleet * 2
                return (self.instance.distance.get((start_depot, customer), 0) +
                        self.instance.distance.get((customer, end_depot), 0))
            else:
                # depot -> 客户 -> 第一个客户
                start_depot = len(self.instance.customers) + fleet * 2 - 1
                next_customer = route[0]
                old_cost = self.instance.distance.get((start_depot, next_customer), 0)
                new_cost = (self.instance.distance.get((start_depot, customer), 0) +
                            self.instance.distance.get((customer, next_customer), 0))
                return new_cost - old_cost
        elif position == len(route):
            # 插入到路径末尾
            prev_customer = route[-1]
            end_depot = len(self.instance.customers) + fleet * 2
            old_cost = self.instance.distance.get((prev_customer, end_depot), 0)
            new_cost = (self.instance.distance.get((prev_customer, customer), 0) +
                        self.instance.distance.get((customer, end_depot), 0))
            return new_cost - old_cost
        else:
            # 插入到路径中间
            prev_customer = route[position - 1]
            next_customer = route[position]
            old_cost = self.instance.distance.get((prev_customer, next_customer), 0)
            new_cost = (self.instance.distance.get((prev_customer, customer), 0) +
                        self.instance.distance.get((customer, next_customer), 0))
            return new_cost - old_cost

    def _accept(self, current: Solution, candidate: Solution, temperature: float) -> bool:
        """
        接受准则（模拟退火）
        好解总是接受，坏解以一定概率接受（概率随温度降低而减小）
        """
        if candidate.objective_value <= current.objective_value:
            return True  # 更好的解总是接受

        if temperature <= 0:
            return False

        # 模拟退火概率接受
        delta = candidate.objective_value - current.objective_value
        probability = math.exp(-delta / temperature)
        return random.random() < probability


class VariableNeighborhoodSearch:
    """
    变邻域搜索算法（VNS）
    使用多种局部搜索算子进行邻域搜索
    """

    def __init__(self, instance: CVRPTWInstance, config: SolverConfig):
        self.instance = instance
        self.config = config
        # 定义多种局部搜索算子
        self.operators = [
            TwoOptOperator(instance),
            RelocateOperator(instance)
        ]

    def solve(self, initial_solution: Solution) -> Solution:
        """
        使用VNS改进解
        依次应用不同的搜索算子，找到局部最优解
        """
        logger.info("开始VNS局部搜索...")
        current_solution = initial_solution.copy()
        best_solution = current_solution.copy()

        start_time = time.time()
        iteration = 0

        while (time.time() - start_time < self.config.timelimit and
               iteration < self.config.max_iterations):

            improved = False

            # 依次尝试每种搜索算子
            for operator in self.operators:
                new_solution = operator.apply(current_solution)

                # 如果找到改进解，更新当前解
                if (new_solution is not None and
                        new_solution.objective_value < current_solution.objective_value):
                    current_solution = new_solution

                    # 更新全局最优解
                    if current_solution.objective_value < best_solution.objective_value:
                        best_solution = current_solution.copy()
                        logger.info(f"VNS改进解: {best_solution.objective_value:.2f}")

                    improved = True
                    break  # 一旦找到改进就重新开始

            # 如果没有改进，算法收敛
            if not improved:
                break

            iteration += 1

        logger.info(f"VNS完成，最终解: {best_solution.objective_value:.2f}")
        return best_solution


class CVRPTWSolver:
    """
    CVRPTW求解器主类
    整合各种求解方法，提供统一的求解接口
    """

    def __init__(self, instance: CVRPTWInstance, config: SolverConfig = None):
        self.instance = instance
        self.config = config or SolverConfig()

    def solve(self, method: str = "lns") -> Solution:
        """
        求解CVRPTW问题
        支持多种求解方法：greedy（贪心）、lns（大邻域搜索）、lns_vns（混合方法）
        """
        logger.info(f"使用{method}方法求解CVRPTW问题")

        if method == "greedy":
            # 仅使用贪心构造，速度快但质量一般
            constructor = GreedyConstructor(self.instance)
            return constructor.construct()

        elif method == "lns":
            # 使用大邻域搜索，平衡求解质量和速度
            lns = LargeNeighborhoodSearch(self.instance, self.config)
            return lns.solve()

        elif method == "lns_vns":
            # LNS + VNS混合方法，先用LNS全局搜索，再用VNS局部优化
            lns = LargeNeighborhoodSearch(self.instance, self.config)
            lns_solution = lns.solve()

            vns = VariableNeighborhoodSearch(self.instance, self.config)
            return vns.solve(lns_solution)

        else:
            raise ValueError(f"不支持的求解方法: {method}")


def create_instance_from_args(args: dict) -> CVRPTWInstance:
    """
    从参数字典创建问题实例
    将原始数据转换为算法所需的数据结构
    """
    return CVRPTWInstance(
        customers=args["customers"],
        fleets=args["fleets"],
        nodes=args["nodes"],
        nodes_fleet=args["nodes_fleet"],
        vehicles=args["vehicles"],
        distance=args["distance"],
        travel_time=args["travel_time"],
        start_early=args["start_early"],
        start_late=args["start_late"],
        duration=args["duration"],
        precedence=args["precedence"],
        flight_type=args["flight_type"],
        need=args["need"],
        adjusted_departure=args["adjusted_departure"]
    )


def solve_instance(fleet_info, distance_dict, val_dataset, opts):
    """
    求解问题实例的主函数
    处理数据加载、实例构建、求解和结果统计的完整流程
    """
    cost = []  # 存储每个实例的目标值
    batch_size = 1
    assert batch_size == 1, "只能逐个求解！"

    # 遍历数据集中的每个实例
    for instance_idx, input in enumerate(
            tqdm(DataLoader(val_dataset, batch_size=batch_size, shuffle=False), disable=opts.no_progress_bar)):

        # ==================== 数据预处理 ====================
        # 提取输入数据
        loc = input["loc"][0]  # 位置坐标
        arrival = input["arrival"][0]  # 到达时间
        departure = input["departure"][0]  # 离开时间
        type = input["type"][0]  # 航班类型
        need_tensor = input["need"][0]  # 客户需求
        need = need_tensor.tolist()

        # 问题规模设置
        fleet_size = 6  # 车队数量
        graph_size = loc.shape[0]  # 客户数量
        # 根据问题规模设置初始车辆数
        initial_vehicle = {20: 15, 50: 20, 100: 25, 200: 40, 300: 50}.get(opts.graph_size, 20)

        # 构建节点集合
        customers = [j for j in range(1, graph_size + 1)]  # 客户节点：1到graph_size
        fleets = [j for j in range(1, fleet_size + 1)]  # 车队：1到6
        # 所有节点 = 客户节点 + depot节点
        nodes = customers + [j for j in range(graph_size + 1, graph_size + 1 + fleet_size * 2)]

        # ==================== 时间窗设置 ====================
        # 初始化时间窗约束
        start_ea = {i: -60 for i in nodes}  # 最早开始时间（默认-60）
        start_la = {i: {f: 10 ** 4 for f in fleets} for i in nodes}  # 最晚开始时间（默认很大）

        adjusted_departure = {}  # 调整后的离开时间

        # 为每个客户设置具体的时间窗
        for i in customers:
            start_ea[i] = arrival[i - 1].item()  # 最早开始时间 = 到达时间
            original_departure = departure[i - 1].item()
            adjusted_departure[i] = {}
            start_la[i] = {}

            # 为每个车队计算调整后的最晚开始时间
            for f in fleets:
                flight_type_i = type[i - 1].item()
                precedence_f = fleet_info['precedence'][f]  # 车队的优先级信息

                # 根据优先级调整离开时间
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

        # ==================== 车队-节点映射 ====================
        # 每个车队可以访问的节点（客户+对应的depot）
        nodes_F = {}
        for f in range(fleet_size):
            # 每个车队可以访问所有客户 + 自己的起始和结束depot
            nodes_F[f + 1] = customers + [graph_size + 1 + 2 * f, graph_size + 2 + 2 * f]

        # 每个车队的车辆列表
        vehicles = {f: [j for j in range(1, initial_vehicle + 1)] for f in range(1, fleet_size + 1)}

        # ==================== 距离矩阵构建 ====================
        distance = {(i, j): 0 for i in nodes for j in nodes}  # 初始化距离矩阵
        for i in nodes:
            # 客户节点使用实际坐标，depot节点坐标为0
            id_i = loc[i - 1].item() if i in customers else 0
            for j in nodes:
                id_j = loc[j - 1].item() if j in customers else 0
                distance[(i, j)] = distance_dict[(id_i, id_j)]  # 从预计算的距离字典获取

        # ==================== 服务时间和旅行时间 ====================
        flight_type = {i: type[i - 1].item() for i in customers}  # 每个客户的航班类型

        # 构建每个车队的服务时间
        S_time = {}
        for f in range(1, fleet_size + 1):
            duration = []
            for i in customers:
                # 根据车队和航班类型确定服务时间
                duration.append(fleet_info["duration"][f][flight_type[i]])
            S_time[f] = duration + [0.0] * 2 * fleet_size

        # 构建旅行时间矩阵（服务时间 + 行驶时间）
        tau = {(i, j, f): (S_time[f][i - 1] if i in customers else 0.0) + distance[i, j] / 80.0
               for i in nodes for j in nodes for f in fleets if i != j}

        # ==================== 优先级约束 ====================
        def get_precedence_from_fleet_info():
            """从车队信息中提取优先级约束"""
            precedence_pairs = []
            # 定义车队间的优先级关系：后续车队必须在前序车队完成后才能开始
            precedence_mapping = {3: 2, 5: 4, 6: 1}  # {后续车队: 前序车队}
            for fleet, prev_fleet in precedence_mapping.items():
                precedence_pairs.append([prev_fleet, fleet])
            return precedence_pairs

        prec = get_precedence_from_fleet_info()
        precedence = {i: prec for i in customers}  # 每个客户都有相同的优先级约束

        # ==================== 构建问题参数字典 ====================
        args = {
            "customers": customers,
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
            "adjusted_departure": adjusted_departure
        }

        # ==================== 求解过程 ====================
        # 创建问题实例
        instance = create_instance_from_args(args)

        # 配置求解器参数
        config = SolverConfig()
        # 根据问题规模设置时间限制
        timelimit_map = {20: 1800, 50: 1800, 100: 1800, 200: 3600, 300: 3600}
        config.timelimit = timelimit_map.get(opts.graph_size, 1800)
        config.max_iterations = 500

        # 创建求解器
        solver = CVRPTWSolver(instance, config)

        print(f"\n--- 实例 {instance_idx + 1} 开始求解 ---")
        start_time = time.time()

        # 根据指定方法求解
        try:
            if opts.val_method == "greedy":
                solution = solver.solve("greedy")
            elif opts.val_method == "lns":
                solution = solver.solve("lns")
            elif opts.val_method == "lns_vns":
                solution = solver.solve("lns_vns")
            else:
                logger.warning(f"未知求解方法: {opts.val_method}, 使用LNS")
                solution = solver.solve("lns")

            solve_time = time.time() - start_time

            # ==================== 结果处理 ====================
            if solution.feasible:
                print(f"找到可行解，目标值: {solution.objective_value:.2f}")
                print(f"求解时间: {solve_time:.2f}秒")
                cost.append(solution.objective_value)

                # 输出解的统计信息
                total_routes = sum(len(routes) for routes in solution.routes.values())
                print(f"使用路径数: {total_routes}")

                # 输出每个车队的路径信息
                for fleet in solution.routes:
                    fleet_routes = [route for route in solution.routes[fleet].values() if route]
                    if fleet_routes:
                        print(f"车队 {fleet}: {len(fleet_routes)} 条路径")

            else:
                print("未找到可行解")
                cost.append(None)

        except Exception as e:
            logger.error(f"求解过程中出现错误: {str(e)}")
            cost.append(None)

    # ==================== 结果统计 ====================
    print("\n=== 求解结果汇总 ===")
    print("所有实例的目标值列表:", cost)

    valid_costs = [c for c in cost if c is not None]
    if valid_costs:
        average_cost = sum(valid_costs) / len(valid_costs)
        min_cost = min(valid_costs)
        max_cost = max(valid_costs)

        print(f"成功求解实例数: {len(valid_costs)}/{len(cost)}")
        print(f"平均目标值: {average_cost:.2f}")
        print(f"最小目标值: {min_cost:.2f}")
        print(f"最大目标值: {max_cost:.2f}")

        if len(valid_costs) > 1:
            std_cost = np.std(valid_costs)
            print(f"标准差: {std_cost:.2f}")
    else:
        print("没有找到任何可行解")

    return cost


class CVRPTWAnalyzer:
    """
    CVRPTW解分析器
    用于分析求解结果的质量和特征
    """

    def __init__(self, instance: CVRPTWInstance, solution: Solution):
        self.instance = instance
        self.solution = solution

    def analyze(self) -> Dict[str, Any]:
        """
        分析解的质量和特征
        返回包含各种统计指标的字典
        """
        analysis = {}

        # 基本统计指标
        analysis['objective_value'] = self.solution.objective_value
        analysis['feasible'] = self.solution.feasible
        analysis['total_distance'] = self.solution.objective_value

        # 路径统计
        total_routes = 0
        fleet_stats = {}

        for fleet in self.solution.routes:
            # 统计每个车队的活跃路径
            active_routes = [route for route in self.solution.routes[fleet].values() if route]
            total_routes += len(active_routes)

            fleet_stats[fleet] = {
                'routes': len(active_routes),  # 路径数量
                'customers_served': sum(len(route) for route in active_routes),  # 服务的客户数
                'avg_route_length': np.mean([len(route) for route in active_routes]) if active_routes else 0  # 平均路径长度
            }

        analysis['total_routes'] = total_routes
        analysis['fleet_statistics'] = fleet_stats

        # 客户服务统计
        served_customers = set()
        for fleet in self.solution.routes:
            for route in self.solution.routes[fleet].values():
                served_customers.update(route)

        analysis['customers_served'] = len(served_customers)
        analysis['service_rate'] = len(served_customers) / len(self.instance.customers)

        # 时间窗利用率
        if hasattr(self.solution, 'schedules'):
            time_window_utilization = self._calculate_time_window_utilization()
            analysis['time_window_utilization'] = time_window_utilization

        return analysis

    def _calculate_time_window_utilization(self) -> float:
        """
        计算时间窗利用率
        衡量服务时间在时间窗中的分布情况
        """
        total_utilization = 0
        count = 0

        for (node, fleet), start_time in self.solution.schedules.items():
            if node in self.instance.customers:
                earliest = self.instance.start_early.get(node, 0)
                latest = self.instance.start_late.get(node, {}).get(fleet, float('inf'))

                if latest != float('inf'):
                    window_size = latest - earliest
                    if window_size > 0:
                        # 利用率 = (实际服务时间 - 最早时间) / 时间窗大小
                        utilization = (start_time - earliest) / window_size
                        total_utilization += min(1.0, max(0.0, utilization))
                        count += 1

        return total_utilization / count if count > 0 else 0.0

    def print_detailed_solution(self):
        """打印详细的解信息，用于调试和分析"""
        print("\n=== 详细解信息 ===")
        print(f"目标值: {self.solution.objective_value:.2f}")
        print(f"可行性: {'是' if self.solution.feasible else '否'}")

        # 按车队输出路径信息
        for fleet in sorted(self.solution.routes.keys()):
            print(f"\n车队 {fleet}:")
            active_routes = {v: r for v, r in self.solution.routes[fleet].items() if r}

            if not active_routes:
                print("  无活跃路径")
                continue

            # 输出每条路径的详细信息
            for vehicle, route in sorted(active_routes.items()):
                route_distance = self._calculate_route_distance(fleet, route)
                print(f"  车辆 {vehicle}: {route} (距离: {route_distance:.2f})")

                # 打印每个客户的服务时间信息
                if hasattr(self.solution, 'schedules'):
                    for customer in route:
                        if (customer, fleet) in self.solution.schedules:
                            start_time = self.solution.schedules[(customer, fleet)]
                            earliest = self.instance.start_early.get(customer, 0)
                            latest = self.instance.start_late.get(customer, {}).get(fleet, float('inf'))
                            print(
                                f"    客户 {customer}: 服务时间 {start_time:.1f} (窗口: [{earliest:.1f}, {latest:.1f}])")

    def _calculate_route_distance(self, fleet: int, route: List[int]) -> float:
        """计算单条路径的总距离"""
        if not route:
            return 0.0

        distance = 0.0

        # 从起始depot到第一个客户
        start_depot = len(self.instance.customers) + fleet * 2 - 1
        distance += self.instance.distance.get((start_depot, route[0]), 0)

        # 客户之间的距离
        for i in range(len(route) - 1):
            distance += self.instance.distance.get((route[i], route[i + 1]), 0)

        # 从最后一个客户到结束depot
        end_depot = len(self.instance.customers) + fleet * 2
        distance += self.instance.distance.get((route[-1], end_depot), 0)

        return distance


def export_solution(solution: Solution, instance: CVRPTWInstance, filename: str):
    """
    导出解到JSON文件
    便于后续分析和可视化
    """
    solution_data = {
        'objective_value': solution.objective_value,
        'feasible': solution.feasible,
        'routes': solution.routes,
        'schedules': getattr(solution, 'schedules', {}),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(solution_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"解已导出到: {filename}")


def main():
    """
    主函数
    解析命令行参数，加载数据，调用求解器
    """
    parser = argparse.ArgumentParser(description="改进的CVRPTW求解器")
    parser.add_argument("--filename", type=str, default="./data/agh/agh300_validation_seed4321.pkl",
                        help="数据集文件名")
    parser.add_argument("--problem", type=str, default='agh', help="目前只支持HHCRSP问题")
    parser.add_argument('--graph_size', type=int, default=300, help="问题实例规模 (20, 50, 100, 200, 300)")
    parser.add_argument('--val_method', type=str, default='lns',
                        choices=['greedy', 'lns', 'lns_vns'], help="求解方法")
    parser.add_argument('--val_size', type=int, default=1, help='用于验证性能的实例数量')
    parser.add_argument('--offset', type=int, default=0, help='数据集中开始的偏移量')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--no_progress_bar', action='store_true', help='禁用进度条')
    parser.add_argument('--export_solution', type=str, help='导出解的文件路径')
    parser.add_argument('--detailed_analysis', action='store_true', help='输出详细分析')

    opts = parser.parse_args()
    print("命令行参数:")
    pp.pprint(vars(opts))

    # 设置随机种子，确保结果可重现
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)

    # ==================== 数据加载 ====================
    fleet_info_path, distance_path = 'problems/agh/fleet_info.pkl', 'problems/agh/distance.pkl'
    try:
        # 加载车队信息（服务时间、优先级等）
        with open(fleet_info_path, 'rb') as f:
            fleet_info = pickle.load(f)
        # 加载预计算的距离矩阵
        with open(distance_path, 'rb') as f:
            distance_dict = pickle.load(f)
    except FileNotFoundError as e:
        print(f"错误: 无法找到数据文件 {e}")
        sys.exit(1)

    # 加载问题实例数据
    problem = load_problem(opts.problem)
    val_dataset = problem.make_dataset(filename=opts.filename, num_samples=opts.val_size, offset=opts.offset)
    print(f'正在验证数据集: {opts.filename}')

    # ==================== 开始求解 ====================
    start_time = time.time()
    costs = solve_instance(fleet_info, distance_dict, val_dataset, opts)
    total_time = time.time() - start_time

    # ==================== 输出最终结果 ====================
    print(f"\n>> 验证结束，总耗时 {total_time:.2f} 秒")
    print(f">> 平均每实例耗时 {total_time / len(costs):.2f} 秒")


if __name__ == "__main__":
    main()
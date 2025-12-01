# -*- coding: utf-8 -*-
'''
遗传算法求解CVRPTW问题
修改输出格式，只打印最终平均距离成本
'''

import os
import io
import sys
import random
import numpy as np
from csv import DictWriter
from deap import base, creator, tools
import argparse
import time
import pickle
import torch
import pprint as pp
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import move_to, load_problem


class CVRPTWInstance:
    """
    CVRPTW问题实例类，基于代码一的逻辑重构
    """

    def __init__(self, args):
        """
        初始化CVRPTW问题实例
        """
        self.args = args
        self.customers = args["customers"]
        self.fleets = args["fleets"]
        self.vehicles = args["vehicles"]
        self.nodes_fleet = args["nodes_fleet"]
        self.distance = args["distance"]
        self.duration = args["duration"]
        self.start_early = args["start_early"]
        self.start_late = args["start_late"]
        self.precedence = args["precedence"]
        self.need = args["need"]
        self.travel_time = args["travel_time"]

        # 车辆速度
        self.speed = 80.0

        # 移除初始化时的打印语句
        # print(f"初始化问题实例: {len(self.customers)} 个客户, {len(self.fleets)} 个车队")

    def get_depot_start(self, fleet):
        """获取车队的起始depot节点ID"""
        n = len(self.customers)
        return n + fleet * 2 - 1

    def get_depot_end(self, fleet):
        """获取车队的结束depot节点ID"""
        n = len(self.customers)
        return n + fleet * 2

    def can_fleet_visit_customer(self, fleet, customer):
        """
        判断车队是否可以访问某个客户
        """
        if customer < 1 or customer > len(self.customers):
            return False

        target_need = self.need[customer - 1]

        if fleet == 1:
            return (target_need == 1) or (target_need == 9)
        elif fleet == 2:
            return (target_need == 2) or (target_need == 7)
        elif fleet == 3:
            return (target_need == 3) or (target_need == 7)
        elif fleet == 4:
            return (target_need == 4) or (target_need == 8)
        elif fleet == 5:
            return (target_need == 5) or (target_need == 8)
        elif fleet == 6:
            return (target_need == 6) or (target_need == 9)
        else:
            return target_need == fleet


def individual_to_routes(individual, instance):
    """
    将个体转换为路径，改进版本
    """
    # 按车队和车辆分组
    fleet_routes = {}
    for fleet in instance.fleets:
        fleet_routes[fleet] = {}
        for vehicle in instance.vehicles[fleet]:
            fleet_routes[fleet][vehicle] = []

    # 验证并分配客户到对应的车队车辆
    for gene in individual:
        if len(gene) == 3:
            customer_id, fleet_id, vehicle_id = gene

            # 检查有效性
            if (fleet_id in fleet_routes and
                    vehicle_id in fleet_routes[fleet_id] and
                    customer_id in instance.customers):
                fleet_routes[fleet_id][vehicle_id].append(customer_id)

    return fleet_routes


def evaluate_cvrptw(individual, instance, unit_cost=1.0, init_cost=100.0, wait_cost=0.5, delay_cost=2.0):
    """
    修复后的适应度评估函数
    修改：返回适应度和总距离成本
    """
    if not individual:
        return (1e-10, 0.0)  # 避免返回0，并返回0距离成本

    total_cost = 0.0
    total_distance_cost = 0.0  # 新增：用于记录纯粹的距离成本
    penalty = 0.0

    try:
        fleet_routes = individual_to_routes(individual, instance)

        visited_customers = set()

        for fleet_id in instance.fleets:
            for vehicle_id in instance.vehicles[fleet_id]:
                route = fleet_routes[fleet_id][vehicle_id]

                if not route:
                    continue

                route_cost = init_cost
                current_time = -60.0
                current_node = instance.get_depot_start(fleet_id)

                for customer_id in route:
                    visited_customers.add(customer_id)

                    if not instance.can_fleet_visit_customer(fleet_id, customer_id):
                        penalty += 50000.0
                        continue

                    distance_key = (current_node, customer_id)
                    if distance_key in instance.distance:
                        distance = instance.distance[distance_key]
                    else:
                        distance = 0.0
                        penalty += 1000.0

                    # 累加距离成本
                    distance_cost = unit_cost * distance
                    route_cost += distance_cost
                    total_distance_cost += distance_cost

                    travel_time = distance / instance.speed
                    arrival_time = current_time + travel_time

                    earliest = instance.start_early.get(customer_id, -60.0)
                    if customer_id in instance.start_late and fleet_id in instance.start_late[customer_id]:
                        latest = instance.start_late[customer_id][fleet_id]
                    else:
                        latest = 10000.0

                    if arrival_time < earliest:
                        waiting_time = earliest - arrival_time
                        route_cost += wait_cost * waiting_time
                        service_start_time = earliest
                    elif arrival_time > latest:
                        delay_time = arrival_time - latest
                        route_cost += delay_cost * delay_time
                        penalty += delay_time * 100.0
                        service_start_time = arrival_time
                    else:
                        service_start_time = arrival_time

                    if fleet_id in instance.duration and customer_id - 1 < len(instance.duration[fleet_id]):
                        service_duration = instance.duration[fleet_id][customer_id - 1]
                    else:
                        service_duration = 10.0

                    current_time = service_start_time + service_duration
                    current_node = customer_id

                if route:
                    return_key = (current_node, instance.get_depot_end(fleet_id))
                    if return_key in instance.distance:
                        return_distance = instance.distance[return_key]
                    else:
                        return_distance = 0.0
                    return_distance_cost = unit_cost * return_distance
                    route_cost += return_distance_cost
                    total_distance_cost += return_distance_cost

                total_cost += route_cost

        unvisited_count = len(instance.customers) - len(visited_customers)
        penalty += unvisited_count * 100000.0

        penalty += check_precedence_constraints_improved(individual, instance)

        final_cost = max(total_cost + penalty, 1.0)
        fitness = 1.0 / final_cost

        return (fitness, total_distance_cost)

    except Exception as e:
        print(f"适应度评估出错: {e}")
        return (1e-10, 0.0)


def check_precedence_constraints_improved(individual, instance):
    """
    改进的优先级约束检查
    """
    penalty = 0.0

    try:
        # 构建客户访问顺序
        customer_fleet_time = {}

        for i, gene in enumerate(individual):
            if len(gene) == 3:
                customer_id, fleet_id, vehicle_id = gene
                if customer_id not in customer_fleet_time:
                    customer_fleet_time[customer_id] = {}
                # 使用基因位置作为时间序号的近似
                customer_fleet_time[customer_id][fleet_id] = i

        # 检查优先级约束
        for customer_id in instance.customers:
            if customer_id in instance.precedence:
                for prec_pair in instance.precedence[customer_id]:
                    if len(prec_pair) >= 2:
                        prev_fleet, next_fleet = prec_pair[0], prec_pair[1]

                        if (customer_id in customer_fleet_time and
                                prev_fleet in customer_fleet_time[customer_id] and
                                next_fleet in customer_fleet_time[customer_id]):

                            prev_time = customer_fleet_time[customer_id][prev_fleet]
                            next_time = customer_fleet_time[customer_id][next_fleet]

                            if prev_time >= next_time:
                                penalty += 50000.0

        return penalty

    except Exception as e:
        print(f"优先级约束检查出错: {e}")
        return 10000.0


def create_individual(instance):
    """
    改进的个体创建函数
    """
    individual = []

    for customer_id in instance.customers:
        # 找到可以访问该客户的车队
        available_fleets = []
        for fleet_id in instance.fleets:
            if instance.can_fleet_visit_customer(fleet_id, customer_id):
                available_fleets.append(fleet_id)

        if available_fleets:
            fleet_id = random.choice(available_fleets)
            if fleet_id in instance.vehicles and instance.vehicles[fleet_id]:
                vehicle_id = random.choice(instance.vehicles[fleet_id])
                individual.append((customer_id, fleet_id, vehicle_id))
            else:
                # 如果没有可用车辆，随机分配
                fleet_id = random.choice(instance.fleets)
                vehicle_id = random.choice(instance.vehicles[fleet_id])
                individual.append((customer_id, fleet_id, vehicle_id))
        else:
            # 如果没有车队可以访问，随机分配
            fleet_id = random.choice(instance.fleets)
            vehicle_id = random.choice(instance.vehicles[fleet_id])
            individual.append((customer_id, fleet_id, vehicle_id))

    # 随机打乱顺序
    random.shuffle(individual)
    return individual


def crossover_cvrptw(ind1, ind2):
    """
    改进的交叉操作
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    size = min(len(ind1), len(ind2))
    if size < 2:
        return ind1, ind2

    # 单点交叉
    cx_point = random.randint(1, size - 1)

    temp = ind1[cx_point:]
    ind1[cx_point:] = ind2[cx_point:]
    ind2[cx_point:] = temp

    return ind1, ind2


def mutate_cvrptw(individual, instance, mut_pb=0.1):
    """
    改进的变异操作
    """
    mutated = False

    for i in range(len(individual)):
        if random.random() < mut_pb:
            customer_id, current_fleet, current_vehicle = individual[i]

            # 尝试改变车队分配
            available_fleets = []
            for fleet_id in instance.fleets:
                if instance.can_fleet_visit_customer(fleet_id, customer_id):
                    available_fleets.append(fleet_id)

            if available_fleets and len(available_fleets) > 1:
                # 选择不同的车队
                new_fleets = [f for f in available_fleets if f != current_fleet]
                if new_fleets:
                    new_fleet = random.choice(new_fleets)
                    new_vehicle = random.choice(instance.vehicles[new_fleet])
                    individual[i] = (customer_id, new_fleet, new_vehicle)
                    mutated = True

            # 或者改变车辆分配
            elif random.random() < 0.5:
                available_vehicles = [v for v in instance.vehicles[current_fleet] if v != current_vehicle]
                if available_vehicles:
                    new_vehicle = random.choice(available_vehicles)
                    individual[i] = (customer_id, current_fleet, new_vehicle)
                    mutated = True

    # 有时进行位置变异（改变服务顺序）
    if random.random() < mut_pb and len(individual) > 1:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
        mutated = True

    return (individual,)


def print_solution(individual, instance):
    """
    打印解的详细信息
    """
    fleet_routes = individual_to_routes(individual, instance)

    total_distance = 0
    vehicle_count = 0

    print("\n=== 解决方案详情 ===")

    for fleet_id in instance.fleets:
        fleet_has_routes = False

        for vehicle_id in instance.vehicles[fleet_id]:
            route = fleet_routes[fleet_id][vehicle_id]

            if route:
                if not fleet_has_routes:
                    print(f"\n车队 {fleet_id}:")
                    fleet_has_routes = True

                vehicle_count += 1
                route_distance = 0

                # 计算路径距离
                current_node = instance.get_depot_start(fleet_id)
                route_str = f"Depot({current_node})"

                for customer_id in route:
                    distance_key = (current_node, customer_id)
                    distance = instance.distance.get(distance_key, 0)
                    route_distance += distance
                    route_str += f" -> {customer_id}"
                    current_node = customer_id

                # 返回depot
                return_key = (current_node, instance.get_depot_end(fleet_id))
                return_distance = instance.distance.get(return_key, 0)
                route_distance += return_distance
                route_str += f" -> Depot({instance.get_depot_end(fleet_id)})"

                total_distance += route_distance
                print(f"  车辆 {vehicle_id}: {route_str} (距离: {route_distance:.2f})")

        if not fleet_has_routes:
            print(f"\n车队 {fleet_id}: 未使用")

    print(f"\n总使用车辆数: {vehicle_count}")
    print(f"总行驶距离: {total_distance:.2f}")


def run_ga_cvrptw(instance, unit_cost=1.0, init_cost=100.0, wait_cost=0.5, delay_cost=2.0,
                  pop_size=50, cx_pb=0.8, mut_pb=0.2, n_gen=100, export_csv=False, verbose=False):
    """
    运行遗传算法求解CVRPTW问题
    修改：在最后返回最佳解时，一并返回其总距离成本
    """
    # 清理之前的类型定义
    if hasattr(creator, 'FitnessMax'):
        del creator.FitnessMax
    if hasattr(creator, 'Individual'):
        del creator.Individual

    # 设置DEAP
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('individual', lambda: creator.Individual(create_individual(instance)))
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # 使用一个辅助函数来包装evaluate，以便只将适应度值赋给个体的fitness属性
    def evaluate_wrapper(individual, *args, **kwargs):
        fitness, _ = evaluate_cvrptw(individual, *args, **kwargs)
        return (fitness,)
    toolbox.register('evaluate', evaluate_wrapper, instance=instance,
                     unit_cost=unit_cost, init_cost=init_cost,
                     wait_cost=wait_cost, delay_cost=delay_cost)
    toolbox.register('mate', crossover_cvrptw)
    toolbox.register('mutate', mutate_cvrptw, instance=instance, mut_pb=mut_pb)
    toolbox.register('select', tools.selTournament, tournsize=3)

    # 初始化种群
    if verbose:
        print('正在初始化种群...')
    pop = toolbox.population(n=pop_size)

    if verbose:
        print('开始进化...')

    # 评估初始种群
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    if verbose:
        print(f'评估了 {len(pop)} 个个体')

    # 检查初始适应度
    if verbose:
        init_fits = [ind.fitness.values[0] for ind in pop]
        print(f'初始适应度范围: {min(init_fits):.8f} - {max(init_fits):.8f}')

    csv_data = []
    best_fitness_history = []

    # 进化主循环
    for gen in range(n_gen):
        if verbose:
            print(f'-- 第 {gen} 代 --')

        # 选择
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 重新评估
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if verbose:
            print(f'  评估了 {len(invalid_ind)} 个个体')

        # 更新种群
        pop[:] = offspring

        # 统计信息
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum([x ** 2 for x in fits])
        std = abs(sum2 / length - mean ** 2) ** 0.5

        min_fit = min(fits)
        max_fit = max(fits)

        if verbose:
            print(f'  最小适应度: {min_fit:.8f}')
            print(f'  最大适应度: {max_fit:.8f}')
            print(f'  平均适应度: {mean:.8f}')
            print(f'  标准差: {std:.8f}')

            if max_fit > 0:
                min_cost = 1.0 / max_fit
                print(f'  最小成本: {min_cost:.2f}')
            else:
                print(f'  最小成本: 无穷大')

        best_fitness_history.append(max_fit)

        if export_csv:
            csv_row = {
                'generation': gen,
                'min_fitness': min_fit,
                'max_fitness': max_fit,
                'avg_fitness': mean,
                'std_fitness': std,
                'min_cost': 1.0 / max_fit if max_fit > 0 else float('inf')
            }
            csv_data.append(csv_row)

    if verbose:
        print('-- 进化完成 --')

    # 输出最优解
    best_ind = tools.selBest(pop, 1)[0]
    best_fitness = best_ind.fitness.values[0]

    # 获取最佳解的距离成本
    _, best_distance_cost = evaluate_cvrptw(best_ind, instance,
                                            unit_cost=unit_cost, init_cost=init_cost,
                                            wait_cost=wait_cost, delay_cost=delay_cost)

    if verbose:
        print(f'\n最优个体适应度: {best_fitness:.8f}')
        if best_fitness > 0:
            print(f'最优总成本: {1.0 / best_fitness:.2f}')
            print(f'最优总距离成本: {best_distance_cost:.2f}') # 增加这一行
            print_solution(best_ind, instance)
        else:
            print('未找到可行解')

    return best_ind, csv_data, best_distance_cost


def get_precedence_from_fleet_info():
    """
    根据车队优先级关系生成优先级对
    """
    precedence_pairs = []

    # 硬编码正确的优先级关系
    precedence_mapping = {
        3: 2,  # 车队3的前驱是车队2
        5: 4,  # 车队5的前驱是车队4
        6: 1  # 车队6的前驱是车队1
    }

    for fleet, prev_fleet in precedence_mapping.items():
        precedence_pairs.append([prev_fleet, fleet])

    return precedence_pairs


def solve_instance_ga(fleet_info, distance_dict, val_dataset, opts):
    """
    使用遗传算法求解问题实例的主函数
    """
    costs, distance_costs = [], []
    batch_size = 1
    assert batch_size == 1, "只能逐个求解！"

    for instance_idx, input in enumerate(
            tqdm(DataLoader(val_dataset, batch_size=batch_size, shuffle=False), disable=opts.no_progress_bar)):
        # 数据处理代码（与原代码相同）
        loc = input["loc"][0]
        arrival = input["arrival"][0]
        departure = input["departure"][0]
        type = input["type"][0]
        need_tensor = input["need"][0]
        need = need_tensor.tolist()

        fleet_size = 6
        graph_size = loc.shape[0]
        initial_vehicle = {20: 15, 50: 20, 100: 25, 200: 40, 300: 50}.get(opts.graph_size, 25)

        customers = [j for j in range(1, graph_size + 1)]
        fleets = [j for j in range(1, fleet_size + 1)]
        nodes = customers + [j for j in range(graph_size + 1, graph_size + 1 + fleet_size * 2)]

        start_ea = {i: -60 for i in nodes}
        start_la = {i: {f: 10 ** 4 for f in fleets} for i in nodes}

        # 计算调整后的时间窗
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
                distance[(i, j)] = distance_dict[(id_i, id_j)]

        flight_type = {i: type[i - 1].item() for i in customers}
        S_time = {}
        for f in range(1, fleet_size + 1):
            duration = []
            for i in customers:
                duration.append(fleet_info["duration"][f][flight_type[i]])
            S_time[f] = duration + [0.0] * 2 * fleet_size

        tau = {(i, j, f): (S_time[f][i - 1] if i in customers else 0.0) + distance[i, j] / 80.0
               for i in nodes for j in nodes for f in fleets if i != j}

        prec = get_precedence_from_fleet_info()
        precedence = {i: prec for i in customers}

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
                }

        # 遗传算法求解
        instance_start_time = time.time()
        instance = CVRPTWInstance(args)

        # 运行遗传算法，不显示详细信息，并接收距离成本
        best_solution, csv_data, final_distance_cost = run_ga_cvrptw(
            instance=instance,
            unit_cost=1.1,
            init_cost=20.0,
            wait_cost=0.5,
            delay_cost=0.5,
            pop_size=opts.pop_size,
            cx_pb=opts.cx_pb,
            mut_pb=opts.mut_pb,
            n_gen=opts.n_gen,
            export_csv=True,
            verbose=False  # 关闭详细输出
        )

        # 收集最终成本和距离成本
        if best_solution.fitness.values[0] > 0:
            final_cost = 1.0 / best_solution.fitness.values[0]
            costs.append(final_cost)
            distance_costs.append(final_distance_cost)
        else:
            costs.append(float('inf'))
            distance_costs.append(float('inf'))

        instance_end_time = time.time()

    # 计算并打印平均距离成本
    valid_distance_costs = [c for c in distance_costs if c != float('inf')]
    if valid_distance_costs:
        avg_distance_cost = sum(valid_distance_costs) / len(valid_distance_costs)
        print(f"\n=== 求解结果 ===")
        print(f"平均距离成本: {avg_distance_cost:.2f}")
        print(f"有效解实例数: {len(valid_distance_costs)}/{len(distance_costs)}")
    else:
        print("\n=== 求解结果 ===")
        print("未找到任何可行解")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="./data/agh/agh300_validation_seed4321.pkl",
                        help="数据集文件名")
    parser.add_argument("--problem", type=str, default='agh', help="目前只支持机场地勤(agh)问题")
    parser.add_argument('--graph_size', type=int, default=300, help="问题实例规模")
    parser.add_argument('--val_method', type=str, default='ga', choices=['ga'], help="验证方法")
    parser.add_argument('--val_size', type=int, default=1, help='验证实例数量')
    parser.add_argument('--offset', type=int, default=0, help='数据集偏移量')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--no_progress_bar', action='store_true', help='禁用进度条')

    # GA特有参数
    parser.add_argument('--pop_size', type=int, default=300, help='种群大小')
    parser.add_argument('--cx_pb', type=float, default=0.8, help='交叉概率')
    parser.add_argument('--mut_pb', type=float, default=0.2, help='变异概率')
    parser.add_argument('--n_gen', type=int, default=4500, help='进化代数')

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
            print(f"车队信息加载成功")
        with open(distance_path, 'rb') as f:
            distance_dict = pickle.load(f)
            print(f"距离信息加载成功")
    except FileNotFoundError as e:
        print(f"错误: 无法找到数据文件。请确保以下文件存在于正确路径:")
        print(f"- {fleet_info_path}")
        print(f"- {distance_path}")
        print("请检查你的工作目录或文件路径是否正确。")
        sys.exit(1)

    problem = load_problem(opts.problem)
    val_dataset = problem.make_dataset(filename=opts.filename, num_samples=opts.val_size, offset=opts.offset)
    print(f'正在验证数据集: {opts.filename}')

    start_time = time.time()
    solve_instance_ga(fleet_info, distance_dict, val_dataset, opts)
    print(f">> 验证结束，总耗时 {time.time() - start_time:.2f} 秒")


if __name__ == '__main__':
    main()
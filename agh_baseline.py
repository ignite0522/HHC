import os, random
import math
import copy
import time
import torch
import math
import pickle
import argparse
import pprint as pp
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils import move_to, load_problem
from torch.utils.data import DataLoader
from multiprocessing import Pool
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用 GPU 0


def cws(input_, problem, opt):
    """
        Clarke and Wright Savings (CWS) 算法实现
    """
    NODE_SIZE, SPEED = 92, 80.0

    sequences = [] # 存储生成的路径序列
    state = problem.make_state(input_) # 初始化问题状态
    batch_size = input_['loc'].size(0) # 获取批次大小
    ids = torch.arange(batch_size)[:, None] # 批次实例索引


    ## 计算节省 (Savings)

    # CWS 核心：计算合并路径能节省的成本（距离 + 时间）。
    # 节省 = (D-i距离 + i-D距离) + (D-j距离 + j-D距离) - (D-i距离 + i-j距离 + j-D距离)
    # 简化后，距离节省 = (i回D的距离 + D到j的距离) - (i到j的距离)

    # 1. 提取从中心点/起点到每个客户点的距离 (from_depot)
    from_depot_locs = input_['loc']
    from_depot_dist = input_['distance'].gather(1, from_depot_locs)

    # 2. 提取从每个客户点返回中心点/终点的距离 (to_depot)
    to_depot_locs = NODE_SIZE * input_['loc']
    to_depot_dist = input_['distance'].gather(1, to_depot_locs)

    # 扩展距离张量，便于广播计算
    from_depot = from_depot_dist[:, None, :].repeat(1, opt.graph_size, 1)
    to_depot = to_depot_dist[:, :, None].repeat(1, 1, opt.graph_size)

    # 3. 计算客户点 i 到客户点 j 的距离 (i_j_distance)
    i_index = input_['loc'][:, :, None].repeat(1, 1, opt.graph_size)
    j_index = input_['loc'][:, None, :].repeat(1, opt.graph_size, 1)
    i_j = NODE_SIZE * i_index + j_index # 扁平化距离矩阵索引
    temp_distance = input_['distance'][:, None, :].expand(batch_size, opt.graph_size, NODE_SIZE * NODE_SIZE)
    i_j_distance = temp_distance.gather(2, i_j) # 获取实际距离

    # 4. 计算距离节省 (Savings_distance)
    savings_distance = from_depot + to_depot - i_j_distance # 经典 CWS 距离节省公式

    # 5. 计算时间节省 (Savings_time) - 针对带时间窗的 VRP
    # 忽略索引为0的中心点时间窗
    tw_left, tw_right = input_['tw_left'][:, 1:], input_['tw_right'][:, 1:]
    # 计算时间窗差异
    savings_time = tw_left[:, None, :].repeat(1, opt.graph_size, 1) - tw_left[:, :, None].repeat(1, 1, opt.graph_size)
    # 减去服务时间 duration 和行驶时间
    savings_time = savings_time - input_['duration'][:, :, None].repeat(1, 1, opt.graph_size) - i_j_distance / SPEED

    # 6. 综合计算总节省 (Savings)
    # 将距离节省和时间节省加权组合。值越大，合并越有利。
    savings = savings_distance - 0.03 * 60 * savings_time


    ## 构建路径

    # 1. 选择第一个要访问的客户点
    # 策略：选择右时间窗口最小的客户点作为起始。
    mask = (tw_right != 0)  # 过滤掉右时间窗为0的客户点
    if mask.sum(dim=1).min().item() == 0:
        # 如果有一行全为0（即某个样本没有可选点），跳过这一步
        pass
    else:
        # 方法：将0替换为一个很大的值，这样排序后0会排在最后
        tw_right_masked = tw_right.clone()
        tw_right_masked[~mask] = float('inf')  # 使用 ~ 取反掩码，找出零值位置，并替换为一个很大的数，如1e9

        _, selected = tw_right_masked.sort(dim=1)  # 对时间窗右边界排序
        selected = 1 + selected[:, 0]  # 获取排序后第一个客户点（索引从1开始）

        state = state.update(selected)  # 更新状态，加入选定节点
        sequences.append(selected)  # 记录节点

    # 2. 调度后续节点，直到所有节点都已访问
    i = 0
    while not (state.all_finished()):
        mask = state.get_mask() # 获取当前可访问节点的掩码（包括时间窗约束）
        prev = state.prev_a - 1 # 当前路径中最后一个访问的节点

        # 获取基于 prev 节点，与其他未访问节点的节省分数
        score = savings[ids, prev, :][:, 0, :]

        # 处理中心点 (Depot) 的分数
        depot_score, _ = score.min(dim=1) # 取出最小的距离节省分数
        depot_score = depot_score[:, None] - 1 # 回到仓库的惩罚分数
        score = torch.cat((depot_score, score), dim=1) # 将depot_score, score拼接起来，得到
        # print(f"score: {score[:4]},shape: {score.shape}")
        score[mask[:, 0, :]] = -math.inf # 应用掩码：不可访问节点分数设为负无穷
        # print(f"掩码过后score: {score[:4]},shape: {score.shape}")


        # 选择下一个要访问的节点：选择节省分数最高的节点
        _, selected = score.sort(descending=True)
        # print(f"排序过后: {selected},shape: {selected.shape}")
        selected = selected[:, 0]
        # print(f"selected: {selected},shape: {selected.shape}")

        state = state.update(selected) # 更新状态
        sequences.append(selected) # 记录节点
        i += 1

    # --- 计算最终成本 ---
    cost, _ = problem.get_costs(input_, torch.stack(sequences, 1)) # 计算最终路径的总成本

    return cost, state.serve_time # 返回总成本和每个节点的实际服务时间


def nearest_neighbor(input, problem, return_state=False):
    state = problem.make_state(input)
    sequences = []
    while not state.all_finished():
        mask = state.get_mask()
        mask = mask[:, 0, :]
        batch_size, n_loc = mask.size()
        prev_a = state.coords[state.ids, state.prev_a]  # [batch_size, 1]
        distance_index = state.NODE_SIZE * prev_a.expand(batch_size, n_loc) + state.coords  # [batch_size, n_loc]
        distance = torch.gather(input["distance"], 1, distance_index)  # [batch_size, n_loc]
        distance[mask] = 10000
        _, selected = distance.min(1)
        state = state.update(selected)
        sequences.append(selected)

    pi = torch.stack(sequences, 1)
    cost, _ = problem.get_costs(input, pi)

    if return_state:
        return cost, state
    else:
        return cost, state.serve_time


def check_insert(start, selected, tour, start_state, tmp_state_list):
    """
    检查一个 'selected' 节点是否可以插入到 'tour' 中的 'start' 位置，
    以及在此插入后，tour 中后续节点是否仍然有效。
    """
    # print(f"尝试将节点 {selected.item()} 插入到位置 {start}")

    # 从初始状态获取 mask，并将其展平为 1D
    mask = start_state.get_mask().squeeze(1)

    # 检查 'selected' 节点本身是否被 mask (例如，已经访问过或无效)
    if mask[0, selected] == 1:
        # print(f"节点 {selected.item()} 已被 mask，插入失败")
        return False, None

    # 如果没有被 mask，则通过添加 selected 节点来更新 start_state
    start_state = start_state.update(selected)
    # 将更新后的状态添加到临时状态列表中
    tmp_state_list.append(copy.deepcopy(start_state))

    # 在插入 'selected' 节点后，遍历原始 tour 的其余部分
    # 从插入点开始，以确保它们仍然有效。
    for s in range(start + 1, len(tour)):
        # 获取原始 tour 中的下一个节点
        selected = torch.LongTensor([tour[s]])
        # 获取当前状态的 mask
        mask = start_state.get_mask().squeeze(1)

        # 检查此后续节点是否现在被 mask (这意味着由于之前的插入而变得无效)
        if mask[0, selected] == 1:
            # print(f"后续节点 {selected.item()} 已被 mask，插入失败")
            return False, None

        # 如果后续节点有效，则更新状态并添加到临时状态列表
        start_state = start_state.update(selected)
        tmp_state_list.append(copy.deepcopy(start_state))

    # 如果所有检查都通过，则插入成功
    # print(f"在位置 {start} 插入成功")
    return True, start_state





def single_insert(i, input, problem, opt):
    """
    使用单点插入启发式算法为单个实例构建旅行路线。
    """
    # 提取距离矩阵，并将其重塑为 1xN*N 的张量
    distance = input["distance"][0].view(1, -1)  # [1, 92*92]

    # 获取批次大小和地点数量 (地点数量 = 原始地点 + 1 个仓库)
    batch_size, n_loc = input['loc'].size(0), input['loc'].size(1) + 1

    # 为当前单个实例准备输入数据
    single_input = {'loc': input['loc'][i:i + 1],
                    'distance': input['distance'][i:i + 1],
                    'duration': input['duration'][i:i + 1],
                    'tw_right': input['tw_right'][i:i + 1],
                    'tw_left': input['tw_left'][i:i + 1],
                    'fleet': input['fleet'][i:i + 1],
                    'need': input['need'][i:i + 1],
                    }

    # 从输入数据创建初始问题状态
    state = problem.make_state(single_input)
    # 初始化状态列表，用于跟踪每个步骤的状态
    state_list = [copy.deepcopy(state)]

    # 循环直到所有节点都被访问 (即路线完成)
    while not state.all_finished():
        mask = state.get_mask().squeeze(1)  # [1, n_loc]

        steps = state.tour.size(1) # 当前路径的长度，也代表当前在哪一步

        # 计算从当前路线中的每个节点到所有其他节点的距离索引
        # 这是为了从扁平化的距离矩阵中高效地查找距离
        distance_index = state.NODE_SIZE * state.coords[0, state.tour].permute(1, 0).repeat(1, n_loc) + state.coords.repeat(steps, 1)  # [steps, n_loc]
        # 根据 distance_index 从距离矩阵中收集实际距离
        d = torch.gather(distance.repeat(steps, 1), 1, distance_index)  # [steps, n_loc]
        # d[s, l] 中的值就是从当前路径中第 s 个节点到所有其他节点 l 的实际距离
        # 将 mask 复制到与距离矩阵相同的维度
        mask_ = mask.repeat(steps, 1)

        # 根据不同的验证方法选择下一个要插入的节点
        if opt.val_method == "nearest_insert":
            # 对于最近插入，将仓库 (索引 0) 的距离设置为一个大值以避免选择它，
            # 并将已 mask 的节点的距离设置为更大的值
            d[:, 0] = 9999  # 仓库惩罚
            d[mask_] = 10000
            # 找到距离最小的节点
            selected = d.argmin() - torch.div(d.argmin(), n_loc, rounding_mode="floor") * n_loc
        elif opt.val_method == "farthest_insert":
            # 对于最远插入，将仓库的距离设置为一个小值，已 mask 节点为负大值
            d[:, 0] = 1  # 仓库惩罚
            d[mask_] = -10000
            # 找到距离最大的节点
            selected = d.argmax() - torch.div(d.argmax(), n_loc, rounding_mode="floor") * n_loc
        elif opt.val_method == "random_insert":
            # 随机插入，选择一个未被 mask 的随机节点
            ids = torch.arange(n_loc)
            selected = ids[(mask == 0).view(-1)]
            random_id = random.randint(0, selected.size(0) - 1)
            selected = selected[random_id]
        # 将 selected 节点重塑为 [1]
        selected = selected.view(-1)  # [1]

        # 将选定的节点插入到合适的位置
        # 如果当前位于仓库 (state.prev_a == 0) 或者选定的节点是仓库 (selected.item() == 0)，
        # 则直接添加到路线末尾
        if state.prev_a.view(-1) == 0 or selected.item() == 0:  # 添加到末尾
            state = state.update(selected)
            state_list.append(copy.deepcopy(state))
        else:
            # 否则，计算将 selected 节点插入到当前路线各个位置的成本增量
            dd, tour = {}, state.tour[0].tolist()
            for j in range(len(tour) - 1):  # 遍历所有可能的插入位置 (例如 0->1, 1->2, ...)
                # 获取旧的两个节点和新的要插入的节点
                old1, old2, new = state.coords[0, tour[j]].item(), state.coords[0, tour[j + 1]].item(), state.coords[
                    0, selected.item()].item()
                # 计算插入新节点后的成本变化：(old1到new) + (new到old2) - (old1到old2)
                dd[j] = distance[0][state.NODE_SIZE * old1 + new] + distance[0][state.NODE_SIZE * new + old2] - \
                        distance[0][state.NODE_SIZE * old1 + old2]

            # 根据成本增量对可能的插入位置进行排序 (升序)
            sorted_dd = sorted(dd.items(), key=lambda item: item[1])

            # 尝试将节点插入到成本增量最小的位置
            inserted = False  # 标志，判断是否成功插入
            for j in range(len(sorted_dd)):
                # 确定当前的插入起始位置
                start = sorted_dd[j][0]
                # 深拷贝状态列表直到插入点，用于 check_insert 函数
                tmp_state_list = copy.deepcopy(state_list[:start + 1])
                # 获取插入点之前的状态
                start_state = tmp_state_list[start]
                # 调用 check_insert 检查是否可以插入
                insert, tmp_state = check_insert(start, selected, tour, start_state, tmp_state_list)
                if insert:
                    # 如果成功插入，更新当前状态和状态列表，并跳出循环
                    # print("成功插入到 {}".format(j))
                    state, state_list = tmp_state, tmp_state_list
                    inserted = True
                    break

            # 如果遍历了所有可能的插入位置都未能成功插入 (例如因为时间窗或容量限制)
            # 则将节点添加到路线的末尾 (作为备用策略)
            if not inserted:  # 或者使用原始的 if j == len(sorted_dd) - 1:
                state = state.update(selected)
                state_list.append(copy.deepcopy(state))

    # 所有节点都已插入后，将路线返回到仓库 (索引 0)
    selected = torch.LongTensor([0])  # 返回仓库
    state = state.update(selected)

    # 返回最终状态和实例索引
    return state, i

def insertion(input, problem, opt):
    """
    对一个批次的输入问题执行插入策略，计算每条路径的成本和完成时间。
    """
    res_list = [] # 用于存储多进程模式下每个任务的结果（异步对象）

    batch_size, n_loc = input['loc'].size(0), input['loc'].size(1) + 1

    cost, serve_time = torch.zeros(batch_size), torch.zeros(batch_size, n_loc)

    # 提示用户如果遇到多进程错误，可以尝试调整文件描述符限制
    # 这是 Linux 系统上的一个常见问题，当进程创建太多文件描述符时可能发生
    # 'ulimit -n 10240' 增加文件描述符限制
    # if multiprocessing error, use command: ulimit -n 10240

    # 判断是否使用多进程并行处理
    if opt.multiprocess:
        print(">> Val using multiprocessing")
        from multiprocessing import Pool # 确保这里导入了 Pool

        # 创建一个进程池，进程数量为 50
        pool = Pool(processes=50)
        # 遍历批次中的每一个问题实例
        for i in range(batch_size):
            # 异步地将单个插入任务提交到进程池
            # single_insert 是一个函数，用于处理单个问题实例的插入逻辑
            # args 是传递给 single_insert 函数的参数
            res = pool.apply_async(single_insert, args=(i, input, problem, opt))
            res_list.append(res) # 将异步结果对象添加到列表中

        pool.close() # 关闭进程池，不再接受新的任务
        pool.join()  # 等待所有子进程完成其工作

        # 遍历所有异步结果，获取每个任务的计算结果
        for r in res_list:
            # r.get() 会阻塞直到对应的任务完成，并返回结果
            # state 通常是一个包含路径信息、长度和时间的对象
            # i 是原始问题的索引
            state, i = r.get()
            # 从 state 中获取路径长度（成本）和总服务时间，并赋值给对应的张量位置
            # .view(-1) 将张量展平为一维，确保维度匹配
            cost[i], serve_time[i] = state.lengths.view(-1), state.serve_time.view(-1)
    else:
        # 不使用多进程，串行处理每个问题实例
        for i in range(batch_size):
            # 直接调用 single_insert 函数处理单个问题实例
            state, _ = single_insert(i, input, problem, opt)
            # print(state.tour) # 调试用，可能打印计算出的路径
            # 从 state 中获取路径长度（成本）和总服务时间，并赋值给对应的张量位置
            cost[i], serve_time[i] = state.lengths.view(-1), state.serve_time.view(-1)

    # 返回计算出的所有问题的成本和总服务时间
    return cost, serve_time


def stochastic_2_swap(input, problem, cur_tour):
    """
        For simulated_annealing to find a neighborhood of current solution,
        may not be feasible solution after swapping.
    """
    state = problem.make_state(input)
    feasibility = torch.zeros(cur_tour.size(0), dtype=torch.uint8, device=input["loc"].device)  # [batch_size]
    ids = torch.arange(cur_tour.size(0), device=input["loc"].device)
    left, right = random.randint(1, cur_tour.size(-1)-1), random.randint(1, cur_tour.size(-1)-1)
    if left > right:
        left, right = right, left
    start = 1
    while start < cur_tour.size(-1):
        if start == left:  # [left, right]
            selected = cur_tour[:, right]  # [batch_size]
        elif start == right:
            selected = cur_tour[:, left]
        else:
            selected = cur_tour[:, start]
        mask = state.get_mask()
        mask = mask[:, 0, :]  # [batch_size, n_loc+1]
        feasibility = feasibility | mask[ids, selected]
        state = state.update(selected)
        start += 1
    selected = torch.LongTensor([0] * cur_tour.size(0)).to(input["loc"].device)  # return to depot
    state = state.update(selected)
    return state, feasibility


def pad_tour(tour1, tour2):
    while True:
        if (tour1[-1] == 0).sum() == tour1.size(0):
            tour1 = tour1[:, :-1]
        elif (tour2[-1] == 0).sum() == tour2.size(0):
            tour2 = tour2[:, :-1]
        else:
            break
    if tour1.size(-1) > tour2.size(-1):
        tour2 = F.pad(tour2, (0, tour1.size(-1)-tour2.size(-1)), "constant", 0)
    elif tour1.size(-1) < tour2.size(-1):
        tour1 = F.pad(tour1, (0, tour2.size(-1) - tour1.size(-1)), "constant", 0)
    return tour1, tour2


def simulated_annealing(input, problem):
    count, start_t = 0, time.time()
    time_limit = 1800
    neighbourhood_size, iterations, T = 110, 20, 10
    cost, state = nearest_neighbor(input, problem, return_state=True)


    print(">> init sol: {}".format(cost.mean()))
    serve_time, tour = state.serve_time, state.tour  # [batch_size, -1]
    cur_sol_cost, cur_sol_serve_time, cur_sol_tour = cost.detach().clone(), serve_time.detach().clone(), tour.detach().clone()
    best_sol_cost, best_sol_serve_time, best_sol_tour = cost.detach().clone(), serve_time.detach().clone(), tour.detach().clone()
    while (count < iterations):
        if time.time() - start_t > time_limit:
            break
        for i in range(0, neighbourhood_size):
            # preprocess cur_sol_tour
            while True:
                # 空路径保护：若当前 tour 为空则直接退出预处理
                if cur_sol_tour.size(1) == 0:
                    break
                if (cur_sol_tour[:, -1] == 0).sum() == cur_sol_tour.size(0):
                    cur_sol_tour = cur_sol_tour[:, :-1]
                else:
                    break
            # 若此时 tour 为空，则跳过本次邻域尝试
            if cur_sol_tour.size(1) == 0:
                continue
            state, feasibility = stochastic_2_swap(input, problem, cur_sol_tour)
            new_sol_cost, new_sol_serve_time, new_sol_tour = state.lengths.view(-1), state.serve_time, state.tour
            delta_cost = new_sol_cost - cur_sol_cost
            # print(feasibility)
            ran_accept = np.random.uniform(0, 1, feasibility.size(0))
            criteria = np.e ** (-delta_cost / T)
            accept_id = (feasibility == 0) & ((delta_cost < 0) | (torch.Tensor(ran_accept).to(input["loc"].device) <= criteria))
            best_id = accept_id & (new_sol_cost < best_sol_cost)
            # print(accept_id, best_id)
            cur_sol_tour, new_sol_tour = pad_tour(cur_sol_tour, new_sol_tour)
            cur_sol_cost[accept_id], cur_sol_serve_time[accept_id], cur_sol_tour[accept_id] = new_sol_cost[accept_id], new_sol_serve_time[accept_id], new_sol_tour[accept_id]
            best_sol_tour, new_sol_tour = pad_tour(best_sol_tour, new_sol_tour)
            best_sol_cost[best_id], best_sol_serve_time[best_id], best_sol_tour[best_id] = new_sol_cost[best_id], new_sol_serve_time[best_id], new_sol_tour[best_id]
        count = count + 1
        T = T * 0.9

    print(">> After SA sol: {}".format(best_sol_cost.mean()))

    return best_sol_cost, best_sol_serve_time


def val(dataset, opt, fleet_info, distance, problem):
    cost = []
    for bat in tqdm(DataLoader(dataset, batch_size=50, shuffle=False), disable=opt.no_progress_bar):
        bat_cost = []
        bat_tw_left = bat['arrival'].repeat(len(fleet_info['next_duration']) + 1, 1, 1).to(opts.device)  # [6, batch_size, graph_size]
        bat_tw_right = bat['departure']  # [batch_size, graph_size]
        need = bat['need']

        for f in fleet_info['order']:
            # merge more data
            next_duration = torch.tensor(fleet_info['next_duration'][fleet_info['precedence'][f]],
                                         device=bat['type'].device).repeat(bat['loc'].size(0), 1)  # [batch_size, 3]
            tw_right = bat_tw_right - torch.gather(next_duration, 1, bat['type'])
            tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)  # [batch_size, graph_size+1]
            tw_left = bat_tw_left[fleet_info['precedence'][f]]
            tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)  # [batch_size, graph_size+1]
            duration = torch.tensor(fleet_info['duration'][f], device=bat['type'].device).repeat(bat['loc'].size(0), 1)  # [batch_size, 3]

            if f == 1:
                mask = (need == 1) | (need == 9)  # 1单 + 1,6组合
            elif f == 2:
                mask = (need == 2) | (need == 7)  # 2单 + 2,3组合
            elif f == 3:
                mask = (need == 3) | (need == 7)  # 3单 + 2,3组合 #
            elif f == 4:
                mask = (need == 4) | (need == 8)  # 4单 + 4,5组合
            elif f == 5:
                mask = (need == 5) | (need == 8)  # 5单 + 4,5组合 #
            elif f == 6:
                mask = (need == 6) | (need == 9)  # 6单 + 1,6组合 ##
            else:
                mask = (need == f)  # 其他预留

            # 滤掉登机口节点
            tw_right_filtered = tw_right.clone()
            tw_right_filtered[:, 1:] = tw_right[:, 1:] * mask.type_as(tw_right).float()

            tw_left_filtered = tw_left.clone()
            tw_left_filtered[:, 1:] = tw_left[:, 1:] * mask.type_as(tw_left).float()

            # need掩码
            need_filtered = need.clone()
            need_filtered = need_filtered * mask.type_as(need).float()

            fleet_bat = {'loc': bat['loc'],
                         'distance': distance.expand(bat['loc'].size(0), len(distance)),
                         'duration': torch.gather(duration, 1, bat['type']),
                         'tw_right': tw_right_filtered,
                         'tw_left': tw_left_filtered,
                         'fleet': torch.full((bat['loc'].size(0), 1), f - 1),
                         'need': need_filtered,
                         }

            if opt.val_method == "cws":
                fleet_cost, serve_time = cws(move_to(fleet_bat, opt.device), problem, opt)
            elif opt.val_method == "nearest_neighbor":
                fleet_cost, serve_time = nearest_neighbor(move_to(fleet_bat, opt.device), problem)
            elif opt.val_method in ["nearest_insert", "farthest_insert", "random_insert"]:
                fleet_cost, serve_time = insertion(move_to(fleet_bat, opt.device), problem, opt)
            elif opt.val_method == "sa":
                c = 2  # 可调：每段固定增量（与距离同单位）
                n = int(distance.size(0) ** 0.5)
                dist2d = distance.view(n, n).clone().to(torch.float32)
                diag_mask = torch.eye(n, dtype=torch.bool, device=dist2d.device)
                dist2d = dist2d + (~diag_mask).float() * c
                distance_sa = dist2d.reshape(-1)

                fleet_bat_sa = dict(fleet_bat)
                fleet_bat_sa['distance'] = distance_sa.expand(bat['loc'].size(0), len(distance_sa))
                fleet_cost, serve_time = simulated_annealing(move_to(fleet_bat_sa, opt.device), problem)
            else:
                print(">> Unsupported val method!")
                return 0

            bat_cost.append(fleet_cost.data.cpu().view(-1, 1))


            next_stage = fleet_info['precedence'][f] + 1
            mask = mask.to(opts.device)  # [batch_size, graph_size]
            if f == 1:
                # f=1 时的特殊逻辑：不加10
                bat_tw_left[next_stage] = torch.where(mask, serve_time[:, 1:], bat_tw_left[next_stage])
            else:
                # 其他情况的原有逻辑：加10
                bat_tw_left[next_stage] = torch.where(mask, serve_time[:, 1:] + 10, bat_tw_left[next_stage])

        bat_cost = torch.cat(bat_cost, 1)
        cost.append(bat_cost)  # [batch_size, 10]

    cost = torch.cat(cost, 0)  # [dataset, 10]
    cost = cost.sum(1)
    print(cost.tolist())
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(avg_cost, torch.std(cost) / math.sqrt(len(cost))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="./data/agh/agh300_validation_seed4321.pkl", help="Filename of the dataset to load")
    parser.add_argument("--problem", type=str, default='agh', help="only support airport ground handling in this code")
    parser.add_argument('--graph_size', type=int, default=300, help="Sizes of problem instances (20, 50, 100)")
    parser.add_argument('--val_method', type=str, default='sa', choices=['cws', 'nearest_insert', 'farthest_insert',
                                                                          'random_insert', 'nearest_neighbor','sa'])
    parser.add_argument('--val_size', type=int, default=1, help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0, help='Offset where to start in dataset (default 0)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--multiprocess', action='store_true', help='Using multiprocessing module')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    opts = parser.parse_args()

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda and \
                    opts.val_method not in ["nearest_insert", "farthest_insert", "random_insert"]
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    print(opts.device)

    pp.pprint(vars(opts))

    # Set the random seed
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    # Figure out what's the problem
    problem_ = load_problem(opts.problem)

    val_dataset = problem_.make_dataset(filename=opts.filename, num_samples=opts.val_size, offset=opts.offset)

    with open('problems/agh/fleet_info.pkl', 'rb') as file_:
        fleet_info_ = pickle.load(file_)
    with open('problems/agh/distance.pkl', 'rb') as file_:
        distance_dict = pickle.load(file_)
    distance_ = torch.tensor(list(distance_dict.values()))

    print('Validating dataset: {}'.format(opts.filename))
    start_time = time.time()
    val(val_dataset, opts, fleet_info_, distance_, problem_)
    print(">> End of validation within {:.2f}s".format(time.time()-start_time))

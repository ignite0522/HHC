import torch
import pickle
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter

# 定义 StateAGH 类，使用 NamedTuple 存储 AGH 问题的状态
class StateAGH(NamedTuple):
    # 固定输入属性（来自问题实例的不可变数据）
    coords: torch.Tensor  # [batch_size, graph_size+1, 2]，车库（节点 0）+登机口的位置坐标
    tw_left: torch.Tensor  # [batch_size, graph_size+1]，时间窗口左边界（最早服务时间）
    tw_right: torch.Tensor  # [batch_size, graph_size+1]，时间窗口右边界（最晚服务时间）
    distance: torch.Tensor  # [batch_size, graph_size+1]，节点间距离矩阵
    duration: torch.Tensor  # [batch_size, graph_size+1]，每个节点的服务时长（车库为 0）
    need: torch.Tensor
    # 状态跟踪索引，用于支持束搜索（beam search）等场景，避免重复存储 coords 和 demand
    ids: torch.Tensor  # [batch_size, 1]，跟踪原始批次索引，用于索引固定数据行

    # 动态状态属性（随路径生成更新）
    prev_a: torch.Tensor  # [batch_size, 1]，上一个访问的节点
    # used_capacity: torch.Tensor  # [batch_size, 1]，已使用的车辆容量
    visited_: torch.Tensor  # [batch_size, 1, graph_size+1]，已访问节点掩码（uint8 或 int64）
    lengths: torch.Tensor  # [batch_size, 1]，当前路径的总距离
    cur_coord: torch.Tensor  # [batch_size, 1, 2]，当前节点的坐标
    cur_free_time: torch.Tensor  # [batch_size, 1]，当前可用时间（上一个节点服务完成后的时间）
    serve_time: torch.Tensor  # [batch_size, graph_size+1]，每个节点的服务开始时间
    tour: torch.Tensor  # [batch_size, steps]，当前路径（访问的节点序列）
    i: torch.Tensor  # [1]，当前步骤计数器
    fleet: torch.Tensor


    SPEED = 80.0  # 车辆速度，单位为分钟（用于将距离转换为旅行时间）
    NODE_SIZE = 92  # 节点总数（graph_size + 1，通常为 91 个登机口 + 1 个车库）

    @property
    def visited(self):
        # 返回已访问节点的布尔掩码，处理不同数据类型（uint8 或 int64）
        if self.visited_.dtype == torch.uint8:
            return self.visited_  # 如果是 uint8，直接返回 visited_（0=未访问，1=已访问）
        else:
            # 如果是 int64，使用 mask_long2bool 转换为布尔掩码，n 为节点数（demand 的最后一个维度）
            return mask_long2bool(self.visited_, n=self.duration.size(-1)-1)

    def __getitem__(self, key):
        # 支持通过索引（张量或切片）获取子状态，保持 NamedTuple 结构
        assert torch.is_tensor(key) or isinstance(key, slice)  # 确保 key 是张量或切片
        # 返回新的 StateAGH 实例，所有张量按 key 索引
        return self._replace(
            ids=self.ids[key],  # 索引批次 ID
            prev_a=self.prev_a[key],  # 索引上一个节点
            # used_capacity=self.used_capacity[key],  # 索引已使用容量
            visited_=self.visited_[key],  # 索引已访问掩码
            lengths=self.lengths[key],  # 索引路径长度
            cur_coord=self.cur_coord[key],  # 索引当前坐标
            cur_free_time=self.cur_free_time[key],  # 索引当前可用时间
            tour=self.tour[key],  # 索引路径序列
            serve_time=self.serve_time[key],  # 索引服务时间
            fleet=self.fleet[key],  # 索引车队标号
            need=self.need[key],  # 索引需求量
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        # 初始化 StateAGH 状态，基于输入数据创建初始状态
        loc = input['loc']  # [batch_size, graph_size, 2]，登机口坐标
        tw_left, tw_right = input['tw_left'], input['tw_right']  # [batch_size, graph_size+1]，时间窗口
        batch_distance = input['distance']  # [batch_size, graph_size+1]，距离矩阵
        fleet = input['fleet']  # [batch_size, 1]，车队标号
        need = input['need']  # [batch_size, graph_size+1]，需求量
        # print(f"fleet: {fleet[:4]},shape: {fleet.shape}")
        # print(f"need: {need[:4]},shape: {need.shape}")



        batch_size, n_loc = loc.size()  # batch_size 和登机口数（graph_size）
        return StateAGH(
            # 固定输入
            fleet=fleet,  # [batch_size, 1]，车队标号
            need=need,
            coords=torch.cat((torch.zeros_like(loc[:, :1], device=loc.device), loc), dim=1),  # [batch_size, graph_size+1, 2]，添加车库坐标 (0,0)
            tw_left=tw_left,  # [batch_size, graph_size+1]，时间窗口左边界
            tw_right=tw_right,  # [batch_size, graph_size+1]，时间窗口右边界
            distance=batch_distance,  # [batch_size, graph_size+1]，距离矩阵
            duration=torch.cat((torch.zeros_like(loc[:, :1], device=loc.device), input['duration']), dim=1),  # [batch_size, graph_size+1]，服务时长（车库为 0）
            # 状态初始化
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # [batch_size, 1]，批次索引
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),  # [batch_size, 1]，初始上一个节点为车库 (0)
            visited_=(
                # 初始化已访问掩码，uint8 更节省内存，int64 支持更多节点
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                ) if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # [batch_size, 1, ceil(n_loc/64)]，int64 掩码
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),  # [batch_size, 1]，初始路径长度为 0
            cur_coord=torch.zeros_like(loc[:, :1], device=loc.device),  # [batch_size, 1, 2]，初始坐标为车库 (0,0)
            cur_free_time=torch.full((batch_size, 1), -60, device=loc.device),  # [batch_size, 1]，初始可用时间为 -60（占位符，车库）
            serve_time=torch.zeros(batch_size, n_loc+1, device=loc.device).float(),  # [batch_size, graph_size+1]，初始服务时间为 0
            tour=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),  # [batch_size, 1]，初始路径为空
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # [1]，初始步骤为 0
        )

    def get_final_cost(self):
        # 获取最终路径成本（未实现，留待束搜索等未来扩展）
        # TODO: 结合束搜索，未来完善
        pass


    def update(self, selected):
        # 更新状态，基于选择的节点（selected）更新路径和状态
        assert self.i.size(0) == 1, "Can only update if state represents single step"  # 确保当前状态只表示单步

        # 更新状态
        selected = selected[:, None]  # [batch_size, 1]，扩展维度以匹配状态
        prev_a = selected  # [batch_size, 1]，更新上一个节点为当前选择
        n_loc = self.duration.size(-1)-1  # 登机口数（不包括车库）

        # 计算路径长度
        cur_coord = self.coords[self.ids, selected]  # [batch_size, 1, 2]，当前节点的坐标
        distance_index = self.NODE_SIZE * self.coords[self.ids, self.prev_a] + cur_coord  # [batch_size, 1]，距离矩阵索引
        # print(f"distance_index = {distance_index},shape: {distance_index.shape}")
        lengths = self.lengths + self.distance.gather(1, distance_index)  # [batch_size, 1]，累加路径距离
        #
        # # 更新已使用容量
        # selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]  # [batch_size, 1]，当前节点的需求（车库需求为 0）
        # used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()  # [batch_size, 1]，非车库时累加需求，车库时重置为 0

        # 计算车队在该节点完成服务的时间
        cur_free_time = (torch.max(
            self.cur_free_time + self.distance.gather(1, distance_index) / self.SPEED,  # 到达时间 = 上一个完成时间 + 旅行时间
            self.tw_left[self.ids, selected])  # 确保不小于时间窗口左边界
                         + self.duration[self.ids, selected]) * (prev_a != 0).float() - 60 * (prev_a == 0).float()  # [batch_size, 1]，非车库时加服务时长，车库时设为 -60

        # 计算车队在该节点开始服务的时间
        cur_start_time = torch.max(
            self.cur_free_time + self.distance.gather(1, distance_index) / self.SPEED,  # 到达时间 = 上一个完成时间 + 旅行时间
            self.tw_left[self.ids, selected])

        # 更新已访问掩码
        if self.visited_.dtype == torch.uint8:
            # 对于 uint8 掩码，直接使用 scatter 设置当前节点为已访问
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)  # [batch_size, 1, graph_size+1]
        else:
            # 对于 int64 掩码，使用 mask_long_scatter 设置（忽略车库，prev_a - 1）
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        # 更新服务时间
        serve_time = self.serve_time.scatter_(1, selected, cur_start_time.float())  # [batch_size, graph_size+1]，记录当前节点的服务时间

        # print(f"locations: {self.coords}")
        # print(f"update.tw_left: {self.tw_left[:4].int()},shape: {self.tw_left.shape}")
        # print(f"update.tw_right: {self.tw_right[:4].int()},shape: {self.tw_right.shape}")
        # print(f"sver_time: {serve_time[:4]},shape: {serve_time.shape}")
        # print(f"distance_time: {self.distance.gather(1, distance_index) / self.SPEED}")
        # print(f"duration: {self.duration[:4]},shape: {self.duration.shape}")
        # 更新路径序列
        tour = torch.cat((self.tour, selected), dim=1)  # [batch_size, steps+1]，添加当前节点到路径
        torch.set_printoptions(threshold=float('inf'), linewidth=1000)# 设置张量（Tensor）的打印选项，可有可无
        # print(f"tour: {tour},shape: {tour.shape}\n")
        # 返回更新后的 StateAGH 实例
        return self._replace(
            prev_a=prev_a,  # 更新上一个节点
            visited_=visited_,  # 更新已访问掩码
            lengths=lengths,  # 更新路径长度
            cur_coord=cur_coord,  # 更新当前坐标
            i=self.i + 1,  # 步骤计数器加 1
            cur_free_time=cur_free_time,  # 更新当前可用时间
            tour=tour,  # 更新路径序列
            serve_time=serve_time  # 更新服务时间
        )


    def all_finished(self):
        # 检查是否所有路径都已完成（所有登机口已访问且步骤数足够）
        mask = (self.tw_right[self.ids] != 0).int()
        flag = (mask==self.visited).all()  # [4]，True 表示该批次完成
        # print(f"ids: {self.ids},shape: {self.ids.shape}")
        # print(f"mask: {mask},shape: {mask.shape}")
        # print(f"visited:{self.visited},shape: {mask.shape}")
        # print(f"flag: {flag},shape: {flag.shape}")
        # print(f"flag: {flag}")
        return flag


    def get_finished(self):
        # 检查每个批次是否完成（所有节点已访问）
        return self.visited.sum(-1) == self.visited.size(-1)  # [batch_size, 1]，True 表示该批次完成

    def get_current_node(self):
        # 获取当前节点（上一个访问的节点）
        return self.prev_a  # [batch_size, 1]

    def get_mask(self):
        """
        生成掩码，标记不可行节点（0=可行，1=不可行）
        - 形状: [batch_size, 1, graph_size+1]
        - 考虑已访问节点、时间窗口约束
        - 禁止连续两次访问车库，除非所有登机口已访问
        """
        # 1.处理已访问节点掩码
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]  # [batch_size, 1, graph_size]，排除车库
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.duration.size(-1)-1)  # [batch_size, 1, graph_size]，转换为布尔掩码
        # print(f"visited_loc: {visited_loc[:4].int()},shape: {visited_loc.shape}")

        # 计算距离索引
        pre_coord = self.coords[self.ids, self.prev_a]  # [batch_size, 1, 2]，上一个节点坐标
        all_coord = self.coords[:, 1:]  # [batch_size, graph_size, 2]，所有登机口坐标
        distance_index = self.NODE_SIZE * pre_coord.expand(self.coords.size(0), self.coords.size(1)-1) + all_coord  # [batch_size, graph_size]，距离矩阵索引

        # 获取必要张量
        arrival_time = torch.max(
            self.cur_free_time.expand(self.coords.size(0), self.coords.size(1)-1) +  # 完成上一个服务后的时间
            self.distance.gather(1, distance_index) / self.SPEED,  # 加上旅行时间
            self.tw_left[:, 1:])
        tw_left_base = self.tw_left[:, 1:]
        tw_right_base = self.tw_right[:, 1:]
        duration_base = self.duration[:, 1:]

        # 初始化掩码
        mask_loc = torch.zeros_like(tw_left_base, dtype=torch.bool)[:, None, :]  # [batch_size, 1, graph_size]

        fleet_ids = self.fleet.squeeze(-1)+1  # [batch_size]
        # print(f"fleet_ids: {fleet_ids[:4]},shape: {fleet_ids.shape}")
        need_tensor = self.need  # [batch_size, graph_size+1]

        for fleet_type in [1, 2, 3, 4, 5, 6]:
            batch_mask = (fleet_ids == fleet_type)
            if not batch_mask.any():
                continue

            # 获取当前 fleet 下的 tw, duration
            tw_left = tw_left_base[batch_mask]
            tw_right = tw_right_base[batch_mask]
            duration = duration_base[batch_mask]
            arrival = arrival_time[batch_mask]
            need = need_tensor[batch_mask]
            # print(f"tw_left: {tw_left[:4]},shape: {tw_left.shape}")
            # print(f"tw_right: {tw_right[:4]},shape: {tw_right.shape}")
            # print(f"arrival: {arrival[:4]},shape: {arrival.shape}")
            # print(f"need: {need[:4]},shape: {need.shape}")

            # 默认 scenario1
            mask_fleet = (
                (arrival + duration > tw_right)[:, None, :]
            )

            if fleet_type in [3, 5]:
                # --- scenario1: need = 7 or 8 ---
                need_alt = need.clone()
                need_alt[(need_alt == (7 if fleet_type == 3 else 8))] = 0
                tw_left_alt = tw_left.clone()
                tw_right_alt = tw_right.clone()
                tw_left_alt[(need_alt == 0)] = 0
                tw_right_alt[(need_alt == 0)] = 0

                mask1 = (
                    (arrival + duration > tw_right_alt)[:, None, :]
                )


                # --- scenario2: need = 3 or 5 ---
                need_alt = need.clone()
                need_alt[(need_alt == (3 if fleet_type == 3 else 5))] = 0
                tw_left_alt = tw_left.clone()

                tw_right_alt = tw_right.clone()
                tw_left_alt[(need_alt == 0)] = 0
                tw_right_alt[(need_alt == 0)] = 0

                lower_bound = tw_left_alt
                upper_bound = tw_left_alt + 30
                adjusted_arrival = torch.max(arrival, lower_bound)
                mask2 = ((arrival > upper_bound)[:, None, :] |
                         ((adjusted_arrival + duration) > tw_right_alt)[:, None, :])

                mask_fleet = mask1 & mask2



            elif fleet_type == 6:
                # --- scenario1: need = 9 ---
                need_alt = need.clone()
                need_alt[need_alt == 9] = 0
                tw_left_alt = tw_left.clone()
                tw_right_alt = tw_right.clone()
                tw_left_alt[need_alt == 0] = 0
                tw_right_alt[need_alt == 0] = 0

                mask1 = ((arrival + duration > tw_right_alt)[:, None, :])

                # --- scenario3: need = 6 ---
                need_alt = need.clone()
                need_alt[need_alt == 6] = 0
                tw_left_alt = tw_left.clone()
                tw_right_alt = tw_right.clone()
                duration_alt = duration.clone()
                tw_left_alt[need_alt == 0] = 0
                tw_right_alt[need_alt == 0] = 0

                adjusted_arrival = torch.max(arrival, tw_left_alt)
                mask_condition1 = (arrival > tw_left_alt)[:, None, :]
                mask_condition2 = ((adjusted_arrival + duration_alt) > tw_right_alt)[:, None, :]
                mask2 = mask_condition1 | mask_condition2

                mask_fleet = mask1 & mask2


            elif fleet_type in [1, 2, 4]:
                # fleet 1、2、4 直接使用场景一
                mask_fleet = ((arrival + duration > tw_right)[:, None, :])

            # 填入总掩码
            mask_loc[batch_mask] = mask_fleet

        # 总掩码 = 已访问 或 时间窗口不可行
        mask_loc = visited_loc.to(mask_loc.dtype) | mask_loc  # [batch_size, 1, graph_size]

        # 3.处理车库约束
        # 车库掩码：如果刚访问车库且仍有未访问登机口
        # 禁止再次访问车库，当刚访问车库（prev_a == 0）且仍有可访问登机口（mask_loc == 0 的节点数 > 0）时，车库被掩码
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)  # [batch_size, 1]
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)  # [batch_size, 1, graph_size+1]，合并车库和登机口掩码

    def construct_solutions(self, actions):
        # 构建完整路径（未实现，留待束搜索等未来扩展）
        # TODO: 结合束搜索，未来完善
        pass
import pickle
# #
# #
# # fleet_info= {'order': [1, 2, 3, 4],
# #              'precedence': {1: 0, 2: 1, 3: 1, 4: 2},
# #              'duration': {1: [30.0, 30.0, 30.0], 2: [36.0, 35.0, 35.0], 3: [35.0, 35.0, 36.0], 4: [30.0, 29.0, 27.0]},
# #              'next_duration': {2: [0.0, 0.0, 0.0],   1: [45.0, 44.0, 42.0], 0: [85.0, 87.0, 84.0]}}
#
# # fleet_info= {'order': [1, 2, 3, 4],
# #              'precedence': {1: 0, 2: 1, 3: 2, 4: 0},
# #              'duration': {1: [30.0, 30.0, 30.0], 2: [36.0, 35.0, 35.0], 3: [35.0, 35.0, 36.0], 4: [30.0, 29.0, 27.0]},
# #              'next_duration': {2: [0.0, 0.0, 0.0],   1: [61.0, 60.0, 60.0], 0: [0.0, 0.0, 0.0]}}
#
#
# #
fleet_info= {'order': [1, 2, 3, 4, 5, 6],
             'precedence': {1: 2, 2: 0, 3: 1, 4: 0, 5: 1, 6: 3},
             'duration': {1: [30.0, 30.0, 30.0], 2: [30.0, 31.0, 31.0], 3: [34.0, 33.0, 32.0], 4: [29.0, 31.0, 30.0], 5:[32.0, 31.0, 30.0], 6: [30.0, 30.0, 30.0]},
             'next_duration': {   3: [0.0, 0.0, 0.0],   2: [0.0, 0.0, 0.0],   1: [0.0, 0.0, 0.0],   0: [60.0, 60.0, 60.0],}}

# #0-45     30-92
#
# 保存到 pickle 文件
with open('fleet_info.pkl', 'wb') as f:
    pickle.dump(fleet_info, f)

with open('fleet_info.pkl', 'rb') as f:
    loaded_fleet_info = pickle.load(f)
    print('fleet_info:', loaded_fleet_info)


# import numpy as np
# # 原始概率数组
# raw_prob = np.array([
#     0.05479452, 0.08219178, 0.05479452, 0.03191781,
#     0.04657534, 0.05479452, 0.03835616, 0.06027397,
#     0.03835616, 0.04109589
# ])

# # 归一化处理
# arrival_prob = raw_prob / raw_prob.sum()
#
#
# np.save('arrival_prob.npy', arrival_prob)
#
# # 可选：验证保存结果
# loaded = np.load('arrival_prob.npy')
# print("归一化后并保存的 arrival_prob:", loaded)
# print("和是否为 1:", loaded.sum())

# import numpy as np
#
# # 论文参数
# mean = 0
# variance = 1
# std_dev = np.sqrt(variance)
#
# # 限定小时范围在10到19之间（共10个小时）
# n_hour = np.arange(10, 20)
#
# # 使用高斯分布的概率密度函数（PDF）计算每个小时的相对概率
# # 将峰值设置在小时范围的中心，例如第15小时
# peak_mean = 15
# peak_std = std_dev
#
# # 计算每个小时的相对概率
# raw_prob_gaussian = np.exp(-((n_hour - peak_mean)**2) / (2 * peak_std**2))
#
# # 归一化处理，使其总和为1
# arrival_prob_gaussian = raw_prob_gaussian / raw_prob_gaussian.sum()
#
# # 保存为 .npy 文件
# np.save('arrival_prob_gaussian_subset.npy', arrival_prob_gaussian)
#
# # 验证保存结果
# loaded_gaussian = np.load('arrival_prob_gaussian_subset.npy')
# print("高斯分布生成的 arrival_prob_gaussian (10-19点):", loaded_gaussian)
# print("和是否为 1:", loaded_gaussian.sum())
#
#
#
# import numpy as np
#
# # 论文参数
# lambda_param = 4  # 期望事件数，即平均每小时有4个航班
#
# # 限定小时范围在10到19之间
# n_hour = np.arange(10, 20)
# num_hours_in_subset = len(n_hour)
#
# # 使用泊松分布生成每个小时内的航班数量，模拟一段时间的观测
# num_days = 30  # 模拟30天的数据
# flights_per_hour = np.random.poisson(lam=lambda_param, size=(num_days, num_hours_in_subset))
#
# # 计算每个小时的平均航班数量，作为相对概率
# raw_prob_poisson = np.mean(flights_per_hour, axis=0)
#
# # 归一化处理，使其总和为1
# arrival_prob_poisson = raw_prob_poisson / raw_prob_poisson.sum()
#
# # 保存为 .npy 文件
# np.save('arrival_prob_poisson_subset.npy', arrival_prob_poisson)
#
# # 验证保存结果
# loaded_poisson = np.load('arrival_prob_poisson_subset.npy')
# print("泊松分布生成的 arrival_prob_poisson (10-19点):", loaded_poisson)
# print("和是否为 1:", loaded_poisson.sum())
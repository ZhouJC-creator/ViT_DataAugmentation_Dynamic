import torch

#
# select_rate = torch.tensor([16, 14, 12, 10, 8, 6, 8, 10, 12, 14, 16, 19])
#
# score = torch.randn((3, 12, 50))
# heads_select_index = [[] for _ in range(3)]
# for b in range(3):
#     for i in range(12):
#         _, select_index = torch.topk(score[b, i, :], select_rate[i], dim=-1)
#         heads_select_index[b].extend(select_index.unsqueeze(0))
#
# count = torch.zeros((3, 50))
# print(len(heads_select_index))
#
# for i in range(len(heads_select_index)):
#     for j, b in enumerate(heads_select_index[i]):
#         # torch.bincount: 得到输入张量中对应索引值出现的次数, 这里就是12个头
#         # count： (32, 784) 每个token对应的分数
#         scores = torch.bincount(b, minlength=50)
#         count[i, :] += scores
#
# print(count.shape)


# import torch
#
# heads_select_index = [
#     torch.tensor([0, 0, 0]),
#     torch.tensor([1, 2, 3]),
#     torch.tensor([0, 2, 4]),
#     torch.tensor([5, 6]),
#     torch.tensor([7, 8]),
#     torch.tensor([9, 1]),
#     torch.tensor([3, 4]),
#     torch.tensor([5, 2, 8]),
#     torch.tensor([6, 9, 7]),
#     torch.tensor([3]),
#     torch.tensor([8, 9]),
#     torch.tensor([0, 6])
# ]
#
# last_select_idx = torch.tensor([0, 3, 7, 9])
#
# # 计算重复元素的数量
# result = torch.zeros(len(heads_select_index), dtype=torch.int)
#
# for i, select_indices in enumerate(heads_select_index):
#     # 使用 isin 判断 select_indices 中的每个元素是否在 last_select_idx 中
#     result[i] = torch.sum(torch.isin(select_indices, last_select_idx))
#
# print(result)
B = 2
heads_num = 3
select_num = 10
heads_select_index = torch.randint(0, 100, (B, heads_num, select_num))
print(heads_select_index)
last_select_idx = torch.randint(0,100,(B,select_num))
print(last_select_idx)
alive_num = torch.zeros((B, heads_num))
for b in range(B):
    for i, select_indices in enumerate(heads_select_index[b]):
        alive_num[b][i] = torch.sum(torch.isin(select_indices, last_select_idx[b]))
print(alive_num)
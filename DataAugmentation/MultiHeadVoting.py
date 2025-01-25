import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from torch.nn import Softmax


class MultiHeadVoting(nn.Module):
    def __init__(self, vote_perhead=128, fix=True):
        super(MultiHeadVoting, self).__init__()
        self.fix = fix
        self.num_heads = 12
        self.vote_perhead = vote_perhead
        self.softmax = Softmax(dim=-1)

        if self.fix:
            self.kernel = torch.tensor([[1, 2, 1],
                                        [2, 4, 2],
                                        [1, 2, 1]], device='cuda').unsqueeze(0).unsqueeze(0).half()
            self.conv = F.conv2d
        else:
            self.conv = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x, heads_select_rate, total_num, select_num=None, last=True):  # (32, 12, 785, 785)
        # 32, 784
        B, heads_num, patch_num = x.shape[0], x.shape[1], x.shape[3] - 1
        # 16
        select_num = self.vote_perhead if select_num is None else select_num
        # (32, 784)
        count = torch.zeros((B, patch_num), dtype=torch.int, device='cuda').half()

        ########
        # 使用FFVT中a0*b0的方法
        ########

        a0 = x[:, :, 0, :]
        b0 = x[:, :, :, 0]

        score = a0 * b0
        score = self.softmax(score)

        # (32, 12, 784)
        score = score[:, :, 1:]

        # 每个头选出的token索引
        heads_select_index = [[] for _ in range(B)]
        for b in range(B):
            for i in range(heads_num):
                _, select_index = torch.topk(score[b, i, :], int(heads_select_rate[i] * total_num), dim=-1)
                heads_select_index[b].extend(select_index.unsqueeze(0))

        for i in range(B):
            for j, b in enumerate(heads_select_index[i]):
                # torch.bincount: 得到输入张量中对应索引值出现的次数, 这里就是12个头
                # count： (32, 784) 每个token对应的分数
                scores = torch.bincount(b, minlength=784)
                count[i, :] += scores

        if last:
            # (32, 784)
            count = self.enhance_local(count)
            pass

        # (32, 784) (32, 784)
        patch_value, patch_idx = torch.sort(count, dim=-1, descending=True)
        last_select_idx = patch_idx[:, :select_num]
        # 每一个head存活下来的数量
        alive_num = torch.zeros((B, heads_num))
        for b in range(B):
            for i, select_indices in enumerate(heads_select_index[b]):
                alive_num[b][i] = torch.sum(torch.isin(select_indices, last_select_idx[b]))

        alive_rate = alive_num / select_num

        return count, alive_rate

    def enhance_local(self, count):
        # B:32 H:28
        B, H = count.shape[0], math.ceil(math.sqrt(count.shape[1]))
        # (32, 28, 28)
        count = count.reshape(B, H, H)
        if self.fix:
            # (32, 1, 28, 28) 卷积后：(32, 784)
            count = self.conv(count.unsqueeze(1), self.kernel, stride=1, padding=1).reshape(B, -1)
        else:
            count = self.conv(count.unsqueeze(1)).reshape(B, -1)
        return count

    def update_layer_select(self, layer_count):
        alpha = 1e-3  # if self.dataset != 'dog' and self.dataset == 'nabirds' else 1e-4
        new_rate = layer_count / layer_count.sum()

        self.select_rate = self.select_rate * (1 - alpha) + alpha * new_rate
        self.select_rate /= self.select_rate.sum()
        self.select_num = self.select_rate * self.total_num

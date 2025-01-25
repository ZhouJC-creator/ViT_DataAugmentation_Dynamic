import math

import numpy as np
import random
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from .MultiHeadVoting import MultiHeadVoting


def attention_crop(attention_maps, input_image, heads_select_rate, total_num):
    # 3, 3, 448, 448
    B, N, W, H = input_image.shape
    input_tensor = input_image
    # 3, 12, 785, 785
    batch_size, num_parts, height, width = attention_maps.shape
    patch_num = int(math.sqrt(height - 1))
    patch_size = int(W / patch_num)
    # # (3, 12, 448, 448) 下采样
    # attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')

    multi_head_voting = MultiHeadVoting()
    # (B, 784), (B, 12)
    count_maps, alive_rate = multi_head_voting(attention_maps, heads_select_rate, total_num)

    ret_imgs = []

    for i in range(B):
        # (784)
        count_map = count_maps[i]
        count_map = count_map.view(patch_num, patch_num)

        mean_score = torch.mean(count_map)
        stddev = torch.std(count_map)
        threshold_crop = mean_score + stddev

        # crop
        itemindex = torch.nonzero(count_map >= threshold_crop)
        height_min = itemindex[:, 0].min() * patch_size
        height_max = itemindex[:, 0].max() * patch_size
        width_min = itemindex[:, 1].min() * patch_size
        width_max = itemindex[:, 1].max() * patch_size

        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        # (1, 3, 448, 448)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
        # (3, 448, 448)
        out_img = out_img.squeeze(0)
        # crop_img.show()
        ret_imgs.append(out_img)

    crop_imgs = torch.stack(ret_imgs)
    return crop_imgs, alive_rate


def attention_drop(attention_maps, input_image):
    # 3, 3, 448, 448
    B, N, W, H = input_image.shape
    input_tensor = input_image
    # 3, 12, 785, 785
    batch_size, num_parts, height, width = attention_maps.shape
    # (3, 12, 448, 448) 下采样
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
    # 平均池化 （3, 12)
    part_weights = F.avg_pool2d(attention_maps.detach(), (W, H)).reshape(B, -1)
    part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
    # 归一化
    part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1)).cpu()
    part_weights = part_weights.numpy()
    masks = []
    for i in range(B):
        # (12, 488, 488)
        attention_map = attention_maps[i]
        # (12)
        part_weight = part_weights[i]
        # 以part_weight为概率选取num_parts中的一个数
        selected_index2 = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]

        # create drop imgs
        mask2 = attention_map[selected_index2:selected_index2 + 1, :, :]
        threshold = random.uniform(0.2, 0.5)
        mask2 = (mask2 < threshold * mask2.max()).float()
        masks.append(mask2)
    masks = torch.stack(masks)
    drop_imgs = input_tensor * masks
    return drop_imgs


def mask2bbox(attention_maps, input_image):
    input_tensor = input_image
    B, C, H, W = input_tensor.shape
    batch_size, num_parts, Hh, Ww = attention_maps.shape
    attention_maps = torch.nn.functional.interpolate(attention_maps, size=(W, H), mode='bilinear')
    ret_imgs = []
    # print(part_weights[3])
    for i in range(batch_size):
        attention_map = attention_maps[i]
        # print(attention_map.shape)
        mask = attention_map.mean(dim=0)
        # print(type(mask))
        # mask = (mask-mask.min())/(mask.max()-mask.min())
        # threshold = random.uniform(0.4, 0.6)
        threshold = 0.1
        max_activate = mask.max()
        min_activate = threshold * max_activate
        itemindex = torch.nonzero(mask >= min_activate)

        padding_h = int(0.05 * H)
        padding_w = int(0.05 * W)
        height_min = itemindex[:, 0].min()
        height_min = max(0, height_min - padding_h)
        height_max = itemindex[:, 0].max() + padding_h
        width_min = itemindex[:, 1].min()
        width_min = max(0, width_min - padding_w)
        width_max = itemindex[:, 1].max() + padding_w
        # print(height_min,height_max,width_min,width_max)
        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
        out_img = out_img.squeeze(0)
        # print(out_img.shape)
        ret_imgs.append(out_img)
    ret_imgs = torch.stack(ret_imgs)
    # print(ret_imgs.shape)
    return ret_imgs


def multi_head_voting(x, select_num=24, last=False):
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], device='cuda').unsqueeze(0).unsqueeze(0).half()
    conv = F.conv2d

    # 32, 784
    B, patch_num = x.shape[0], x.shape[3] - 1
    # (32, 784)
    count = torch.zeros((B, patch_num), dtype=torch.int, device='cuda').half()

    # (32, 12, 784)
    score = x[:, :, 0, 1:]

    # 选出和cls_token关系最强的前几个tokens： select: (32, 12, 24)
    _, select = torch.topk(score, select_num, dim=-1)
    # (32, 288)
    select = select.reshape(B, -1)

    for i, b in enumerate(select):
        # torch.bincount: 得到输入张量中对应索引值出现的次数, 这里就是12个头
        # count： (32, 784) 每个token对应的分数
        scores = torch.bincount(b, minlength=patch_num)
        count[i, :] += scores

    if not last:
        H = math.ceil(math.sqrt(count.shape[1]))
        count = count.reshape(B, H, H)
        count = conv(count.unsqueeze(1), kernel, stride=1, padding=1).reshape(B, -1)
        pass
    # (32, 784) (32, 784)
    patch_value, patch_idx = torch.sort(count, dim=-1, descending=True)
    # 索引+1 (32, 784)
    patch_idx += 1
    # 选取前select_num个token的索引
    # (32, select_num) (32, 784)
    return patch_idx[:, :select_num], count


def tensor_to_image(tensor):
    # 将tensor移动到CPU上，并移除批次维度 (如果有的话)
    tensor = tensor.squeeze().cpu()

    # 创建一个反归一化 transform，如果你的图像是归一化后的
    unnormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1 / s for s in [0.229, 0.224, 0.225]]
    )

    # 反归一化 (仅当你的图像被归一化时需要)
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = unnormalize(tensor)

    # 确保数值在 [0, 1] 范围内
    tensor = torch.clamp(tensor, 0, 1)

    # 将tensor转换为PIL图像
    img = transforms.ToPILImage()(tensor)

    return img

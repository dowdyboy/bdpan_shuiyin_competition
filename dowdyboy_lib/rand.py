import numpy as np
import torch
import random
import os


def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True


def wheel_rand_index(weight_list):
    total = sum(weight_list)
    val = total * random.random()
    thres = 0.
    for idx, weight in enumerate(weight_list):
        if val < thres + weight:
            return idx
        else:
            thres += weight
    return len(weight_list) - 1


def wheel_ratio(weight_list):
    total = sum(weight_list)
    ret = []
    r = 0.
    for weight in weight_list:
        r += weight / total
        ret.append(r)
    return ret





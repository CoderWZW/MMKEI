
import torch
import torch.nn as nn
def assign_values_linear(numbers, base_lr, newLearn_late):
    # 计算最大值和最小值
    min_num, max_num = min(numbers), max(numbers)
    # 定义映射范围
    min_val, max_val = base_lr, newLearn_late
    # 如果所有数相同，则直接返回中间值
    if min_num == max_num:
        return [round((min_val + max_val) / 2, 4)] * len(numbers)
    # 分配值
    values = [min_val + (max_val - min_val) * (num - min_num) / (max_num - min_num) for num in numbers]

    return values


def adjust_learning_rate_of_rows( grad, rows, new_lr, base_lr, listcountitem):
    # print(listcountitem[2])
    values = assign_values_linear(listcountitem, base_lr, new_lr)
    adjusted_grad = grad.clone()
    values=[x / base_lr for x in values]
    # print(len(adjusted_grad))
    # print(len(values))
    for row in rows:

             adjusted_grad[row] *=  torch.tensor(values[row], device='cuda')
    return adjusted_grad
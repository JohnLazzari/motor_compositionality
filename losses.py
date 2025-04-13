import torch

def l1_dist(x, y):
    """L1 loss"""
    return torch.mean(torch.sum(torch.abs(x - y), dim=-1))

def l1_weight(rnn, scale):
    l1 = 0
    for name, param in rnn.named_parameters():
        l1 += torch.mean(torch.abs(torch.flatten(param)))
    l1 *= scale
    return l1

def l1_rate(act, scale):
    l1 = scale * torch.mean(torch.abs(torch.flatten(act)))
    return l1

def l1_muscle_act(act, scale):
    l1 = scale * torch.mean(torch.sum(torch.abs(act), dim=(1, 2)))
    return l1
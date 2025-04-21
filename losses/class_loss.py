import torch
import torch.nn as nn

import numpy as np
import math

class Loss_Softmax(nn.Module):
    def __init__(self):
        super(Loss_Softmax, self).__init__()
        self.criterion  = torch.nn.CrossEntropyLoss()
        print('Initialised Softmax Loss')

    def forward(self, x, label=None):
        nloss   = self.criterion(x, label)

        prec1, _ = accuracy(x.detach().cpu(), label.detach().cpu(), topk=(1, 5))
        return nloss, prec1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, f_a, f_b):
        norm_a = self.at(f_a)
        norm_b = self.at(f_b)
        loss = (norm_a - norm_b).pow(2).sum()/f_a.size(0)
        return loss

    def at(self, f):
        return F.normalize(f.pow(2).mean(1).view(f.size(0), -1))



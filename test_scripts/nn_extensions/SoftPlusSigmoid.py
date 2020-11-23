import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftPlusSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.softplus(input) * F.sigmoid(input)

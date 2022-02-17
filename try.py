import torch
import torch.nn as nn
conv = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=1, stride=1, groups=1)
print(conv.weight.size())
conv = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=1, stride=1, groups=2)
print(conv.weight.size())
conv = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=1, stride=1, groups=3)
print(conv.weight.size())
conv = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=1, stride=1, groups=4)
print(conv.weight.size())
conv = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=1, stride=1, groups=6)
print(conv.weight.size())
conv = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=1, stride=1, groups=12)
print(conv.weight.size())

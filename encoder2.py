import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_dim=1, h_dim=128, n_res_layers=4, res_h_dim=128):
        super(Encoder, self).__init__()
        
        self.conv_stack = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_dim, h_dim // 4, kernel_size=5, stride=2, padding=2),  # 输出 [1, 64, 112, 32]
            nn.BatchNorm2d(h_dim // 4),
            nn.ReLU(),
            # 第二层卷积
            nn.Conv2d(h_dim // 4, h_dim // 2, kernel_size=5, stride=2, padding=2),  # 输出 [1, 128, 56, 16]
            nn.BatchNorm2d(h_dim // 2),
            nn.ReLU(),
            # 第三层卷积
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=5, stride=2, padding=2),  # 输出 [1, 256, 28, 8]
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
            # 第四层卷积
            nn.Conv2d(h_dim, 128, kernel_size=5, stride=2, padding=2),  # 输出 [1, 128, 14, 4]
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((224, 64))  # Update shape to (1, 100, 64)
    x = torch.tensor(x).float().unsqueeze(0).unsqueeze(0)  # Add batch dimension to make it (1, 1, 100, 64)

    # test encoder
    encoder = Encoder()
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)


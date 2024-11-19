import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z, p_phi maps back to the original space z -> x.
    """

    def __init__(self, in_dim=64, h_dim=128, res_h_dim=128, n_res_layers=4):
        super(Decoder, self).__init__()

        self.inverse_conv_stack = nn.Sequential(
            # 第一个反卷积层
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=4, stride=2, padding=1),  # 输出 [32, 128, 28, 8]
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
            # 残差层
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            # 第二个反卷积层
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),  # 输出 [32, 64, 56, 16]
            nn.BatchNorm2d(h_dim // 2),
            nn.ReLU(),
            # 第三个反卷积层
            nn.ConvTranspose2d(h_dim // 2, 1, kernel_size=4, stride=2, padding=1),  # 输出 [32, 1, 112, 32]
            # 最后一个反卷积层，将形状从 [32, 1, 112, 32] 转换为 [32, 1, 224, 64]
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # 输出 [32, 1, 224, 64]
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)

# 测试
if __name__ == "__main__":
    decoder = Decoder()
    z_q = torch.randn(32, 64, 14, 4)  # 示例输入
    output = decoder(z_q)
    print('Decoder output shape:', output.shape)  # 应输出 [32, 1, 224, 64]


# if __name__ == "__main__":
#     # random data
#     x = np.random.random_sample((3, 40, 40, 200))
#     x = torch.tensor(x).float()

#     # test decoder
#     decoder = Decoder(40, 128, 3, 64)
#     decoder_out = decoder(x)
#     print('Dncoder out shape:', decoder_out.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

class MaskNet(nn.Module):
    def __init__(self):
        super(MaskNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 输出通道增加到16
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # 输出通道保持1
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(1)
          # Initialize the weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize conv1 layer weights using a normal distribution
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv1.bias is not None:
            init.constant_(self.conv1.bias, 0)
        
        # Initialize conv2 layer weights using a normal distribution
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv2.bias is not None:
            init.constant_(self.conv2.bias, 0)
        
        # Initialize batchnorm1 and batchnorm2
        init.constant_(self.bn1.weight, 1)
        init.constant_(self.bn1.bias, 0)
        
        init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)  # 使用 LeakyReLU 激活
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = torch.sigmoid(x)  # 添加 Sigmoid 激活函数
        return x

# 假设你的输入数据为 input_tensor，形状为 [32, 1, 224, 64]
# input_tensor = torch.rand(32, 1, 224, 64)  # 示例输入，实际应为你的数据
# normalized_tensor = normalize_tensor(input_tensor)

# # 创建模型并前向传播
# model = MaskNet()
# output = model(normalized_tensor)


# 示例：初始化模型并打印输出形状
# if __name__ == "__main__":
#     model = MaskNet()
#     # 输入数据形状为 (batch_size, channels, height, width)
#     x = torch.randn(32, 1, 224, 64)  # 示例输入
#     output = model(x)
#     print("Output shape:", output.shape)  # 输出形状

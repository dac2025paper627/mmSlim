import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder
from models.vqvae import VQVAE
from models.new_vqvae import VQVAE1
from models.masknet import MaskNet
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import Counter
import heapq
import os

def huffman_encode(tensor):
    # 将张量展平成一维列表
    flat_data = tensor.view(-1).tolist()
    
    # 统计每个值的频率
    frequency = Counter(flat_data)
    
    # 构建哈夫曼树
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # 获取哈夫曼编码字典
    huffman_code = {symbol: code for symbol, code in sorted(heap[0][1:], key=lambda p: (len(p[-1]), p))}
    
    # 编码数据
    encoded_data = ''.join(huffman_code[value] for value in flat_data)
    
    return encoded_data, huffman_code

def huffman_decode(encoded_data, huffman_code, original_shape, device=None):
    # Generate the decode dictionary
    decode_huffman = {v: k for k, v in huffman_code.items()}
    
    # Decode the bitstream
    decoded_values = []
    code = ""
    for bit in encoded_data:
        code += bit
        if code in decode_huffman:
            decoded_values.append(decode_huffman[code])
            code = ""

    # Ensure decoded_values has been populated
    if not decoded_values:
        raise ValueError("Decoding failed. `decoded_values` is empty. Check encoded_data and huffman_code.")
    
    # Convert decoded values to tensor and reshape
    decoded_tensor = torch.tensor(decoded_values).view(original_shape)
    
    # Move tensor to the specified device if provided
    if device is not None:
        decoded_tensor = decoded_tensor.to(device)
    
    return decoded_tensor

def plot_and_save_image_comparison(x_original, i, idx, type):
    # 将tensor转换为numpy数组并取消梯度追踪
    x_original_np = x_original.cpu().detach().numpy()
    x_original_np=x_original_np[0,0,:,:]
        

    # 创建一个包含两个子图的figure，设置大小
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 在第一个子图中绘制原始图片
    axes[0].imshow(x_original_np, cmap='viridis')  # 使用 'viridis' colormap
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # 关闭坐标轴

    # 在第二个子图中绘制重构后的图片


    # 设置整个图的标题，指定更新次数和索引
    fig.suptitle(f"Comparison of Original and Reconstructed Image at update {i}, index {idx}, {'Amplitude' if type == 0 else 'Phase'}")

    # 调整布局
    plt.tight_layout()

    # 确定保存目录
  # folder = 'amplitude' if type == 0 else 'phase'
    # folder = 'amp1' if type == 0 else 'ph1'
    folder = 'test_ph2' if type == 0 else 'test_ph3'
    save_dir = f'/image32'

    # 创建文件夹（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图片到指定目录
    save_path = os.path.join(save_dir, f'comparison_update_{i}_index_{idx}.png')
    plt.savefig(save_path)
    np.save(save_path,x_original_np)
    # 关闭图像，防止内存泄漏
    plt.close()

    # 返回保存路径
    return save_path

class Dual_vqvae(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False,masknet_weights=None):
        super(Dual_vqvae,self).__init__()

        self.vqvae_1=VQVAE(h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta)
        self.masknet=MaskNet()

        # 如果提供了外部权重，则加载这些权重
        if masknet_weights is not None:
            self.masknet.load_state_dict(torch.load(masknet_weights))
        
            # 冻结 MaskNet 的参数
            for param in self.masknet.parameters():
                param.requires_grad = False
            
        self.vqvae_2=VQVAE1(h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta)

    
    def ssim_loss(self, x, y, C1=1e-4, C2=9e-4):
        mu_x = F.conv2d(x, torch.ones(1, 1, 11, 11).to(x.device) / 121, padding=5)
        mu_y = F.conv2d(y, torch.ones(1, 1, 11, 11).to(y.device) / 121, padding=5)

        sigma_x = F.conv2d(x * x, torch.ones(1, 1, 11, 11).to(x.device) / 121, padding=5) - mu_x * mu_x
        sigma_y = F.conv2d(y * y, torch.ones(1, 1, 11, 11).to(y.device) / 121, padding=5) - mu_y * mu_y
        sigma_xy = F.conv2d(x * y, torch.ones(1, 1, 11, 11).to(x.device) / 121, padding=5) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                    ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        return 1 - ssim_map.mean()  # Loss: we want to minimize the SSIM value
        
    def forward(self,x,amp,verbose=False):
        mask = self.masknet(x)  # 将 amp 作为输入传递给 MaskNet
        amp1=mask*amp
        amp2=(1-mask)*amp

        amp1_embedding_loss,amp1_hat,amp1_perplexity,embedding_indices1=self.vqvae_1(amp1)
        np.set_printoptions(threshold=np.inf)  # 设置为无限大，取消省略
        amp2_embedding_loss,amp2_hat,amp2_perlexity,embedding_indices2=self.vqvae_2(amp2)
        # embedding_indices=torch.cat((embedding_indices1,embedding_indices2),dim=0)
        # self.reconstruct_from_indices(embedding_indices,amp1.numel*32)
        amp_hat=amp1_hat+amp2_hat
        amp_hat=amp_hat*(-1)
        plot_and_save_image_comparison(amp_hat,2,2,0)
        # print(amp_hat)
        # recon_error1=torch.mean((amp1-amp1_hat)**2)
        # recon_error2=torch.mean((amp2-amp2_hat)**2)
        ssim_loss1 = self.ssim_loss(amp1, amp1_hat)  # 替代 recon_error1
        ssim_loss2 = self.ssim_loss(amp2, amp2_hat)  # 替代 recon_error2
        return amp_hat,amp1_embedding_loss,amp2_embedding_loss,amp1_perplexity,amp2_perlexity,ssim_loss1,ssim_loss2,mask


    
    def reconstruct_from_indices(self, embedding_indices,x_digits):
        """
        使用嵌入索引重构 z_q，然后通过解码器重建图像
        # """
        # # 获取编码器的输出形状
        # batch_size, _, h, w = self.z_e_shape  # h 和 w 是编码器输出的高度和宽度

        # # 确保总大小相匹配
        # total_indices = embedding_indices.numel()  # 获取索引的总数量
        # # print(f"重建时: batch_size: {batch_size}, h: {h}, w: {w}, total_indices: {total_indices}")
        # assert total_indices == batch_size * h * w, f"Expected {batch_size * h * w} total indices, but got {total_indices}"

        # # 将 embedding_indices 调整为 [batch_size, h, w]
        # embedding_indices = embedding_indices.view(batch_size, h, w)
        # Get original data size in bits
        embedding_size_bits = embedding_indices.numel() * 32  # Assuming each value is 32 bits

        # Encode using Huffman encoding
        encoded_data, huffman_code = huffman_encode(embedding_indices)
        print("Encoded data:", encoded_data)
        print("Huffman code:", huffman_code)

        # Get encoded data size in bits
        encoded_size_bits = len(encoded_data)  # Each character in the string represents 1 bit

        # Display the original and encoded sizes
        print(f"Original size: {x_digits} bits")
        print(f"embedding size: {embedding_size_bits} bits")
        print(f"Encoded size: {encoded_size_bits} bits")
        print(f"Compression ratio(without huffman): { x_digits/ embedding_size_bits:.2f}")
        print(f"Compression ratio(with huffman): {x_digits / encoded_size_bits:.2f}")

        # 解码
        # Make sure both tensors are on the same device
        # Get the device of embedding_indices
        device = embedding_indices.device

        # Decode with device specified
        decoded_tensor = huffman_decode(encoded_data, huffman_code, embedding_indices.size(), device=device)

        # Now you can safely compare
        print("Decoded tensor matches original:", torch.equal(decoded_tensor, embedding_indices))
        # # 获取量化器的 codebook (embedding 的 weight)
        # codebook = self.vector_quantization.embedding.weight
        # print(f"codebook大小为:{codebook.shape}")
        # print(f"embedding_indices大小为{embedding_indices.shape}")
        # # 使用嵌入索引从 codebook 中获取对应的嵌入向量
        # z_q = codebook[embedding_indices.view(-1)]  # 展开为 (batch_size * h * w, embedding_dim)
        # z_q = z_q.view(batch_size, h, w, -1).permute(0, 3, 1, 2)  # 恢复为 (batch_size, channels, h, w)

        # # 使用解码器解码 z_q，重建数据
        # x_reconstructed = self.decoder(z_q)

        # return x_reconstructed

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Dual_vqvae(h_dim=128, res_h_dim=32, n_res_layers=2, n_embeddings=512, embedding_dim=64, beta=0.25).to(device)

# # # 假设输入数据 x 是一个随机生成的 100x32 图像
# x = torch.randn(32, 1, 224, 124).to(device)
# # original_x_digits=x.numel()*32
# # # # 前向传播：编码并重建图像
# embedding_loss, x_recon, perplexity, embedding_indices = model(x)
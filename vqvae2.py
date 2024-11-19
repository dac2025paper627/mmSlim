import torch
import torch.nn as nn
from models.quantizer import VectorQuantizer
from models.new_decoder import Decoder
from models.new_encoder import Encoder
import os
from collections import Counter
import heapq

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置可见的 GPU 为第二张卡

def generate_mask(x_hat):
    # 确保输入是 PyTorch 张量
    if not isinstance(x_hat, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    # 计算 x_hat 的最小值和最大值
    amplitude_data_min = torch.min(x_hat)
    amplitude_data_max = torch.max(x_hat)

    # 归一化数据
    normalize_data = (x_hat - amplitude_data_min) / (amplitude_data_max - amplitude_data_min)

    # 计算阈值
    threshold = torch.max(normalize_data) * 0.32

    # 生成掩码
    mask = normalize_data > threshold
    
    return mask
    # 调整 mask 的形状为与 ph 一致



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


    # return mask
class VQVAE1(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False,kernel_size=4):
        super(VQVAE1, self).__init__()

        # 编码器：将图像编码为连续的潜在空间
        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)

        # 向量量化：通过离散化瓶颈将连续的潜在向量转换为离散表示
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)

        # 解码器：从离散的潜在表示重建图像
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)


        # 可选：保存图像与嵌入映射关系
        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        # 编码器：输入数据编码为连续潜在表示
        z_e = self.encoder(x)

        # 打印编码器输出的形状
        print(f"编码器输出 z_e 的形状: {z_e.shape}")

        # 获取z_e的空间尺寸，用于之后的重建
        self.z_e_shape = z_e.shape  # (batch_size, channels, height, width)


        # 通过1x1卷积减少通道数，准备输入向量量化器
        z_e = self.pre_quantization_conv(z_e)
        # print(f"预量化z_e 的形状: {z_e.shape}")
        # 向量量化
        embedding_loss, z_q, perplexity, min_encodings, embedding_indices = self.vector_quantization(z_e)
        codebook = self.vector_quantization.embedding.weight
        print(f"codebook大小为:{codebook.shape}")
        # 打印 embedding_indices 的形状
        # print(f"embedding_indices 的形状: {embedding_indices.shape}")
        # print(f"z_q 的形状: {z_q.shape}")
        # 解码器：从量化后的向量重建数据
        x_hat = self.decoder(z_q)
        
        if verbose:
            print('Original data shape:', x.shape)
            print('Encoded data shape:', z_e.shape)
            print('Reconstructed data shape:', x_hat.shape)

        return embedding_loss, x_hat, perplexity, embedding_indices

    def reconstruct_from_indices(self, embedding_indices,x_digits):
        """
        使用嵌入索引重构 z_q，然后通过解码器重建图像
        """
        # 获取编码器的输出形状
        batch_size, _, h, w = self.z_e_shape  # h 和 w 是编码器输出的高度和宽度

        # 确保总大小相匹配
        total_indices = embedding_indices.numel()  # 获取索引的总数量
        # print(f"重建时: batch_size: {batch_size}, h: {h}, w: {w}, total_indices: {total_indices}")
        assert total_indices == batch_size * h * w, f"Expected {batch_size * h * w} total indices, but got {total_indices}"

        # 将 embedding_indices 调整为 [batch_size, h, w]
        embedding_indices = embedding_indices.view(batch_size, h, w)
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
        # 获取量化器的 codebook (embedding 的 weight)
        codebook = self.vector_quantization.embedding.weight
        print(f"codebook大小为:{codebook.shape}")
        print(f"embedding_indices大小为{embedding_indices.shape}")
        # 使用嵌入索引从 codebook 中获取对应的嵌入向量
        z_q = codebook[embedding_indices.view(-1)]  # 展开为 (batch_size * h * w, embedding_dim)
        z_q = z_q.view(batch_size, h, w, -1).permute(0, 3, 1, 2)  # 恢复为 (batch_size, channels, h, w)

        # 使用解码器解码 z_q，重建数据
        x_reconstructed = self.decoder(z_q)

        return x_reconstructed


# # # # 实例化模型
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = VQVAE1(h_dim=128, res_h_dim=32, n_res_layers=2, n_embeddings=512, embedding_dim=64, beta=0.25).to(device)

# # # # 假设输入数据 x 是一个随机生成的 100x32 图像
# x = torch.randn(32, 1, 224, 124).to(device)
# # # original_x_digits=x.numel()*32
# # # # # 前向传播：编码并重建图像
# embedding_loss, x_recon, perplexity, embedding_indices = model(x)

# # 通过 embedding 索引从 codebook 中重建
# x_reconstructed_from_indices = model.reconstruct_from_indices(embedding_indices,original_x_digits)

# # 输出原始数据大小和重建后数据大小
# print("原始数据大小:", x.shape)
# print("通过 embedding 索引重建后的数据大小:", x_reconstructed_from_indices.shape)


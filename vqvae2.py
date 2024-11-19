import torch
import torch.nn as nn
from models.quantizer import VectorQuantizer
from models.decoder2 import Decoder
from models.encoder2 import Encoder
import os
from collections import Counter
import heapq

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def generate_mask(x_hat):
    if not isinstance(x_hat, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    amplitude_data_min = torch.min(x_hat)
    amplitude_data_max = torch.max(x_hat)

    normalize_data = (x_hat - amplitude_data_min) / (amplitude_data_max - amplitude_data_min)

    # Calculate threshold
    threshold = torch.max(normalize_data) * 0.32

    # Generate mask
    mask = normalize_data > threshold

    return mask


def huffman_encode(tensor):
    flat_data = tensor.view(-1).tolist()

    frequency = Counter(flat_data)

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

    huffman_code = {symbol: code for symbol, code in sorted(heap[0][1:], key=lambda p: (len(p[-1]), p))}

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
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False, kernel_size=4):
        super(VQVAE1, self).__init__()

        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)

        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)

        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)

        self.z_e_shape = z_e.shape  # (batch_size, channels, height, width)

        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, min_encodings, embedding_indices = self.vector_quantization(z_e)
        codebook = self.vector_quantization.embedding.weight

        x_hat = self.decoder(z_q)

        if verbose:
            print('Original data shape:', x.shape)
            print('Encoded data shape:', z_e.shape)
            print('Reconstructed data shape:', x_hat.shape)

        return embedding_loss, x_hat, perplexity, embedding_indices

    def reconstruct_from_indices(self, embedding_indices, x_digits):
        """
        Reconstruct z_q using embedding indices, then reconstruct the image through the decoder
        """
        # Get encoder's output shape
        batch_size, _, h, w = self.z_e_shape  # h and w are the height and width of the encoder output

        # Ensure total size matches
        total_indices = embedding_indices.numel()  # Get the total number of indices
        # print(f"Reconstruction: batch_size: {batch_size}, h: {h}, w: {w}, total_indices: {total_indices}")
        assert total_indices == batch_size * h * w, f"Expected {batch_size * h * w} total indices, but got {total_indices}"

        # Reshape embedding_indices to [batch_size, h, w]
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
        print(f"Embedding size: {embedding_size_bits} bits")
        print(f"Encoded size: {encoded_size_bits} bits")
        print(f"Compression ratio (without Huffman): {x_digits / embedding_size_bits:.2f}")
        print(f"Compression ratio (with Huffman): {x_digits / encoded_size_bits:.2f}")

        # Decode
        # Make sure both tensors are on the same device
        # Get the device of embedding_indices
        device = embedding_indices.device

        # Decode with device specified
        decoded_tensor = huffman_decode(encoded_data, huffman_code, embedding_indices.size(), device=device)

        # Now you can safely compare
        print("Decoded tensor matches original:", torch.equal(decoded_tensor, embedding_indices))
        # Get the quantizer's codebook (embedding weights)
        codebook = self.vector_quantization.embedding.weight
        print(f"Codebook size: {codebook.shape}")
        print(f"Embedding indices size: {embedding_indices.shape}")
        # Use embedding indices to get corresponding embedding vectors from the codebook
        z_q = codebook[embedding_indices.view(-1)]  # Flatten to (batch_size * h * w, embedding_dim)
        z_q = z_q.view(batch_size, h, w, -1).permute(0, 3, 1, 2)  # Reshape to (batch_size, channels, h, w)

        # Use the decoder to decode z_q and reconstruct data
        x_reconstructed = self.decoder(z_q)

        return x_reconstructed

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = VQVAE1(h_dim=128, res_h_dim=32, n_res_layers=2, n_embeddings=512, embedding_dim=64, beta=0.25).to(device)
# x = torch.randn(32, 1, 224, 124).to(device)
# # # original_x_digits=x.numel()*32
# embedding_loss, x_recon, perplexity, embedding_indices = model(x)

# x_reconstructed_from_indices = model.reconstruct_from_indices(embedding_indices, original_x_digits)

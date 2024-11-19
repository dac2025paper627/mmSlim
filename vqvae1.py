import sys
import torch
import os
import torch.nn as nn
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder
from models.masknet import MaskNet
import numpy as np
import time
from collections import Counter
import heapq

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def huffman_encode(tensor):
    # Flatten the tensor into a 1D list
    flat_data = tensor.view(-1).tolist()

    # Count the frequency of each value
    frequency = Counter(flat_data)

    # Build the Huffman tree
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

    # Get the Huffman coding dictionary
    huffman_code = {symbol: code for symbol, code in sorted(heap[0][1:], key=lambda p: (len(p[-1]), p))}

    # Encode the data
    encoded_data = ''.join(huffman_code[value] for value in flat_data)

    return encoded_data, huffman_code


def huffman_decode(encoded_data, huffman_code, original_shape, device=None):
    # Generate the decoding dictionary
    decode_huffman = {v: k for k, v in huffman_code.items()}

    # Decode the bitstream
    decoded_values = []
    code = ""
    for bit in encoded_data:
        code += bit
        if code in decode_huffman:
            decoded_values.append(decode_huffman[code])
            code = ""

    # Ensure decoded_values is populated
    if not decoded_values:
        raise ValueError("Decoding failed. `decoded_values` is empty. Check encoded_data and huffman_code.")

    # Convert decoded values to tensor and reshape
    decoded_tensor = torch.tensor(decoded_values).view(original_shape)

    # Move tensor to the specified device if provided
    if device is not None:
        decoded_tensor = decoded_tensor.to(device)

    return decoded_tensor


# return mask
class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False, kernel_size=4):
        super(VQVAE, self).__init__()

        # Encoder: Encodes the image into continuous latent space
        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)

        # Vector quantization: Converts continuous latent vectors into discrete representations
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)

        # Decoder: Reconstructs the image from discrete latent representations
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        # Optional: Save image-to-embedding map relations
        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        # Encoder: Encode input data into continuous latent representation
        z_e = self.encoder(x)

        # Print shape of encoder output
        print(f"Encoder output z_e shape: {z_e.shape}")

        # Get the spatial dimensions of z_e for reconstruction later
        self.z_e_shape = z_e.shape  # (batch_size, channels, height, width)

        # Reduce channels through 1x1 convolution and prepare for vector quantizer input
        z_e = self.pre_quantization_conv(z_e)

        # Vector quantization
        embedding_loss, z_q, perplexity, min_encodings, embedding_indices = self.vector_quantization(z_e)

        # Print the shape of embedding_indices
        # print(f"Shape of embedding_indices: {embedding_indices.shape}")
        # print(f"Shape of z_q: {z_q.shape}")

        # Decoder: Reconstruct data from quantized vectors
        x_hat = self.decoder(z_q)

        if verbose:
            print('Original data shape:', x.shape)
            print('Encoded data shape:', z_e.shape)
            print('Reconstructed data shape:', x_hat.shape)

        return embedding_loss, x_hat, perplexity, embedding_indices

    def benchmark_inference(self, npy_folder, npy_folder2, num_runs=1000):
        """
        Load 1000 randomly generated .npy files and calculate the average inference time and standard deviation for each run.

        Parameters:
            npy_folder: Path to the folder containing all .npy files
            num_runs: Number of inference runs (i.e., number of files)
        """
        # Initialize time accumulation variables
        quantization_times = []
        decoder_times = []

        # Ensure the folder exists
        if not os.path.exists(npy_folder):
            raise ValueError(f"The folder {npy_folder} does not exist.")

        # Get all .npy files
        npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]
        npy_files = npy_files[:num_runs]  # Use the first num_runs files
        npy_files2 = [f for f in os.listdir(npy_folder2) if f.endswith('.npy')]
        npy_files2 = npy_files2[:num_runs]  # Use the first num_runs files

        for npy_file in npy_files:
            # Load each .npy file
            npy_path = os.path.join(npy_folder, npy_file)
            npy_path2 = os.path.join(npy_folder2, npy_file)  # Corrected path for the second folder

            # Load data from the first and second folder
            x = np.load(npy_path)
            x2 = np.load(npy_path2)

            # Convert to tensors and move to the correct device (GPU or CPU)
            x = torch.tensor(x).unsqueeze(0).unsqueeze(0).float().to(device)  # Add batch and channel dimensions
            x2 = torch.tensor(x2).unsqueeze(0).unsqueeze(0).float().to(device)

            # Combine the data from both sources
            combined_data = np.stack([x.cpu().detach().numpy(), x2.cpu().detach().numpy()],
                                     axis=0)  # Move to CPU and convert to NumPy

            # Stack the data safely in CPU memory before converting back to a tensor
            combined_data = torch.tensor(combined_data).unsqueeze(0).to(device)
            combined_data = combined_data.view(combined_data.shape[0], combined_data.shape[1], combined_data.shape[4],
                                               combined_data.shape[5])

            # Quantization process
            start_time = time.time()
            z_e = self.encoder(combined_data)
            z_e = self.pre_quantization_conv(z_e)
            embedding_loss, z_q, perplexity, min_encodings, embedding_indices = self.vector_quantization(z_e)
            quantization_time = time.time() - start_time
            quantization_times.append(quantization_time)

            # Decoding process
            start_time = time.time()
            x_hat = self.decoder(z_q)
            decoder_time = time.time() - start_time
            decoder_times.append(decoder_time)

        # Calculate average time
        avg_quantization_time = np.mean(quantization_times)
        avg_decoder_time = np.mean(decoder_times)

        # Calculate standard deviation
        std_quantization_time = np.std(quantization_times)
        std_decoder_time = np.std(decoder_times)

        # Print results
        print(f"Average quantization time: {avg_quantization_time:.6f} seconds")
        print(f"Standard deviation of quantization time: {std_quantization_time:.6f} seconds")
        print(f"Average decoder time: {avg_decoder_time:.6f} seconds")
        print(f"Standard deviation of decoder time: {std_decoder_time:.6f} seconds")

        return avg_quantization_time, std_quantization_time, avg_decoder_time, std_decoder_time

    def reconstruct_from_indices(self, embedding_indices, x_digits):
        """
        Reconstruct z_q from embedding indices, then reconstruct the image through the decoder
        """
        # Get the shape of the encoder's output
        batch_size, _, h, w = self.z_e_shape  # h and w are the height and width of the encoder's output

        # Ensure the total size matches
        total_indices = embedding_indices.numel()  # Get the total number of indices
        # print(f"Reconstructing: batch_size: {batch_size}, h: {h}, w: {w}, total_indices: {total_indices}")
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
        # Ensure both tensors are on the same device
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

        # Use the embedding indices to retrieve the corresponding embedding vectors from the codebook
        z_q = codebook[embedding_indices.view(-1)]  # Flatten to (batch_size * h * w, embedding_dim)
        z_q = z_q.view(batch_size, h, w, -1).permute(0, 3, 1, 2)  # Reshape to (batch_size, channels, h, w)

        # Use the decoder to decode z_q and reconstruct the data
        x_reconstructed = self.decoder(z_q)

        return x_reconstructed

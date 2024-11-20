import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder
from models.vqvae1 import VQVAE
from models.vqvae2 import VQVAE1
from models.masknet import MaskNet
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import Counter
import heapq
import os


# Huffman encoding function
def huffman_encode(tensor):
    # Flatten the tensor to a 1D list
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

    # Get the Huffman encoding dictionary
    huffman_code = {symbol: code for symbol, code in sorted(heap[0][1:], key=lambda p: (len(p[-1]), p))}

    # Encode the data
    encoded_data = ''.join(huffman_code[value] for value in flat_data)

    return encoded_data, huffman_code


# Huffman decoding function
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


# Plotting and saving image comparison
def plot_and_save_image_comparison(x_original, i, idx, type):
    # Convert tensor to numpy array and detach gradient tracking
    x_original_np = x_original.cpu().detach().numpy()
    x_original_np = x_original_np[0, 0, :, :]

    # Create a figure with two subplots, set the size
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image in the first subplot
    axes[0].imshow(x_original_np, cmap='viridis')  # Use 'viridis' colormap
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Turn off axes

    # Set the title for the entire figure, specifying the update number and index
    fig.suptitle(
        f"Comparison of Original and Reconstructed Image at update {i}, index {idx}, {'Amplitude' if type == 0 else 'Phase'}")

    # Adjust layout
    plt.tight_layout()

    # Set save directory
    folder = 'test_amp' if type == 0 else 'test_ph'
    save_dir = f'./image32'

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the image to the specified directory
    save_path = os.path.join(save_dir, f'comparison_update_{i}_index_{idx}.png')
    plt.savefig(save_path)
    np.save(save_path, x_original_np)

    # Close the plot to avoid memory leaks
    plt.close()

    # Return the save path
    return save_path


# mmSlim model
class mmSlim(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False, masknet_weights=None):
        super(mmSlim, self).__init__()

        self.vqvae_1 = VQVAE(h_dim, res_h_dim, n_res_layers,
                             n_embeddings, embedding_dim, beta)
        self.masknet = MaskNet()

        # If external weights are provided, load them into MaskNet
        if masknet_weights is not None:
            self.masknet.load_state_dict(torch.load(masknet_weights))

            # Freeze MaskNet's parameters
            for param in self.masknet.parameters():
                param.requires_grad = False

        self.vqvae_2 = VQVAE1(h_dim, res_h_dim, n_res_layers,
                              n_embeddings, embedding_dim, beta)

    # SSIM loss function (used to compare image similarity)
    def ssim_loss(self, x, y, C1=1e-4, C2=9e-4):
        mu_x = F.conv2d(x, torch.ones(1, 1, 11, 11).to(x.device) / 121, padding=5)
        mu_y = F.conv2d(y, torch.ones(1, 1, 11, 11).to(y.device) / 121, padding=5)

        sigma_x = F.conv2d(x * x, torch.ones(1, 1, 11, 11).to(x.device) / 121, padding=5) - mu_x * mu_x
        sigma_y = F.conv2d(y * y, torch.ones(1, 1, 11, 11).to(x.device) / 121, padding=5) - mu_y * mu_y
        sigma_xy = F.conv2d(x * y, torch.ones(1, 1, 11, 11).to(x.device) / 121, padding=5) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        return 1 - ssim_map.mean()  # Loss: we want to minimize the SSIM value

    def forward(self, x, amp, verbose=False):
        mask = self.masknet(x) #use amplitude to generate mask
        amp1 = mask * amp
        amp2 = (1 - mask) * amp

        amp1_embedding_loss, amp1_hat, amp1_perplexity, embedding_indices1 = self.vqvae_1(amp1)
        np.set_printoptions(threshold=np.inf)  # Disable numpy array truncation
        amp2_embedding_loss, amp2_hat, amp2_perplexity, embedding_indices2 = self.vqvae_2(amp2)
        amp_hat = amp1_hat + amp2_hat
        amp_hat = amp_hat * (-1)

        # SSIM loss calculation for each part
        ssim_loss1 = self.ssim_loss(amp1, amp1_hat)  # Replace with recon_error1
        ssim_loss2 = self.ssim_loss(amp2, amp2_hat)  # Replace with recon_error2
        return amp_hat, amp1_embedding_loss, amp2_embedding_loss, amp1_perplexity, amp2_perplexity, ssim_loss1, ssim_loss2, mask

    # Function to reconstruct from embedding indices
    def reconstruct_from_indices(self, embedding_indices, x_digits):
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
        print(f"Compression ratio(without huffman): {x_digits / embedding_size_bits:.2f}")
        print(f"Compression ratio(with huffman): {x_digits / encoded_size_bits:.2f}")

        # Decode
        # Make sure both tensors are on the same device
        device = embedding_indices.device
        decoded_tensor = huffman_decode(encoded_data, huffman_code, embedding_indices.size(), device=device)

        # Now you can safely compare
        print("Decoded tensor matches original:", torch.equal(decoded_tensor, embedding_indices))

# Example usage:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = mmSlim(h_dim=128, res_h_dim=32, n_res_layers=2, n_embeddings=512, embedding_dim=64, beta=0.25).to(device)
# x = torch.randn(32, 1, 224, 124).to(device)
# embedding_loss, x_recon, perplexity, embedding_indices = model(x)

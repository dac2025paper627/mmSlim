import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import utils
import single_utils
from torch.utils.tensorboard import SummaryWriter
from models.vqvae import VQVAE
from models.new_dual_vqvae import Dual_vqvae
import matplotlib.pyplot as plt
import thop

parser = argparse.ArgumentParser()
# Define hyperparameters
timestamp = utils.readable_timestamp()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=500)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate_1", type=float, default=5e-5)
parser.add_argument("--learning_rate_2", type=float, default=1e-5)
parser.add_argument("--learning_rate_mask", type=float, default=5e-5)
parser.add_argument("--log_interval", type=int, default=2)
parser.add_argument("--dataset", type=str, default='CUSTOM')
# parser.add_argument("--data_dir", type=str, default='./dataset/phase/0/2', help="Directory of the dataset")
parser.add_argument("--amplitude_dir", type=str, default='./new_dataset_all_12/amplitude/data64', help="Directory of the dataset")
parser.add_argument("--phase_dir", type=str, default='./new_dataset_all_12/phase/data64', help="Directory of the dataset")
parser.add_argument("--save_npy", action="store_true", help="Save dataset as npy files")

# Whether to save the model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename", type=str, default="vqvae")

args = parser.parse_args()
args.save = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

# Load data and define data loaders
training_data, validation_data, training_loader, validation_loader = single_utils.load_data_and_data_loaders(
    args.dataset, args.batch_size, args.amplitude_dir, args.phase_dir, save_npy_files=args.save_npy)

masknet_path = './results/vqvae_data_sun_nov_10_01_18_16_2024masknet_params.pth'

# Set up VQ-VAE model
model = Dual_vqvae(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta, masknet_path).to(device)

log_dir = './runs/vqvae_experiment' + timestamp
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# Define TensorBoard writer
writer = SummaryWriter(log_dir)

# Set up optimizer and training loop
# optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
optimizer = optim.Adam([
    {'params': model.vqvae_1.parameters(), 'lr': args.learning_rate_1},  # Learning rate for VQVAE
    {'params': model.vqvae_2.parameters(), 'lr': args.learning_rate_2},  # Learning rate for VQVAE1
    # {'params': model.masknet.parameters(), 'lr': args.learning_rate_mask}  # Learning rate for MaskNet (if needed)
], amsgrad=True)

# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_updates)  # Total training epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}
overall_loss = {
    'train_loss': 0,
    'test_loss': 0
}


def plot_and_save_image_comparison(x_original, x_reconstructed, i, idx, type):
    # Convert tensor to numpy array and detach gradient tracking
    x_original_np = x_original[type, :, :].cpu().detach().numpy()  # 224x128
    x_reconstructed_np = x_reconstructed[0, :, :].cpu().detach().numpy()  # 224x128

    # Create a figure with two subplots and set the size
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image in the first subplot
    axes[0].imshow(x_original_np, cmap='viridis')  # Use 'viridis' colormap
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Turn off axis

    # Plot the reconstructed image in the second subplot
    axes[1].imshow(x_reconstructed_np, cmap='viridis')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    # Set title for the entire figure, specifying update count and index
    fig.suptitle(f"Comparison of Original and Reconstructed Image at update {i}, index {idx}, {'Amplitude' if type == 0 else 'Phase'}")

    # Adjust layout
    plt.tight_layout()

    # Determine save directory
    folder = 'amplitude_image' if type == 0 else 'phase_image'
    save_dir = f'./images_32/{folder}'

    # Create the folder if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the image to the specified directory
    save_path = os.path.join(save_dir, f'comparison_update_{i}_index_{idx}.png')
    plt.savefig(save_path)

    # Close the figure to prevent memory leaks
    plt.close()

    # Return the save path
    return save_path

def ssim_loss(x, y, C1=1e-4, C2=9e-4):
    # Expand the convolution kernel to match the number of channels in the input
    channels = x.size(1)  # Get the number of channels in the input
    window = torch.ones(channels, 1, 11, 11).to(x.device) / 121  # Adjust the kernel to match the number of channels

    # Calculate the mean
    mu_x = F.conv2d(x, window, groups=channels, padding=5)
    mu_y = F.conv2d(y, window, groups=channels, padding=5)

    # Calculate variance and covariance
    sigma_x = F.conv2d(x * x, window, groups=channels, padding=5) - mu_x * mu_x
    sigma_y = F.conv2d(y * y, window, groups=channels, padding=5) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, window, groups=channels, padding=5) - mu_x * mu_y

    # Calculate SSIM
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    # Return SSIM loss
    return 1 - ssim_map.mean()  # Loss: we want to minimize the SSIM value

def validate(idx):
    model.eval()  # Set model to evaluation mode
    total_recon_loss_sum = 0
    total_embedding_loss_sum = 0
    total_loss_sum = 0
    total_perplexity_sum = 0
    total_recon1_sum = 0
    total_recon2_sum = 0

    with torch.no_grad():  # No gradient computation
        for batch_idx, x in enumerate(validation_loader):
            x = x.to(device)
            amp = x[:, 0, :, :].unsqueeze(1)
            ph = x[:, 1, :, :].unsqueeze(1)

            # Forward pass to get x_hat
            ph_hat, amp1_embedding_loss, amp2_embedding_loss, amp1_perplexity, amp2_perplexity, recon_error1, recon_error2 = model(amp, ph)
            # Calculate loss
            # amp_recon_loss = torch.mean((amp_hat - amp) ** 2)
            amp_recon_loss = ssim_loss(ph, ph_hat)
            # amp_recon_loss = torch.mean((amp_hat - amp) ** 2) / amp_train_var
            # ph_recon_loss = torch.mean((ph_hat - ph) ** 2) / ph_train_var
            recon_loss = amp_recon_loss
            embedding_loss = amp1_embedding_loss + amp2_embedding_loss
            loss = recon_loss + embedding_loss
            perplexity = amp1_perplexity + amp2_perplexity

            # Accumulate batch losses
            total_recon_loss_sum += recon_loss.item()
            total_embedding_loss_sum += embedding_loss.item()
            total_loss_sum += loss.item()
            total_perplexity_sum += perplexity.item()
            total_recon1_sum += recon_error1.item()
            total_recon2_sum += recon_error2.item()

        # Calculate average loss
        avg_recon_loss = total_recon_loss_sum / len(validation_loader)
        avg_embedding_loss = total_embedding_loss_sum / len(validation_loader)
        avg_loss = total_loss_sum / len(validation_loader)
        avg_perplexity = total_perplexity_sum / len(validation_loader)
        avg_recon1 = total_recon1_sum / len(training_loader)
        avg_recon2 = total_recon2_sum / len(training_loader)

    print(f'Validation | Ave Recon Error: {avg_recon_loss:.6f} | Ave Loss: {avg_loss:.6f} | Ave Perplexity: {avg_perplexity:.6f}')

    # Write validation results to TensorBoard
    writer.add_scalar('Validation/Loss', avg_loss, idx)
    writer.add_scalar('Validation/Reconstruction_Error', avg_recon_loss, idx)
    writer.add_scalar('Validation/Embedding_Loss', avg_embedding_loss, idx)
    writer.add_scalar('Validation/Perplexity', avg_perplexity, idx)
    writer.add_scalar('Validation/average_recon1_M', avg_recon1, idx)
    writer.add_scalar('Validation/average_recon2_1-M', avg_recon2, idx)

    model.train()  # Set model back to training mode
    return avg_loss

def train():
    for i in range(args.n_updates):
        count = 0
        idx = 0
        temp_x = [[[]]]
        temp_x_hat = [[[]]]
        total_recon_loss_sum = 0
        total_embedding_loss_sum = 0
        total_loss_sum = 0
        total_perplexity_sum = 0
        total_recon1_sum = 0
        total_recon2_sum = 0
        data_iter = iter(training_loader)  # Reinitialize the iterator for each training update
        for batch_idx in range(len(training_loader)):
            x = next(data_iter)

            x = x.to(device)
            amp = x[:, 0, :, :]
            amp = amp.unsqueeze(1)
            # print(amp.shape)
            ph = x[:, 1, :, :]
            ph = ph.unsqueeze(1)

            # Forward pass to get x_hat
            optimizer.zero_grad()
            # flops, params = thop.profile(model, inputs=(amp, ph))
            # print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
            # print("------|-----------|------")
            # print("%s | %.7f | %.7f" % ("Model  ", params / (1000 ** 2), flops / (1000 ** 3)))
            amp_hat, amp1_embedding_loss, amp2_embedding_loss, amp1_perplexity, amp2_perplexity, recon_loss1, recon_loss2 = model(amp, ph)
            # embedding_loss, x_hat, perplexity = model(x)
            # print(x_hat.shape)

            # print(f"x shape: {x.shape}")
            # print(f"x_hat shape: {x_hat.shape}")
            # Calculate reconstruction loss
            # amp_recon_loss = torch.mean((amp_hat - amp) ** 2)
            ssim_loss_temp = ssim_loss(ph, amp_hat)
            # alpha = 0.8  # SSIM loss weight
            # beta = 0.2  # MSE loss weight
            # recon_loss = alpha * ssim_loss_temp + beta * torch.mean((amp_hat - ph) ** 2)
            recon_loss = ssim_loss_temp
            embedding_loss = amp1_embedding_loss + amp2_embedding_loss
            loss = recon_loss + embedding_loss
            perplexity = amp1_perplexity + amp2_perplexity

            # x_hat = torch.cat((amp_hat, ph_hat), dim=1)

            loss.backward()
            optimizer.step()

            # Accumulate batch losses
            total_recon_loss_sum += recon_loss.item()
            total_embedding_loss_sum += embedding_loss.item()
            total_loss_sum += loss.item()
            total_perplexity_sum += perplexity.item()
            total_recon1_sum += recon_loss1.item()
            total_recon2_sum += recon_loss2.item()

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["n_updates"] = i

            # Write loss values to TensorBoard
            if batch_idx == len(training_loader) - 1:
                # Calculate the average loss for all batches in the current update
                avg_recon_loss = total_recon_loss_sum / len(training_loader)
                avg_embedding_loss = total_embedding_loss_sum / len(training_loader)
                avg_loss = total_loss_sum / len(training_loader)
                avg_perplexity = total_perplexity_sum / len(training_loader)
                avg_recon1 = total_recon1_sum / len(training_loader)
                avg_recon2 = total_recon2_sum / len(training_loader)

                # Log average loss values to TensorBoard
                writer.add_scalar('Loss/average_total_loss', avg_loss, i)
                writer.add_scalar('Loss/average_reconstruction_loss', avg_recon_loss, i)
                writer.add_scalar('Loss/average_embedding_loss', avg_embedding_loss, i)
                writer.add_scalar('Perplexity/average_perplexity', avg_perplexity, i)
                writer.add_scalar('Loss/average_recon1_M', avg_recon1, i)
                writer.add_scalar('Loss/average_recon2_1-M', avg_recon2, i)
                # Log average losses for amp and ph

                # Print average loss information for the current update
                print(f'Update #{i} | Ave Recon Error: {avg_recon_loss:.6f} | Ave Loss: {avg_loss:.6f} | Ave Perplexity: {avg_perplexity:.6f}')

            # if i % args.log_interval == 0:

            print('Update #', i, 'Loader #', count, 'Recon Error:', np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))
            count += 1
            idx = np.random.randint(x.size(0))
            temp_x = x[idx]
            temp_x_hat = amp_hat[idx]

        if i % args.log_interval == 0:
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(model, results, hyperparameters, timestamp)
            plot_and_save_image_comparison(temp_x, temp_x_hat, i, idx, 1)
            # plot_and_save_image_comparison(temp_x, temp_x_hat, i, idx, 0)

        if i % 10 == 0:
            loss = validate(i)
            scheduler.step(loss)

    print("Training Completed")


train()

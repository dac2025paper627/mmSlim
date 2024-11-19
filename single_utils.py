import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Gait20Data(Dataset):
    def __init__(self, amplitude_dir, phase_dir, train=True, trainset_ratio=0.75):
        self.amplitude_dir = amplitude_dir
        self.phase_dir = phase_dir
        self.list_all_raw = self.load_data()
        self.prepare_dataset(train, trainset_ratio)

    def load_data(self):
        # Get all file names in the directory
        amplitude_files = sorted([f for f in os.listdir(self.amplitude_dir) if 'pc_ti_kinect_key' in f])
        phase_files = sorted([f for f in os.listdir(self.phase_dir) if 'pc_ti_kinect_key' in f])
        # Assuming self.amplitude_dir and self.phase_dir are defined
        # amplitude_files = sorted([
        #     f for f in os.listdir(self.amplitude_dir)
        #     if any(f'pc_ti_kinect_key_{i}' in f for i in range(1, 6))
        # ])
        # phase_files = sorted([
        #     f for f in os.listdir(self.phase_dir)
        #     if any(f'pc_ti_kinect_key_{i}' in f for i in range(1, 6))
        # ])

        print(f"Number of amplitude files: {len(amplitude_files)}")
        print(f"Number of phase files: {len(phase_files)}")

        # Check if the number of files matches
        if len(amplitude_files) != len(phase_files):
            raise ValueError("Amplitude and Phase directories must have the same number of files.")

        # Initialize list to store data for each file
        list_all_raw = []


        # Iterate over each file, load the data, and reshape it
        for amplitude_file, phase_file in zip(amplitude_files, phase_files):
            # Load amplitude data
            amplitude_data = np.load(os.path.join(self.amplitude_dir, amplitude_file)).astype(np.float32)
            if amplitude_data.size != 224 * 64:
                raise ValueError(f"Amplitude file {amplitude_file} cannot be reshaped to (188, 64)")

            amplitude_data = amplitude_data.reshape(224, 64)  # Reshape to (224, 128)
            amplitude_data = normalize(amplitude_data)

            # Load phase data
            phase_data = np.load(os.path.join(self.phase_dir, phase_file)).astype(np.float32)

            if phase_data.size != 224 * 64:
                raise ValueError(f"Phase file {phase_file} cannot be reshaped to (188, 64)")

            phase_data = phase_data.reshape(224, 64)  # Reshape to (224, 128)
            phase_data = normalize(phase_data)

            # Combine amplitude and phase data
            combined_data = np.stack((amplitude_data, phase_data), axis=0)  # Shape (2, 224, 128)

            # Normalize
            # combined_data = (combined_data - np.min(combined_data)) / (np.max(combined_data) - np.min(combined_data))

            list_all_raw.append(combined_data)

        # Convert list to numpy array of shape (num_files, 2, 224, 128)
        list_all_raw = np.array(list_all_raw)

        print(list_all_raw.shape)
        print("### Data load end ###")
        print("Shape of raw <#files, 2, 224, 64>:", list_all_raw.shape)

        return list_all_raw

    def prepare_dataset(self, train, trainset_ratio):
        self.shuffle()

        self.data_raw = []
        self.data_label = []

        # Split dataset into training and validation sets
        dataset_size = len(self.list_all_raw)
        train_size = int(trainset_ratio * dataset_size)
        val_size = dataset_size - train_size

        if train:
            self.data_raw = self.list_all_raw[:train_size]
        else:
            self.data_raw = self.list_all_raw[train_size:]

    def shuffle(self):
        random_state = np.random.RandomState(1)
        random_state.shuffle(self.list_all_raw)

    def __getitem__(self, index):
        return self.data_raw[index]

    def __len__(self):
        return len(self.data_raw)


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def normalize_per_channel(data):
    # Get the two channels
    amplitude_data = data[:, 0, :, :]  # First channel data
    phase_data = data[:, 1, :, :]  # Second channel data

    # Normalize the first channel data
    amplitude_min, amplitude_max = np.min(amplitude_data), np.max(amplitude_data)
    amplitude_normalized = (amplitude_data - amplitude_min) / (amplitude_max - amplitude_min)

    # Normalize the second channel data (0 to pi)
    # phase_min, phase_max = 0, np.pi
    phase_min, phase_max = np.min(phase_data), np.max(phase_data)
    phase_normalized = (phase_data - phase_min) / (phase_max - phase_min)

    # Combine the normalized data back
    data[:, 0, :, :] = amplitude_normalized
    data[:, 1, :, :] = phase_normalized

    return data


def load_data_and_data_loaders(dataset_name, batch_size, amplitude_dir, phase_dir, save_npy_files=False):
    if dataset_name == 'CUSTOM':
        train_dataset = Gait20Data(amplitude_dir, phase_dir, train=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)
        val_dataset = Gait20Data(amplitude_dir, phase_dir, train=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=16)
    else:
        raise ValueError('Invalid dataset name: only CUSTOM dataset is supported.')

    return train_dataset, val_dataset, train_loader, val_loader


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = "./results"

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save, SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + "_end_to_end" + "_dual_channel" + '.pth')

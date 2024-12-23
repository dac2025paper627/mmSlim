# mmSlim.py

## Overview

`mmSlim.py` is the central module of this project, combining the functionality of two **VQVAE (Vector Quantized Variational AutoEncoders)** and a **MaskNet** to create a robust machine learning pipeline. This module serves as the core component of the project, orchestrating operations between these models to achieve data encoding, quantization, masking, and decoding.

---

## Key Components

### 1. First VQVAE
The first VQVAE model is constructed using the following components:
- **Encoder**: Encodes the input data into a continuous latent representation.
- **Decoder**: Decodes the discrete latent representation back to the original data space.
- **Quantizer**: Converts the continuous latent space into discrete representations using vector quantization.
- **Residual**: Implements residual connections to enhance the model's learning capability.

### 2. Second VQVAE
The second VQVAE model uses the following analogous components:
- **Encoder2**: A variant of the encoder specific to the second VQVAE.
- **Decoder2**: A variant of the decoder for the second VQVAE.
- **Quantizer**: Shares functionality with the first VQVAE.
- **Residual**: Common residual connections for efficient learning.

### 3. MaskNet
The **MaskNet** module is responsible for dynamically masking the input data, splitting it into two distinct parts:
- One part is fed into the first VQVAE.
- The other part is fed into the second VQVAE.

This masking mechanism allows the `mmSlim.py` module to handle data in a unique and structured manner, enabling efficient and adaptive processing.

---

## Training and Testing Details

`mmSlim` is constructed on **PyTorch 1.8.1**, and the training and testing are conducted on an **Nvidia GeForce RTX 3090 GPU** with **24GB of memory**. The following settings are used:

1. **Training Settings**:
   - Training consists of **2000 epochs**.
   - **Batch size**: 32
   - **Learning rate**: 1e-5
   - **Optimizer**: Adam (used to update model gradients).

2. **Compression Settings**:
   - **Amplitude Compression**:
     - The `$M$ branch` selects a cookbook rate of 16 by setting the encoder output size to (56, 32).
     - The `$1-M$ branch` selects a cookbook rate of 248 by setting the encoder output size to (14, 8).
   - **Phase Compression**:
     - The same compression settings as the amplitude compression stage are adopted.

---

## Current Project Status

- **Uploaded**: Model definitions for both VQVAEs (`encoder.py`, `decoder.py`, `encoder2.py`, `decoder2.py`, `quantizer.py`, `residual.py`) and `MaskNet` (`masknet.py`).
- **Not Yet Uploaded**: Training scripts for `mmSlim.py` and examples of how to train or use the model in practice.

---

## Planned Functionality

Once training scripts are added, the `mmSlim.py` module will enable:
1. **Data Preprocessing**:
   - Input data will be processed through `MaskNet` to generate two distinct masked outputs.

2. **Parallel Encoding and Decoding**:
   - The first masked output will pass through the first VQVAE.
   - The second masked output will pass through the second VQVAE.

3. **Model Outputs**:
   - The module will combine the outputs of both VQVAEs to reconstruct the original data or achieve the desired task.

---


### Description of Files and Folders

- **`dataset/`**: Contains the input data for the model.
  - `amplitude/`: Includes compressed files (`part_1.zip`) containing amplitude-related data.
  - `phase/`: Includes compressed files (`part_1.zip`) containing phase-related data.

- **`models/`**: Folder for model-related definitions, such as VQVAE, MaskNet, and other custom modules.

- **`README.md`**: This documentation file.

- **`mmSlim_train.py`**: Core training script integrating all components of the pipeline.

- **`requirements.txt`**: Lists the dependencies required to run this project.

- **`single_utils.py`**: Contains utility functions for data preprocessing, loading, and augmentation.

---

## How to Use

Here’s the revised version of the **How to Use** section based on your requirements:
## How to Use

### Step 1: Install Python and Required Libraries
1. **Install Python**:
   Ensure you have Python 3.7 or 3.8 installed. If not, download and install it from [Python's official website](https://www.python.org/).

2. **Install Dependencies**:
   After installing Python, install the required libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 2: Prepare the Dataset
1. **Unzip the Dataset**:
   Unzip the files in `dataset/amplitude/` and `dataset/phase/` directories. This step prepares the amplitude and phase data for training.

   ```bash
   unzip dataset/amplitude/part_1.zip -d dataset/amplitude/
   unzip dataset/phase/part_1.zip -d dataset/phase/
   ```

2. **Verify Dataset Structure**:
   Ensure that the unzipped files are placed correctly in their respective folders (`dataset/amplitude/` and `dataset/phase/`).

---

### Step 3: Train Amplitude Compression Model and Save Parameters
1. **Modify `mmSlim_train.py`**:
   Open the `mmSlim_train.py` file and adjust the training pipeline to focus on **amplitude compression**. 
   - Ensure that the model processes the amplitude data (`dataset/amplitude/`).
   - Update the training loop to specifically train the **MaskNet** and related components for amplitude compression.

2. **Save the Model Parameters**:
   Our mmSlim_train.py code will automatically save the trained model parameters for the amplitude compression model in results folder. The saved pth file may
   be named as SAVE_MODEL_PATH + '/vqvae_data_' + timestamp+"end_to_end"+"_dual_channel" + '.pth', which is completed in the save_model_and_results method in    
   single_utils.py:
   ```python
   def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = "./results"

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save, SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + "_end_to_end" + "_dual_channel" + '.pth')

   ```
   This will save the model parameters after the amplitude compression training is completed.

4. **Run the Training Script**:
   Execute the training script to train the amplitude compression model:
   ```bash
   python mmSlim_train.py
   ```

---

### Step 4: Train Phase Compression Model Using Saved Amplitude Parameters
1. **Load the Saved Amplitude Model Parameters**:
   Modify `mmSlim_train.py` to load the saved `MaskNet` parameters from the amplitude compression model. Add the model relading code snippet like the following    
   example :
   ```python
   # state_dict=torch.load('./results/vqvae_data_sun_nov_10_01_18_16_2024end_to_end_dual_channel.pth')
   # model.load_state_dict(state_dict['model'])
   ```

2. **Adjust the Code for Phase Compression**:
   Update the training pipeline to process **phase data** from the `dataset/phase/` folder. Ensure that the model's input and loss functions are adjusted to focus on phase compression.

3. **Train the Phase Compression Model**:
   Run the script to train the model for phase compression using the loaded `MaskNet` parameters from the amplitude compression model:
   ```bash
   python mmSlim_train.py
   ```

---

## Requirements


### Dependencies

The following dependencies are required to run the model:

- **Python**: 3.7.x or 3.8.x (tested on Python 3.8.20)
- **PyTorch**: 1.8.1
- **NumPy**: Compatible with the above Python and PyTorch versions

Ensure that your Python environment meets the required version constraints, as the project is not compatible with Python versions lower than 3.7 or higher than 3.8.

Additional dependencies may be listed in the `requirements.txt` file.

---

## Contribution

Contributions are welcome to enhance the functionality of `mmSlim.py`, add training scripts, or improve documentation. To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request for review.

---


## Code and Data Availability

The code for `mmSlim` have been uploaded to an anonymous GitHub repository.

---


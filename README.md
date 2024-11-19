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

## File Structure Context

The project includes the following files:

```
README.md
__init__.py
decoder.py
decoder2.py
encoder.py
encoder2.py
masknet.py
mmSlim.py
quantizer.py
residual.py
vqvae1.py
vqvae2.py
```

- **mmSlim.py**: Core module that integrates both VQVAEs and MaskNet.
- **MaskNet**: Dynamically masks the input data to split it for processing by the two VQVAEs.
- **First VQVAE**:
  - `encoder.py`
  - `decoder.py`
  - `quantizer.py`
  - `residual.py`
- **Second VQVAE**:
  - `encoder2.py`
  - `decoder2.py`
  - `quantizer.py` (shared with the first VQVAE)
  - `residual.py` (shared with the first VQVAE)

---

## How to Use

1. **Initialize the `mmSlim` Model**:
   Use the components of the `mmSlim` module to define the architecture in your script.

   Example:
   ```python
   from mmSlim import mmSlim

   model = mmSlim(h_dim=128, res_h_dim=32, n_res_layers=2, 
                  n_embeddings=512, embedding_dim=64, beta=0.25)
   ```

2. **Forward Pass**:
   Once initialized, the model can process input data through `MaskNet` and the two VQVAEs.

   Example:
   ```python
   input_data = torch.randn(32, 1, 224, 224)  # Example input
   output = model(input_data)
   ```

3. **Training**:
   Training scripts will be added to this repository later. These scripts will define the loss functions and optimization process.

---

## Requirements

The following dependencies are required to run the model:
- Python 3.8+
- PyTorch 1.8.1
- NumPy

Additional dependencies may be listed in the `requirements.txt` file.

---

## Contribution

Contributions are welcome to enhance the functionality of `mmSlim.py`, add training scripts, or improve documentation. To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request for review.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Code and Data Availability

The code and sample data for `mmSlim` have been uploaded to an anonymous GitHub repository.

---

Let me know if you need further adjustments or additional details!

# Diffusion Models (CFG and EMA)

This project implements a conditional denoising diffusion probabilistic model (DDPM) using a UNet architecture for generating high-quality images.

## Project Structure
- **`train.py`**: Main training script.
- **`utils.py`**: Utility functions for data loading, image saving, and logging.
- **`ddpm.py`**: Implements the Gaussian diffusion process.
- **`unet.py`**: Contains the UNet architecture with self-attention and other modules.
- **`ema.py`**: Implements an exponential moving average (EMA) for model parameter smoothing.

## Usage
1. Update the dataset path and training parameters in `train.py`:
   - `args.dataset_path`
   - `args.run_name`
   - `args.epoch`
   - `args.batch_size`
   - `args.img_size`
   - `args.lr`
2. Run the training script:
   ```
   python train.py
   ```
3. Check the logs, generated images, and saved models in the respective `runs/`, `results/`, and `models/` directories.

## Training Overview
- The training script loads the data, builds the UNet model, and applies the Gaussian diffusion process.
- Loss is computed using mean squared error (MSE), optimized with AdamW.
- EMA is used to maintain a stable version of the model.
- Conditional generation is supported via optional label inputs.

## CFG and EMA
- The implementation contain the **Classifier free guidance**(`CFG`) and **Exponential Moving Average**(`EMA`)

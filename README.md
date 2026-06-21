# DAS-Traffic-Denoising_tools

This repository contains the signal processing algorithms developed for my Bachelor's Thesis (TFG) at the University of Granada.

## Project Overview
The objective of this project is to implement an efficient denoising architecture for Distributed Acoustic Sensing (DAS) records, focusing on urban traffic monitoring.

## Key Components
- **f-x Wiener Filtering**: Predictive spatial filtering with directional (forward-backward) processing.
- **Wavelet Denoising**: Multiresolution analysis using Daubechies wavelets.
- **Preprocessing Architecture**: A novel strategy using Ricker wavelet injection to regularize trajectories and prevent signal leakage in real urban records.

## Requirements
To run these tools, you will need the following libraries:
- `numpy`
- `scipy`
- `pywt` (PyWavelets)

## Usage
Simply import the functions into your main script:
```python
from your_filename import apply_fx_wiener_denoise, wavelet_denoise

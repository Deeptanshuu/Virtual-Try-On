# DiffuseFit Virtual Try-On Application

A deep learning powered virtual try-on application that allows users to visualize clothing items on their own photos.

## Overview

This project implements a state-of-the-art virtual try-on system using diffusion models and computer vision techniques to generate realistic garment try-on images. Users can upload a person image, a clothing item, and optionally a background to see how the clothing would look when worn by the person.

## Features

- Upload or select person images and garment images
- Automatic body parsing and pose estimation
- Clothing description support for better fitting
- Custom background support
- Advanced settings to control the generation process
- Web interface with API endpoint support

## Technical Stack

- **Core Models**:
  - Stable Diffusion XL for image generation
  - DensePose for 3D human body mapping
  - OpenPose for human pose estimation
  - Human parsing for body segmentation

- **Frameworks**:
  - PyTorch for deep learning processing
  - Gradio for the web interface
  - Hugging Face Transformers for text and image encoders
  - Diffusers library for diffusion model pipeline

- **Pre-trained Models**:
  - Base model: `yisol/IDM-VTON`
  - Includes UNet, CLIP encoders, VAE components

## Installation

1. Create and activate a conda environment:
```
conda create -n VTON python=3.10
conda activate VTON
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the checkpoint files (not included in the repository).

## Usage

1. Activate the environment:
```
conda activate VTON
```

2. Run the application:
```
python app.py
```

3. Open the provided URL in your browser to access the interface

## Workflow

1. The system processes the input person image to identify body poses and segments
2. The garment image is analyzed and encoded
3. A diffusion model generates a realistic composite image showing the clothing on the person
4. Optional background processing can be applied to the final result

## Advanced Options

- **Denoising Steps**: Control the quality of the generated image (20-40)
- **Seed**: Set a specific random seed for reproducible results
- **Auto-masking**: Toggle automatic detection of areas to apply garments
- **Auto-cropping**: Automatically handle images with different aspect ratios

## API Usage

The application exposes an API endpoint that can be accessed programmatically:

```
curl -X POST http://localhost:7860/api/tryon -F "imgs=@person.jpg" -F "garm_img=@garment.jpg" -F "prompt=blue t-shirt" -F "is_checked=true"
```

## Project Structure

- `app.py`: Main application file with the Gradio interface
- `apply_net.py`: DensePose application
- `utils_mask.py`: Mask generation utilities
- `preprocess/`: Contains OpenPose and human parsing models
- `src/`: Core model implementations
- `configs/`: Configuration files for DensePose

## Credits

This project uses models and techniques from the following projects:
- Stable Diffusion XL
- DensePose
- OpenPose
- Human Parsing
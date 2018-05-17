# VehicleSuperRes

Image and Video Super-Resolution, specialized for vehicle and traffic view processing and performed by using Deep
Convolutional Neural Networks. The program is implemented using [PyTorch](https://pytorch.org/docs/stable/index.html)
with Python 3, and runs on [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit).

This repository contains a script to upscale with a 2x resolution a directory of images or videos. Though some trained
weights are provided, included are also scripts for training custom PyTorch weight files (.pth) and testing the model.

## References
- [waifu2x](https://github.com/nagadomi/waifu2x) by [nagadomi](https://github.com/nagadomi)

## Summary


## Installation
### Cloning the Repository
To download the program, simply clone the git repository.
```bash
git clone https://github.com/blitzingeagle/VehicleSuperRes.git
cd VehicleSuperRes
mkdir images output videos frames   # Default input and output directories for convert.py
```
### Python Packages
For your convenience, the Python packages used in these scripts have been organized into a text file. To set up these
dependencies run the following with `pip`. (Note: Depending on your Python installation, you may use `pip3` instead)
```bash
pip install -r requirements.txt
```

## Command Line Interface (CLI)
### Upscale Conversion
The convert tool supports a total of four different functions: image to image, image to video, video to image, and video
to video. The execution parameters are obtained by parsing command line arguments following `python convert.py`.

The convert tool is organized into two commands `image` and `video` signifying the type of the input data.

#### Image Input
```bash
python convert.py
```
Without specifying input parameters, the image batches are taken from the `images/` directory from the root folder. All
images should be under a subdirectory of the `images/` directory with the following topology:
```text
images/batch_dir_01/input_0001.jpg
images/batch_dir_01/input_0002.jpg
images/batch_dir_01/input_0003.bmp
    ...
images/batch_dir_02/random_file_name.png
    ...
images/random_batch_name/input_0123.png
```

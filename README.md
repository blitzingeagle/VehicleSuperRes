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
```
### Python Packages
For your convenience, the Python packages used in these scripts have been organized into a text file. To set up these
dependencies run the following with `pip`. (Note: Depending on your Python installation, you may use `pip3` instead)
```bash
pip install -r requirements.txt
```

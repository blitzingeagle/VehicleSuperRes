# VehicleSuperRes

Image and Video Super-Resolution, specialized for vehicle and traffic view processing and performed by using Deep
Convolutional Neural Networks. The program is implemented using [PyTorch](https://pytorch.org/docs/stable/index.html)
with Python 3, and runs on [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit).

This repository contains a script to upscale with a 2x resolution a directory of images or videos. Though some trained
weights are provided, included are also scripts for training custom PyTorch weight files (.pth) and testing the model.

## Table of contents
<!--ts-->
* [VehicleSuperRes](#vehiclesuperres)
* [Table of contents](#table-of-contents)
* [Summary](#summary)
* [Installation](#installation)
    * [Cloning the Repository](#cloning-the-repository)
    * [Python Packages](#python-packages)
* [Command Line Interface (CLI)](#command-line-interface-cli)
    * [Upscale Conversion](#upscale-conversion)
        * [Image Input](#image-input)
            * [Default image Command](#default-image-command)
            * [Custom input directory](#custom-input-directory)
            * [Custom output directory](#custom-output-directory)
            * [Custom file type](#custom-file-type)
            * [Video output from image](#video-output-from-image)
            * [Image converting examples](#image-converting-examples)
        * [Video Input](#video-input)
            * [Default video Command](#default-video-command)
            * [Video output from video](#video-output-from-video)
        * [Additional CLI Flags](#additional-cli-flags)
            * [Verbose](#verbose)
            * [Weights file](#weights-file)
* [References](#references)
<!--te-->

## Summary
This repository contains a set of tools to convert images and videos to a 2x upscale version and contains tools to train
and test custom weights. The Deep Convolutional Neural Network model was adapted from [waifu2x](#references), which
featured an implementation in Lua Torch7 with slight modifications to the neural net structure and retrained weights. 
The code in this repository was instead implemented in the newer PyTorch, which saw some deprecated functionality from 
Lua Torch7.

In particular the original network had this structure:
```lua
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> output]
  (1): nn.SpatialConvolutionMM(3 -> 16, 3x3)
  (2): nn.LeakyReLU(0.1)
  (3): nn.SpatialConvolutionMM(16 -> 32, 3x3)
  (4): nn.LeakyReLU(0.1)
  (5): nn.SpatialConvolutionMM(32 -> 64, 3x3)
  (6): nn.LeakyReLU(0.1)
  (7): nn.SpatialConvolutionMM(64 -> 128, 3x3)
  (8): nn.LeakyReLU(0.1)
  (9): nn.SpatialConvolutionMM(128 -> 128, 3x3)
  (10): nn.LeakyReLU(0.1)
  (11): nn.SpatialConvolutionMM(128 -> 256, 3x3)
  (12): nn.LeakyReLU(0.1)
  (13): nn.SpatialFullConvolution(256 -> 3, 4x4, 2,2, 3,3) without bias
  (14): nn.View(-1)
}
```
which was later modified to
```python
import torch.nn as nn

nn.Sequential(
    nn.Conv2d(3, 16, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.ConvTranspose2d(256, 3, (4, 4), (2, 2), (1, 1), (0, 0), bias=False),
    nn.Sigmoid()
)
```
The padding parameters in the deconvolution layer were altered to guarantee the output image size to be twice the size
of the input. To preserve the organizational structure of the output neurons, the output to the deconvolution was no
longer flattened. Instead, a `Sigmoid` filter was used to ensure that the output neurons contained normalized values
between 0 and 1.


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
##### Default image Command
```bash
python convert.py image
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
By default, the results of the conversion tool are images. The images will retain their file names but exist under the
`output` directory. The inputs from above will produce the following:
```text
output/batch_dir_01/input_0001.jpg
output/batch_dir_01/input_0002.jpg
output/batch_dir_01/input_0003.bmp
    ...
output/batch_dir_02/random_file_name.png
    ...
output/random_batch_name/input_0123.png
```

##### Custom input directory
The `-i` flag can be used to specify a custom input directory.
```bash
python convert.py image -i path/to/images/
```

##### Custom output directory
The `-o` flag can be used to specify a custom output directory.
```bash
python convert.py image -o path/to/output/
```

##### Custom file type
To manipulate the file type for the output the `--ext` flag can be used. For images, the accepted file extensions are
`jpg`, `png`, and `bmp`.
```bash
python convert.py image --ext jpg
```

##### Video output from image
The conversion tool can produce a video format instead of images by also using the `--ext` flag with `avi`. The default
frame rate is `30 fps` but can be altered using the `-fps` flag.
```bash
python convert.py image --ext avi -fps 60
```
This will produce a video for each subdirectory of the `images/` directory, with the frames determined by the
lexicographical order of the image file names. As such, it is recommended that the image files be serialized in the
following manner to avoid frame mismatch.
```text
input_0000001.jpg
input_0000002.jpg
input_0000003.jpg
    ...
input_0001234.jpg
```
Note that the following arrangement would likely cause undesired behaviour.
```text
input_1.jpg
input_2.jpg
input_3.jpg
    ...
input_10.jpg
input_11.jpg
    ...
```
The videos will be stored in the `output/` directory with the name of the directory containing the frames as the base
name of the video file.

##### Image Converting Examples
```bash
# Upscales images from myimagedir/ to myoutputdir/ with png extension
python convert.py image -i myimagedir/ -o myoutputdir/ --ext png

# Upscales images to 30fps video from myimagedir/ to in myoutputdir/ with avi extension
python convert.py image -i myimagedir/ -o myoutputdir/ --ext avi

# Upscales images to 60fps video from myimagedir/ to in myoutputdir/ with avi extension
python convert.py image -i myimagedir/ -o myoutputdir/ --ext avi -fps 60
```

#### Video Input
##### Default video Command
```bash
python convert.py video
```
Without specifying input parameters, the video is taken from the `videos/input.avi` from the root folder. By default, 
the output would be placed in the `frames/` directory. A subdirectory will be created for the `avi` video file using 
the file's base name. Its contents will be `jpg` images for each of the video's frames in the following format:
```text
frame0000001.jpg
frame0000002.jpg
frame0000003.jpg
    ...
```
Note that the numbers are 1 indexed. The conversion tool will also produce a `file_list.txt` file containing, in order,
the list of the frames. The result should be identical to the result of `ls -1 frame*` so be cautious of putting other 
files in the same directory.

The command flags for the `video` command is largely identical to those for `image`. The
[input](#custom-input-directory), [output](#custom-output-directory), and [file type](#custom-file-type) follow the same
protocol.

##### Video output from video
The conversion tool can produce a video format instead of images by using the `--ext` flag with `avi`. The default frame
rate matches the input file but can be altered using the `-fps` flag.
```bash
python convert.py image --ext avi -fps 60
```


#### Additional CLI Flags
##### Verbose
To view logging outputs use the `-v` flag. This will print additional logging information.
```bash
python convert.py image -v
```

##### Weights file
To specify a custom weights file, use the `w` flag followed by the path to the weights file.
```bash
python convert.py image -w path/to/myweights.pth
```


## References
- [waifu2x](https://github.com/nagadomi/waifu2x) by [nagadomi](https://github.com/nagadomi)

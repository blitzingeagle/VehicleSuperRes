import argparse
import os
from os.path import dirname, splitext

import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils

from time import time


# CLI parser
parser = argparse.ArgumentParser(description="Vehicle Super Resolution Converter")
subparsers = parser.add_subparsers(help="commands")

image_parser = subparsers.add_parser("image", help="Convert Image")
image_parser.add_argument("-i", type=str, default="./images", metavar="INPUT", help="Input image batch directory (default: ./images)")
image_parser.add_argument("-o", type=str, default="./output", metavar="OUTPUT", help="Output image directory (default: ./output)")
image_parser.add_argument("-v", "--verbose", action="store_true")
image_parser.add_argument("-w", type=str, default="weights-beta.pth", metavar="WEIGHTS", help="Path to weights file (default: weights-beta.pth)")
image_parser.add_argument("--ext", type=str, default="(keep)", metavar="ext", help="File extension for output (default: <uses the same extension as input>)")

video_parser = subparsers.add_parser("video", help="Convert Video")
video_parser.add_argument("-i", type=str, default="./videos", metavar="INPUT", help="Input video directory (default: ./videos)")
video_parser.add_argument("-o", type=str, default="./frames", metavar="OUTPUT", help="Output image directory (default: ./frames)")
video_parser.add_argument("-v", "--verbose", action="store_true")
video_parser.add_argument("-w", type=str, default="weights-beta.pth", metavar="WEIGHTS", help="Path to weights file (default: weights-beta.pth)")
video_parser.add_argument("--ext", type=str, default="(keep)", metavar="ext", help="File extension for output (default: <uses the same extension as input>)")


# Model Setup
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda")

model = nn.Sequential(
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
    nn.ConvTranspose2d(256, 3, (4, 4), (2, 2), (1, 1), (0, 0), bias=False)
).to(device)


# Time logging functions
def tic():
    global args, start_time
    if args.verbose: start_time = time()

def toc(log):
    global args, start_time
    if args.verbose: print(log.format(time() - start_time))


def image():
    # Load weights
    tic()
    weights = torch.load(args.w)
    model.load_state_dict(weights)
    toc("Loaded weights in {:6f} seconds.")

    # Load inputs
    tic()
    dataset = datasets.ImageFolder(root=args.i, transform=transforms.ToTensor())
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    toc("Prepared dataset in {:6f} seconds.")

    model.eval()
    with torch.no_grad():
        for idx, x in enumerate(dataset_loader):
            filepath = dataset.samples[idx][0].replace(args.i, args.o)
            if args.ext is not "(keep)": filepath = splitext(filepath)[0] + (
                "." if args.ext[0] != "." else "") + args.ext
            directory = dirname(filepath)

            tic()
            input = x[0].cuda()
            output = model(input)
            print("{}:\t{} {} --> {} {}".format(idx, dataset.samples[idx][0], tuple(input.shape), filepath,
                                                tuple(output.shape)))
            toc("Conversion time: {:06f} seconds.")

            tic()
            if not os.path.exists(directory): os.makedirs(directory)
            utils.save_image(input, filepath)
            toc("Saved image: {:06f} seconds.")

def video():
    # Load weights
    tic()
    weights = torch.load(args.w)
    model.load_state_dict(weights)
    toc("Loaded weights in {:6f} seconds.")

    # Load inputs


# Main
if __name__ == "__main__":
    global args
    args = parser.parse_args()

    if args.which is "image":
        image()
    elif args.which is "video":
        video()


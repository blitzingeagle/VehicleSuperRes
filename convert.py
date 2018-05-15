import argparse
import os
from os.path import dirname, splitext

import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils

from time import time


# CLI parser
parser = argparse.ArgumentParser(description="Vehicle Super Resolution Converter")
parser.add_argument("-i", type=str, default="./images", metavar="I", help="Input image batch directory (default: ./images)")
parser.add_argument("-o", type=str, default="./output", metavar="O", help="Output image directory (default: ./output)")
parser.add_argument("-w", type=str, default="weights-beta.pth", metavar="W", help="Path to weights file (default: weights-beta.pth)")
parser.add_argument("--ext", type=str, default="(keep)", metavar="ext", help="File extension for output (default: <uses the same extension as input>)")
parser.add_argument("-v", "--verbose", action="store_true")


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


def main():
    global args
    args = parser.parse_args()

    # Load weights
    tic()
    weights = torch.load(args.w)
    model.load_state_dict(weights)
    toc("Loaded weights in {:6f} seconds.")

    # Load inputs
    tic()
    data_transform = transforms.ToTensor()
    dataset = datasets.ImageFolder(root=args.i, transform=data_transform)
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

# Main
if __name__ == "__main__":
    main()

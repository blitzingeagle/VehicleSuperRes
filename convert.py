import argparse
import cv2
from glob import glob
import numpy as np
import os
import os.path as path
from time import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils


# CLI parser
parser = argparse.ArgumentParser(description="Vehicle Super Resolution Converter")
subparsers = parser.add_subparsers(help="commands")

image_parser = subparsers.add_parser("image", help="Convert Image")
image_parser.add_argument("-i", type=str, default="./images", metavar="INPUT", help="Input image batch directory (default: ./images)")
image_parser.add_argument("-o", type=str, default="./output", metavar="OUTPUT", help="Output image directory (default: ./output)")
image_parser.add_argument("-v", "--verbose", action="store_true")
image_parser.add_argument("-w", type=str, default="weights.pth", metavar="WEIGHTS", help="Path to weights file (default: weights-beta.pth)")
image_parser.add_argument("--ext", type=str, default="(keep)", metavar="ext", help="File extension for output (default: <uses the same extension as input>)")
image_parser.add_argument("-fps", type=float, default=30.0, metavar="FPS", help="Frame per second for video output (default: 30 FPS)")
image_parser.set_defaults(which="image")

video_parser = subparsers.add_parser("video", help="Convert Video")
video_parser.add_argument("-i", type=str, default="./videos", metavar="INPUT", help="Input video directory (default: ./videos)")
video_parser.add_argument("-o", type=str, default="./frames", metavar="OUTPUT", help="Output image directory (default: ./frames)")
video_parser.add_argument("-v", "--verbose", action="store_true")
video_parser.add_argument("-w", type=str, default="weights.pth", metavar="WEIGHTS", help="Path to weights file (default: weights-beta.pth)")
video_parser.add_argument("--ext", type=str, default="jpg", metavar="ext", help="File extension for output (default: <uses the same extension as input>)")
video_parser.set_defaults(which="video")


# Extensions
image_ext = ["jpg", "png", "bmp"]
video_ext = ["avi"]


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
    nn.ConvTranspose2d(256, 3, (4, 4), (2, 2), (1, 1), (0, 0), bias=False),
    nn.Sigmoid()
).to(device)

model.eval()


# Time logging functions
def tic():
    global args, start_time
    if args.verbose: start_time = time()

def toc(log):
    global args, start_time
    if args.verbose: print(log.format(time() - start_time))


# Execution for image command
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

    with torch.no_grad():
        if args.type == "video":
            videos = {}

        for idx, x in enumerate(dataset_loader):
            filepath = dataset.samples[idx][0].replace(args.i, args.o)
            if args.ext != "(keep)": filepath = path.splitext(filepath)[0] + "." + args.ext
            directory = path.dirname(filepath)
            if args.type == "image":
                if not os.path.exists(directory): os.makedirs(directory)

            tic()
            input = x[0].cuda()
            output = model(input)
            if args.verbose: print("{}:\t{} {} --> {} {}".format(idx, dataset.samples[idx][0], tuple(input.shape), filepath, tuple(output.shape)))
            toc("Conversion time: {:06f} seconds.")

            if args.type == "image":
                tic()
                utils.save_image(output, filepath)
                toc("Saved image: {:06f} seconds.")
            elif args.type == "video":
                tic()
                img = output.permute(0, 2, 3, 1).cpu().numpy()
                img = img.reshape(img.shape[1:])
                img = img[:, :, ::-1] * 255

                if directory not in videos:
                    height, width, channels = img.shape
                    videos[directory] = cv2.VideoWriter(directory + "." + args.ext, cv2.VideoWriter_fourcc(*"XVID"), args.fps, (width, height))
                videos[directory].write(np.uint8(img))
                toc("Saved frame: {:06f} seconds.")

        if args.type == "video":
            for video in videos.values():
                video.release()


# Execution for video command
def video():
    # Load weights
    tic()
    weights = torch.load(args.w)
    model.load_state_dict(weights)
    toc("Loaded weights in {:6f} seconds.")

    # Load inputs
    tic()
    videos = glob(path.join(args.i, "*.avi"))
    toc("Found %d video(s) in {:6f} seconds." % len(videos))

    with torch.no_grad():
        for video in videos:
            directory = path.splitext(video.replace(args.i, args.o))[0]
            if args.type == "video":
                filepath = directory + "." + args.ext
                directory = path.dirname(filepath)
            if not os.path.exists(directory): os.makedirs(directory)

            tic()
            cap = cv2.VideoCapture(video)
            toc("Loaded video %s in {:6f} seconds." % video)
            
            time_depth = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if args.type == "video":
                width = 2 * int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = 2 * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                video_out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

            for idx in range(1, time_depth+1):
                if args.type == "image":
                    filepath = path.join(directory, "frame{:07d}.".format(idx) + args.ext)

                tic()
                success, input = cap.read()
                toc("Read frame in {:6f} seconds.")

                if success:
                    tic()
                    input = input[:,:,::-1]
                    input = np.swapaxes(np.swapaxes(np.array(input, dtype=float), 0, 2), 1, 2) / 255.0
                    input = torch.from_numpy(input.reshape((1,) + input.shape)).float().cuda()
                    output = model(input)
                    if args.verbose:
                        if args.type == "image":
                            print("{0}/{1}:\t{2} {3} --> {4} {5}".format(idx, time_depth, video, tuple(input.shape), filepath, tuple(output.shape)))
                        elif args.type == "video":
                            print("{0}/{1}:\t{2} {3} --> {4} [{0}] {5}".format(idx, time_depth, video, tuple(input.shape), filepath, tuple(output.shape)))

                    toc("Conversion time: {:06f} seconds.")

                    if args.type == "image":
                        tic()
                        # cv2.imwrite(filepath, img * 255.0)
                        utils.save_image(output, filepath)
                        toc("Saved image: {:06f} seconds.")
                    elif args.type == "video":
                        tic()
                        img = output.permute(0, 2, 3, 1).cpu().numpy()
                        img = img.reshape(img.shape[1:])
                        img = img[:, :, ::-1] * 255.0
                        # print(img.shape)
                        # cv2.imshow("img", np.uint8(img))
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()
                        video_out.write(np.uint8(img))
                        toc("Saved frame: {:06f} seconds.")

            if args.type == "image":
                pwd = os.getcwd()
                os.chdir(directory)
                os.system("ls -1 frame* > file_list.txt")
                os.chdir(pwd)
            elif args.type == "video":
                video_out.release()




# Main
if __name__ == "__main__":
    global args
    args = parser.parse_args()
    args.ext = args.ext.replace('.', '')
    args.type = "image" if args.ext.lower() in image_ext else "video" if args.ext.lower() in video_ext else None

    if args.which is "image":
        if args.ext == "(keep)": args.type = "image"
        image()
    elif args.which is "video":
        video()


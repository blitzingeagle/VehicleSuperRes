import numpy as np
import cv2

import torch
import torch.nn as nn

import time

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

weights = torch.load("tmp/weights100.pth")
model.load_state_dict(weights)

def img_to_input(img):
    img = np.swapaxes(np.swapaxes(np.array(img, dtype=float), 0, 2), 1, 2) / 255.0
    shape = (1,) + img.shape
    return torch.from_numpy(img.reshape(shape)).float().cuda()

def output_to_img(output):
    img = output.cpu().detach().numpy()
    img = img.reshape(img.shape[1:])
    return np.swapaxes(np.swapaxes(np.array(img, dtype=float), 0, 2), 0, 1)

if __name__ == "__main__":
    model.eval()
    with torch.no_grad():
        reading_times = []
        convert_times = []
        total_tic = time.time()

        for i in range(1, 3393):
            tic = time.time()
            # img_in = cv2.imread("train/inputs/input_{:04d}.bmp".format(i))
            img_in = cv2.imread("images/street/{}.jpg".format(i))
            reading_time = (time.time() - tic) * 1000000

            if img_in is None:
                continue

            input_conv_tic = time.time()
            input = img_to_input(img_in)
            input_conv_time = (time.time() - tic) * 1000000

            tic = time.time()
            output = model(input)
            convert_time = (time.time() - tic) * 1000000

            output_conv_tic = time.time()
            img_out = output_to_img(output)
            cv2.imwrite("test1.png", img_out)
            output_conv_time = (time.time() - tic) * 1000000

            print("Reading: {:6f}μs\tConverting: {:6f}μs\tInput: {:6f}μs\tOutput: {:6f}μs".format(reading_time, convert_time, input_conv_time, output_conv_time))
            # print("{:6f},{:6f},{:6f},{:6f}".format(reading_time, input_conv_time, convert_time, output_conv_time))

            reading_times.append(reading_time)
            convert_times.append(convert_time)

        total_time = time.time() - total_tic
        reading_times = sum(reading_times) / 1000000
        convert_times = sum(convert_times) / 1000000
        print("Total Reading: {:6f}s\nTotal Converting: {:6f}s\nTotal: {:6f}s".format(reading_times, convert_times, total_time))


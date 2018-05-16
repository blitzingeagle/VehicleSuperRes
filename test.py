import numpy as np
import cv2

import torch
import torch.nn as nn

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

weights = torch.load("weights-beta.pth")
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
        # img_in = cv2.imread("2437.jpg")
        #
        # input = img_to_input(img_in)
        # output = model(input)
        #
        # img_out = output_to_img(output)
        #
        # cv2.imshow("input", img_in)
        # cv2.moveWindow("input", 0, 0)
        # cv2.imshow("output", img_out)
        # cv2.moveWindow("output", 500, 0)
        #
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        for i in range(1, 2579):
            img_in = cv2.imread("train/inputs/input_{:04d}.bmp".format(i))

            input = img_to_input(img_in)
            output = model(input)

            img_out = output_to_img(output)

            print(img_in.shape)
            print(input.shape)
            print(output.shape)
            print(img_out.shape)

            target = cv2.imread("train/outputs/output_{:04d}.png".format(i))

            cv2.imshow("input", img_in)
            cv2.moveWindow("input", 0, 0)
            cv2.imshow("output", img_out)
            cv2.moveWindow("output", 500, 0)
            cv2.imshow("target", target)
            cv2.moveWindow("target", 1000, 0)

            cv2.waitKey()
            cv2.destroyAllWindows()


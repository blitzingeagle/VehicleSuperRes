import numpy as np
import cv2

import torch
import torch.nn as nn

# nn.Sequential {
#   [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> output]
#   (1): nn.SpatialConvolutionMM(3 -> 32, 3x3)
#   (2): nn.LeakyReLU(0.1)
#   (3): nn.SpatialConvolutionMM(32 -> 32, 3x3)
#   (4): nn.LeakyReLU(0.1)
#   (5): nn.SpatialConvolutionMM(32 -> 64, 3x3)
#   (6): nn.LeakyReLU(0.1)
#   (7): nn.SpatialConvolutionMM(64 -> 64, 3x3)
#   (8): nn.LeakyReLU(0.1)
#   (9): nn.SpatialConvolutionMM(64 -> 128, 3x3)
#   (10): nn.LeakyReLU(0.1)
#   (11): nn.SpatialConvolutionMM(128 -> 128, 3x3)
#   (12): nn.LeakyReLU(0.1)
#   (13): nn.SpatialConvolutionMM(128 -> 3, 3x3)
#   (14): nn.View(-1)
# }

def img_to_input(img):
    img = np.swapaxes(np.swapaxes(np.array(img, dtype=float), 0, 2), 1, 2) / 255.0
    shape = (1,) + img.shape
    return torch.from_numpy(img.reshape(shape)).float().cuda()

def output_to_img(output):
    img = output.cpu().detach().numpy()
    img = img.reshape(img.shape[1:])
    return np.swapaxes(np.swapaxes(np.array(img, dtype=float), 0, 2), 0, 1)

torch.set_default_tensor_type("torch.cuda.FloatTensor")
device = torch.device("cuda")

model = nn.Sequential(
    nn.Conv2d(3, 32, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(128, 3, (3, 3), padding=(1, 1))
).to(device)

weights = torch.load("noise3_model.pth")
model.load_state_dict(weights)

model.eval()

img_in = cv2.imread("input_0010.bmp")
cv2.imshow("img_in", img_in)


img_in = img_in[:,:,::-1]

input = img_to_input(img_in)
output = model(input)

img_out = output_to_img(output)[:,:,::-1]
cv2.imshow("img_out", img_out)

cv2.imwrite("test_out.png", img_out * 255)

cv2.waitKey()
cv2.destroyAllWindows()

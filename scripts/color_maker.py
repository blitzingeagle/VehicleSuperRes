import numpy as np
import cv2

cnt = 2579

for r in range(0, 256, 32):
    for g in range(0, 256, 32):
        for b in range(0, 256, 32):
            img = np.array([[[b, g, r]]*32]*32)
            img2 = np.array([[[b, g, r]]*64]*64)

            cv2.imwrite("train/batches/batch_49/input_{:04d}.bmp".format(cnt), img)
            cv2.imwrite("train/batches/batch_49/output_{:04d}.png".format(cnt), img2)

            cnt += 1

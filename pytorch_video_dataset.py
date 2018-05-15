import numpy as np
import cv2

import os
import pickle
import glob

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torchvision import transforms, utils


class VideoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, rootDir, channels, timeDepth, xSize, ySize, mean, transform=None):
        """
		Args:
			clipsList (string): Path to the clipsList file with labels.
			rootDir (string): Directory with all the videoes.
			transform (callable, optional): Optional transform to be applied
				on a sample.
			channels: Number of channels of frames
			timeDepth: Number of frames to be loaded in a sample
			xSize, ySize: Dimensions of the frames
			mean: Mean valuse of the training set videos over each channel
		"""

        self.clipsList = glob.glob(rootDir + "/*.avi")
        self.rootDir = rootDir
        self.channels = channels
        self.timeDepth = timeDepth
        self.xSize = xSize
        self.ySize = ySize
        self.mean = mean
        self.transform = transform


    def __len__(self):
        return len(self.clipsList)

    def readVideo(self, videoFile):
        # Open the video file
        cap = cv2.VideoCapture(videoFile)
        # nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = torch.FloatTensor(self.channels, self.timeDepth, self.xSize, self.ySize)
        failedClip = False
        for f in range(self.timeDepth):

            ret, frame = cap.read()
            if ret:
                frame = torch.from_numpy(frame)
                # HWC2CHW
                frame = frame.permute(2, 0, 1)
                frames[:, f, :, :] = frame

            else:
                print("Skipped!")
                failedClip = True
                break

        for c in range(3):
            frames[c] -= self.mean[c]
        frames /= 255
        return frames, failedClip

    def __getitem__(self, idx):
        videoFile = os.path.join(self.rootDir, self.clipsList[idx][0])
        clip, failedClip = self.readVideo(videoFile)
        if self.transform:
            clip = self.transform(clip)
        sample = {'clip': clip, 'label': self.clipsList[idx][1], 'failedClip': failedClip}

        return sample

x = VideoDataset("videos", 3, 366, 1920, 1080, 0, transforms.ToTensor)
print("Hello ", x)

# cap.get(cv2.CAP_PROP_FRAME_COUNT)
# cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

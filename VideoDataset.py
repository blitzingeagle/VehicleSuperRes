import cv2

import os.path as path
from glob import glob

import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root, transform=None):
        self.video_list = glob(root + "/*.avi")
        self.video_dir = root
        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        clip, failedClip = self.readVideo(self.video_list[idx])
        if self.transform:
            clip = self.transform(clip)
        sample = {'clip': clip, 'failedClip': failedClip}

        return sample

    def readVideo(self, videoFile):
        # Open the video file
        cap = cv2.VideoCapture(videoFile)

        channels = 3
        time_depth = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = torch.FloatTensor(time_depth, channels, width, height)

        failedClip = False
        for f in range(time_depth):
            ret, frame = cap.read()
            if ret:
                frame = torch.from_numpy(frame)
                # HWC2CHW
                frame = frame.permute(2, 1, 0)
                frames[f] = frame
            else:
                print("Skipped!")
                failedClip = True
                break

        frames /= 255

        return frames, failedClip
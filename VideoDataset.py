import cv2

import os.path as path
from glob import glob

import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform):
        self.video_list = glob(video_dir + "/*.avi")
        self.video_dir = video_dir
        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        videoFile = path.join(self.rootDir, self.clipsList[idx][0])
        clip, failedClip = self.readVideo(videoFile)
        if self.transform:
            clip = self.transform(clip)
        sample = {'clip': clip, 'label': self.clipsList[idx][1], 'failedClip': failedClip}

        return sample

    def readVideo(self, videoFile):
        # Open the video file
        cap = cv2.VideoCapture(videoFile)

        channels = 3
        time_depth = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = torch.FloatTensor(channels, time_depth, width, height)

        failedClip = False
        for f in range(time_depth):
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

        frames /= 255
        return frames, failedClip
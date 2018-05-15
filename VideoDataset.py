from glob import glob

from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform):
        self.video_list = glob(video_dir + "/*.avi")
        self.video_dir = video_dir
        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        None

    def readVideo(self, videoFile):
        None
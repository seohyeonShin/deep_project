import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from model import CNNtoLSTM
from torchvision import transforms
from PIL import Image

# Remove label
class SignLanguageDataset(Dataset):
    def __init__(self, video_dir, labels, transform=None, visualize=False):
        """
        Args:
            video_dir (string): Directory with all the videos.
            labels (dict): A dictionary mapping video filenames to their respective label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.video_dir = video_dir
        self.labels = labels
        self.transform = transform
        self.visualize = visualize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_name = list(self.labels.keys())[idx]
        label = self.labels[video_name]['category_id']
        frames = self.load_frames(os.path.join(self.video_dir, video_name), idx)
        # if self.transform:
        #     frames = self.transform(frames)
        return frames, label

    def load_frames(self, video_path, idx):
        if self.visualize:
            os.makedirs(f'visualize/{str(idx).zfill(2)}', exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
        frames_to_sample = 10
        interval = np.linspace(30, 90, frames_to_sample+1, dtype=int)  # Uniformly spaced frame indices
        # interval = np.linspace(0, total_frames-1, frames_to_sample+1, dtype=int)  # Uniformly spaced frame indices

        frames = []
        frame_ids = set(interval)  # Convert to set for fast lookup

        current_frame, cnt = 0, 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_frame in frame_ids:
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if self.visualize:
                        frame.save(f'visualize/{str(idx).zfill(2)}/{str(cnt).zfill(2)}_{str(current_frame).zfill(3)}.png')
                        cnt += 1 
                    frame = self.transform(frame)  # Assuming 'transform' is already defined and includes ToTensor()
                    frames.append(frame)
                    if len(frames) == frames_to_sample:
                        break
                current_frame += 1
        finally:
            cap.release()

        frames = torch.stack(frames)  # Stack all the frame tensors
        return frames



if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the frame to the size expected by the CNN.
        transforms.ToTensor(),  # Convert the frame to a PyTorch tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor.
    ])

    # with open('/ailab_mat/dataset/signlanguage/sub10/info.json', 'r') as f:
    with open('/HDD/pv/heeseon/src/sign-language/dset/annotations/info_val.json', 'r', encoding='utf-8-sig') as f:
        labels = json.load(f)

    print('1')
    # sl = SignLanguageDataset('/ailab_mat/dataset/signlanguage/sub10', labels, transform, True)
    sl = SignLanguageDataset('/HDD/pv/heeseon/src/sign-language/dset/videos/val', labels, transform, True)
    print('2')
    print(sl.__len__())
    print(sl.__getitem__(0))

    model = CNNtoLSTM(num_classes=10)
    frames_1, label_1 = sl.__getitem__(1)
    out = model(frames_1.unsqueeze(0))

    frames_17, label_17 = sl.__getitem__(17)
    frames_30, label_30 = sl.__getitem__(30)


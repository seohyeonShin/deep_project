from .model import CNNtoLSTM

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from PIL import Image


class MTModel():
    def __init__(self, model_path):
        self.sample = 'data/sample.mp4'
        self.device = self.set_device()
        self.model = CNNtoLSTM(num_classes=10).to(self.device)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.sample_to_text = {
            0 : 'this is zero!',
            1 : 'this is one!',
            2 : 'this is two!',
            3 : 'this is three!',
            4 : 'this is four!',
            5 : 'this is five!',
            6 : 'this is six!',
            7 : 'this is seven!',
            8 : 'this is eight!',
            9 : 'this is nine!'
        }

    def load_frames(self, video_path):
        transform = transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the frame to the size expected by the CNN.
            transforms.ToTensor(),  # Convert the frame to a PyTorch tensor.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor.
        ])
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
        frames_to_sample = 10
        interval = np.linspace(30, 90, frames_to_sample + 1, dtype=int)  # Uniformly spaced frame indices
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
                    frame = transform(frame)  # Assuming 'transform' is already defined and includes ToTensor()
                    frames.append(frame)
                    if len(frames) == frames_to_sample:
                        break
                current_frame += 1
        finally:
            cap.release()

        frames = torch.stack(frames)  # Stack all the frame tensors
        return frames

    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # print('Device is cuda')
        else:
            device = torch.device('cpu')
            # print('Device is cpu')

        return device

    def infer(self, path):
        input = self.load_frames(path).to(self.device)
        outputs = self.model(input.unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        text = self.sample_to_text[predicted.item()]

        return text


if __name__ == '__main__':
    mt_model = MTModel()
    frame = mt_model.load_frames('data/sample.mp4')
    text = mt_model.infer(frame)
    print(text)
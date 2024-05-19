import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # 첫 번째 합성곱 층
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 두 번째 합성곱 층
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 세 번째 합성곱 층
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 네 번째 합성곱 층
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 플랫텐
        x = self.classifier(x)
        return x

class CNNtoLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNtoLSTM, self).__init__()
        # Using a pre-trained ResNet-50
        self.cnn = nn.Sequential(*list(models.resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1])
        # self.cnn = CustomCNN(num_classes=num_classes)
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=1, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # Flatten dimensions for CNN processing: combine batch and timesteps
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        # Reshape to put timesteps back
        r_out = c_out.view(batch_size, timesteps, -1)
        # Pass through LSTM
        lstm_out, _ = self.lstm(r_out)
        # Decode the hidden state of the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out

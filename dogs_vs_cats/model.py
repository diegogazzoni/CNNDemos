import torch.nn.functional as F
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() # input is 128x128
        self.conv = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, padding=1, stride=2), # out 64x64x6
            nn.ReLU(),
            nn.BatchNorm2d(6),

            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4, padding=1, stride=2), # out 32x32x12
            nn.ReLU(),
            nn.BatchNorm2d(12),

            nn.MaxPool2d( kernel_size=2, stride=2, padding=0) # out 16x16x12
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*16*12, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)

        return x



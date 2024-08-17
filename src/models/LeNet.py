import torch
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)  # 'same' padding equivalent
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)  # 16 channels, 4x4 spatial dimensions after pooling
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.output_layer = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # Forward pass
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.output_layer(x)
        return F.softmax(x, dim=1)  # Apply softmax for output layer

if __name__ == "__main__":
    # Build and summarize the model
    model = LeNet5()
    print("LeNet5 Model Summary:")
    print(model)

import torch
import torch.nn as nn

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)
    self.downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_channels)
    ) if stride != 1 or in_channels != out_channels else nn.Identity()

  def forward(self, x):
    identity = x
    identity = self.downsample(identity)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x += identity
    x = self.relu(x)
    x = self.dropout(x)
    return x
  

class SoundClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.con1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
    self.dropout = nn.Dropout(0.2)
    self.bn1 = nn.BatchNorm2d(16)
    self.layers = nn.Sequential(
        ResBlock(16, 16, 1),
        ResBlock(16, 32, 2),
        ResBlock(32, 32, 1),
        ResBlock(32, 32, 1),
        ResBlock(32, 64, 2),
        ResBlock(64, 64, 1),
        ResBlock(64, 64, 1),
        ResBlock(64, 128, 2)
        )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(128, 41)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.bn1(self.con1(x)))
    x = self.dropout(x)
    x = self.layers(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x
  


# class MLP(nn.Module):
#     def __init__(self, num_classes=41):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(64 * 2442, 512)  # Flatten the input size
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, num_classes)
        
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the tensor to [batch_size, 64*1024]
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
        
#         return x
class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.linear = nn.Linear(64 * 2442, 41)
  def forward(self, x):
    x = x.view(x.size(0), -1)  # x.size(0) is the batch size
    out = self.linear(x)
    return out
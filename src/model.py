# src/model.py  -- minimal, fast CNN classifier (drop-in)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallConvNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, width=32):
        super().__init__()
        # width controls capacity; set width=32 or 48 for small/medium
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(width*2)
        self.conv3 = nn.Conv2d(width*2, width*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(width*4)

        self.pool = nn.MaxPool2d(2,2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(width*4, num_classes)

        # small dropout for regularization
        self.dropout = nn.Dropout(0.2)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)               # /2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)               # /4
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Drop-in factory used by your train/eval code
def create_classification_model(num_classes, backbone_name='small'):
    """
    Returns a simple convnet. 'backbone_name' argument kept for compatibility.
    """
    # choose width by name (easy to tune)
    if backbone_name and 'tiny' in backbone_name:
        width = 16
    elif backbone_name and 'small' in backbone_name:
        width = 32
    elif backbone_name and 'med' in backbone_name:
        width = 48
    else:
        width = 32
    model = SmallConvNet(num_classes=num_classes, width=width)
    return model

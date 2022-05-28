import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34

resnet = resnet34(pretrained=True)

class Net1(nn.Module):
    # Net1 refers to LeNet
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

class Net2(nn.Module):
    # Net2 uses two 3*3 kernels instead of one 5*5 kernel
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, 3),
            nn.ReLU(),
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 6, 3),
            nn.ReLU(),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

class Net3(nn.Module):
    # Net3 adds more layers and parameters to see if overfit occurs
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 400),
            nn.ReLU(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x)
        return x

class ResNet_attend(nn.Module):
    # ResNet_attend simply uses a pretrained resnet34 network as the basic model, and then adds some layers
    def __init__(self) -> None:
        super().__init__()
        self.resize = torchvision.transforms.Resize(224)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        for p in self.parameters():
            p.requires_grad = False
        
        self.C13 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.C13(x)
        x = self.resnet(x)
        x = self.conv(x)
        x = self.fc(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1) -> None:
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.left(x)        
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    # this net work is used to learn the idea of residual links
    def __init__(self, in_dim=1, residualBlock=BasicBlock, n_classes=10) -> None:
        super().__init__()
        self.in_channel = 64
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_dim, self.in_channel, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )   # reshape the input to use the resnet framework
        self.layer1 = self._make_layer(residualBlock, 64, 2, 1)
        self.layer2 = self._make_layer(residualBlock, 128, 2, 2)
        self.layer3 = self._make_layer(residualBlock, 256, 2, 2)
        self.layer4 = self._make_layer(residualBlock, 512, 2, 2)
        self.avgpool = nn.AvgPool2d(4)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def _make_layer(self, block, channels, block_num, stride):
        layers = []
        layers.append(block(self.in_channel, channels, stride))
        self.in_channel = channels
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.preprocess(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

class Net4(nn.Module):
    # Net4 is based on the residual link idea
    def __init__(self) -> None:
        super().__init__()
        self.layer1_left = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6)
        )
        self.layer1_shortcut = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=1, stride=2),
            nn.BatchNorm2d(6)
        )
        self.layer2_left = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )
        self.layer2_shortcut = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=1, stride=2),
            nn.BatchNorm2d(16)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        out1 = self.layer1_left(x)
        out1 += self.layer1_shortcut(x)
        out1 = F.relu(out1)
        out2 = self.layer2_left(out1)
        out2 += self.layer2_shortcut(out1)
        out2 = F.relu(out2)
        out = self.flatten(out2)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    inputs = torch.randn(1, 1, 28, 28)
    net = Net4()
    outputs = net(inputs)
    print(outputs.shape)
    resnetatt = ResNet_attend()
    print(resnetatt(inputs).shape)
    inputs = torch.randn(1, 1, 224, 224)
    net = ResNet()
    print(net(inputs).shape)
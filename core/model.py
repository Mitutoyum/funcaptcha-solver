import torchvision
import torch

from torch import nn


class SiameseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = torchvision.models.resnet50(num_classes=1)
        self.sigmoid = nn.Sigmoid()

        in_features = self.resnet.fc.in_features
        out_features = self.resnet.fc.out_features

        # self.fc = nn.Linear(in_features * 2, out_features)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(in_features * 2, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, out_features),
        )

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)

        output = torch.cat((x1, x2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)

        return output

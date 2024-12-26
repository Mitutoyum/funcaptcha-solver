import torch
import numpy
import random

from torch.utils import data
from torchvision.datasets import ImageFolder


class Dataset(data.Dataset):
    def __init__(self, dataset: ImageFolder) -> None:
        super().__init__()
        self.dataset = dataset

        self.group = {}

        targets = numpy.array(self.dataset.targets)

        for idx in self.dataset.class_to_idx.values():
            self.group[idx] = numpy.where(targets == idx)[0]

    def __getitem__(self, idx):
        x1 = self.dataset[idx]

        if idx % 2 == 0:
            idx2 = numpy.random.choice(self.group[x1[1]])

            while idx == idx2:
                idx2 = numpy.random.choice(self.group[x1[1]])

            x2 = self.dataset[idx2]

            target = torch.tensor(1, dtype=torch.float)
        else:
            cls = list(self.dataset.class_to_idx.values())
            cls.pop(x1[1])

            x2 = self.dataset[numpy.random.choice(self.group[random.choice(cls)])]

            target = torch.tensor(0, dtype=torch.float)
        return x1, x2, target

    def __len__(self):
        return len(self.dataset.imgs)

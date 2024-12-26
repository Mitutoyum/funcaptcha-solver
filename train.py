import torch
import tqdm
import pathlib

from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from core.dataset import Dataset
from core.model import SiameseNetwork
from core import utils


def train(model, criterion, optimizer, dataset, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set = random_split(dataset, [0.8, 0.2])

    batch_size = 32

    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    weights = pathlib.Path("./weights") / "checkpoint.pt"
    last_epoch = 0

    if weights.exists():
        checkpoint = torch.load("weights/checkpoint.pt", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        last_epoch = checkpoint["epoch"]

    model.train()

    for epoch in range(last_epoch, epochs):
        try:
            train_correct = 0
            with tqdm.tqdm(train_loader, unit="batch") as tepoch:
                for batch_idx, (x1, x2, targets) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}/{epochs}")

                    x1[0], x2[0] = x1[0].to(device), x2[0].to(device)
                    x1[1], x2[1] = x1[1].to(device), x2[1].to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    output = model(x1[0], x2[0]).squeeze()

                    loss = criterion(output, targets)
                    loss.backward()

                    optimizer.step()

                    pred = torch.where(output > 0.5, 1, 0)
                    correct = pred.eq(targets.view_as(pred)).sum().item()
                    train_correct += correct

                    tepoch.set_postfix(
                        loss=loss.item(), accuracy=100.0 * correct / batch_size
                    )
        finally:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, "weights/checkpoint.pt")


if __name__ == "__main__":
    model = SiameseNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    dataset = Dataset(
        ImageFolder(
            "./dataset",
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )
    )
    train(model, criterion, optimizer, dataset, 10)
    torch.save(model.state_dict(), "weights/best.pt")

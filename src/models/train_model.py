import argparse
import os
import sys

import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# from data import mnist
from src.models.model import MyAwesomeModel
import wandb

# initialise wandb
wandb.init()

@click.group()
def cli():
    pass


class dataloader(Dataset):
    def __init__(self, dataset):
        self.images = dataset['images']
        self.labels = dataset['labels']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training using:", device)

    print("Training day and night")

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train_set, _ = mnist()
    train_set = torch.load(r"data/processed/train.pt")
    train = dataloader(train_set)
    trainloader = DataLoader(train, batch_size=64, shuffle=True)

    epochs = 20

    running_loss = []

    for e in range(epochs):

        train_loss = 0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # for plotting
        running_loss.append(train_loss)
        wandb.log({"train loss": train_loss})

        print(f"Epoch {e}   Training loss: {train_loss}")  # Val loss: {running_val_loss}  Val accuracy: {accuracy.item() * 100}%")

    # plt.plot(running_loss)
    # plt.xlabel("Epochs")
    # plt.ylabel("NLL loss")
    # plt.legend(["Train loss"])
    # plt.savefig(os.path.join(r"reports/figures", "training_curve.png"))
    # plt.show()

    # print("Saving model")
    # torch.save(model.state_dict(), os.path.join(r"models",'model.pt'))


# @click.command()
# @click.argument("model_checkpoint")
# def evaluate(model_checkpoint):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Training using:", device)
#
#     print("Evaluating until hitting the ceiling")
#     print(model_checkpoint)
#
#     # TODO: Implement evaluation logic here
#     model = MyAwesomeModel()
#     model.load_state_dict(torch.load('model.pt'))
#     model.eval()
#     _, test_set = mnist()
#     test = dataloader(test_set)
#     testloader = DataLoader(test, batch_size=64, shuffle=False)
#
#     criterion = nn.NLLLoss()
#
#     val_loss = 0
#     with torch.no_grad():
#         equals_list = []
#         for images, labels in testloader:
#             log_ps = model(images)
#             loss = criterion(log_ps, labels)
#
#             val_loss += loss.item()
#
#             top_p, top_class = torch.exp(log_ps).topk(1, dim=1)
#             equals = top_class == labels.view(*top_class.shape)
#
#             equals_list += equals.type(torch.FloatTensor)
#
#         accuracy = torch.mean(torch.tensor(equals_list))
#
#     print(f"Test loss: {val_loss}  Test accuracy: {accuracy.item() * 100}%")
#

cli.add_command(train)
# cli.add_command(evaluate)

if __name__ == "__main__":
    cli()






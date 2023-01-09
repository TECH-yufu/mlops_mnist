import argparse
import os
import sys

import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset

# from data import mnist
from src.models.model import MyAwesomeModel
from PIL import Image


@click.group()
def cli():
    pass

wandb.init()

class dataloader(Dataset):
    '''
    Class for PyTorch dataloader given to the

            Parameters:
                    dataset: a dict with keys 'images' and 'labels'
    '''
    def __init__(self, dataset):
        self.images = dataset['images']
        self.labels = dataset['labels']

    def __len__(self):
        '''
        Returns the length of the dataset

                Returns
                        length of dataset (int)
        '''
        return len(self.images)

    def __getitem__(self, idx):
        '''


                Parameters:
                        idx (int): index in the dataset.
                Returns:
                        image (tensor) and label (tensor)
        '''
        image = self.images[idx]
        label = self.labels[idx]

        return image, label



@click.command()
@click.argument("model_checkpoint")
@click.argument("test_path")
def evaluate(model_checkpoint, test_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training using:", device)

    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    test_set = torch.load(test_path)
    test = dataloader(test_set)
    testloader = DataLoader(test, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()

    val_loss = 0
    with torch.no_grad():
        equals_list = []
        for images, labels in testloader:
            log_ps = model(images)
            loss = criterion(log_ps, labels)

            val_loss += loss.item()

            # for the output log-probabilities, get the top class and the probability for that class
            top_p, top_class = torch.exp(log_ps).topk(1, dim=1)
            # check if prediction == label
            equals = top_class == labels.view(*top_class.shape)

            equals_list += equals.type(torch.FloatTensor)

        accuracy = torch.mean(torch.tensor(equals_list))

    print(f"Test loss: {val_loss}  Test accuracy: {accuracy.item() * 100}%")
    wandb.log({'Test loss': val_loss, 'Test accuracy': accuracy.item() * 100})



    print("Examples:")
    print("Labels:     ", labels[:10])
    print("Predictions:", top_class[:10].flatten())

    # I use the unique wandb run id to organize my artifacts
    test_data_at = wandb.Artifact("test_samples_" + str(wandb.run.id), type="predictions")

    columns = ["id","image", "label", "prediction"]
    my_table = wandb.Table(columns=columns)

    for i in range(8):
        img = wandb.Image(images[i,:,:], caption="Example")
        my_table.add_data(i, img, labels[i], top_class[i].flatten().item())



    # Log your Table to wandb
    test_data_at.add(my_table, "predictions")
    wandb.run.log_artifact(test_data_at)

cli.add_command(evaluate)

if __name__ == "__main__":
    cli()






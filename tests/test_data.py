import torch
from tests import _PATH_DATA
import os
import pytest


# train = torch.load(r"C:\Users\Yucheng\OneDrive - Danmarks Tekniske Universitet\DTU\HCAI\7. semester\MLOps\Day 2\MNIST\data\processed\train.pt")
# test = torch.load(r"C:\Users\Yucheng\OneDrive - Danmarks Tekniske Universitet\DTU\HCAI\7. semester\MLOps\Day 2\MNIST\data\processed\test.pt")


@pytest.mark.skipif(not os.path.exists('data/processed'), reason="Data files not found") # skip this test if dir data/processed does not exist
def test_data_length():
    train = torch.load(os.path.join(_PATH_DATA, 'processed/train.pt'))
    test = torch.load(os.path.join(_PATH_DATA, 'processed/test.pt'))
    assert len(train['images']) == 25000 and len(test['images']) == 5000, "Datasets did not have the correct number of samples"

def test_data_shape():
    train = torch.load(os.path.join(_PATH_DATA, 'processed/train.pt'))
    test = torch.load(os.path.join(_PATH_DATA, 'processed/test.pt'))
    assert torch.all(torch.tensor([i.unsqueeze(0).shape == (1,28,28) for i in train['images']])).item() == True and torch.all(torch.tensor([i.unsqueeze(0).shape == (1,28,28) for i in test['images']])).item() == True, "Samples in the datasets do not have the correct shape" # each datapoint has shape [1,28,28]

def test_data_labels():
    train = torch.load(os.path.join(_PATH_DATA, 'processed/train.pt'))
    test = torch.load(os.path.join(_PATH_DATA, 'processed/test.pt'))
    assert torch.all(train['labels'].unique() == torch.arange(0,10)) and torch.all(test['labels'].unique() == torch.arange(0,10)), "Not all labels are present in datasets" # all labels are represented

# parameterized decorator
train = torch.load(os.path.join(_PATH_DATA, 'processed/train.pt'))
test = torch.load(os.path.join(_PATH_DATA, 'processed/test.pt'))
@pytest.mark.parametrize("condition,expected", [([torch.all(torch.le(i, 3)).item() for i in train['images']], True),
                                                ([torch.all(torch.ge(i, -1)).item() for i in train['images']], True),
                                                ([torch.all(torch.le(i, 3)).item() for i in test['images']], True),
                                                ([torch.all(torch.ge(i, -1)).item() for i in test['images']], True)])
def test_data_values(condition, expected):
    assert torch.all(torch.tensor(condition)).item() == expected # check that all values in the train and test datasets are in the range [-1, 3]
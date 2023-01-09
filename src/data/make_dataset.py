# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

     # exchange with the corrupted mnist dataset
    d1 = np.load(os.path.join(input_filepath, 'train_0.npz'))

    train = {'images': torch.tensor(d1['images']), 'labels': torch.tensor(d1['labels'])}

    for i in range(1,5):
        d2 = np.load(os.path.join(input_filepath, 'train_{}.npz'.format(i)))
        train['images'] = torch.cat((train['images'], torch.tensor(d2['images'])))
        train['labels'] = torch.cat((train['labels'], torch.tensor(d2['labels'])))

    test_ = np.load(os.path.join(input_filepath, 'test.npz'))


    test = {'images': torch.tensor(test_['images']), 'labels': torch.tensor(test_['labels'])}

    train_ = {'images': (train['images'] - train['images'].mean()) / train['images'].std(), 'labels': train['labels']}
    test_ = {'images': (test['images'] - test['images'].mean()) / test['images'].std(), 'labels': test['labels']}

    torch.save(train_, os.path.join(output_filepath, 'train.pt'))
    torch.save(test_, os.path.join(output_filepath, 'test.pt'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

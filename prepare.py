"""
Downloading and processing IMDB dataset for language model
"""

import argparse
import os
import tarfile
import glob
import requests


parser = argparse.ArgumentParser(description='Preparation scrpit for downloading and concatenating text.')
parser.add_argument('--url', type=str, default='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                    help='link to dataset')
parser.add_argument('--save', type=str, default='data',
                    help='path to directory where train and test data would be saved')

args = parser.parse_args()

def download():
    """
    Download dataset from url given in command line arguments.
    """
    filename = os.path.join(args.save, 'aclImdb_v1.tar.gz')
    if not os.path.exists(filename):
        print('Downloading')
        with open(filename, 'wb') as f:
            r = requests.get(args.url)
            f.write(r.content)

    datadir = os.path.join(args.save, 'aclImdb')
    if not os.path.exists(datadir):
        print('Extracting archive')
        with tarfile.open(name=filename, mode='r:gz') as tar:
            tar.extractall(path=args.save)


def concat_reviews():
    """
    Go through all the reviews and concat them all in two files:
    train.txt and test.txt
    """
    datadir = os.path.join(args.save, 'aclImdb')
    if os.path.exists(datadir):
        return

    dirs = os.path.join(datadir, 'train'), os.path.join(datadir, 'test')
    out = os.path.join(args.save, 'train.txt'), os.path.join(args.save, 'test.txt')

    for directory, outfile in zip(dirs, out):
        print(f'Walking through {directory}')
        filenames = glob.glob(os.path.join(directory, '*/*.*'))
        print('\nFilenames:\n-----------')
        for filename in filenames[:5]:
            print(f'{filename},')
        print(f'...,\n{filenames[-1]}\n')
        with open(outfile, 'w') as wfd:
            for f in filenames:
                with open(f, 'r') as fd:
                    for line in fd:
                        wfd.write(line + '\n')


download()
concat_reviews()
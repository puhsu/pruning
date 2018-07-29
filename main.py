import argparse
import time
import math
import os
import pickle

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

import model
import utils
from utils import TextDataset, TextSampler, PadCollate, RunningAverage

parser = argparse.ArgumentParser(description='PyTorch IMDB LSTM classifier')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='data/models/model.pt',
                    help='path to save the final model')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Load data
###############################################################################

train_npz = np.load('data/train.npz')
test_npz = np.load('data/test.npz')

train_ds = TextDataset(x=train_npz['texts'], y=train_npz['labels'])
test_ds = TextDataset(x=test_npz['texts'], y=test_npz['labels'])

eval_batch_size = 10
train_sp = TextSampler(train_ds, key=lambda i: len(train_npz['texts'][0]), batch_size=args.batch_size)

train_dl = data.DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sp, collate_fn=PadCollate())
test_dl = data.DataLoader(test_ds, batch_size=eval_batch_size, shuffle=True, collate_fn=PadCollate())

with open('data/itos.pkl', 'rb') as f:
    itos = pickle.load(f)

print('Loaded train and test data.')


###############################################################################
# Create model
###############################################################################


ntokens = len(itos)
model = nn.Sequential(
    model.ClassifierRNN(args.bptt, ntokens, args.emsize, args.nhid),
    model.LinearDecoder(args.nhid, 2)
)

print(f'Created model with {utils.count_parameters(model)} parameters:')
print(model)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.8, 0.99))


###############################################################################
# Training code
###############################################################################

def evaluate(loader):
    """Calculates loss and prediction accuracy given torch dataloader"""
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            # run model
            inp, target = batch.to(device)
            out = model(inp.t())

            # calculate loss
            loss = criterion(out.view(-1), target.float())
            total_loss += loss.item()
            
            # calculate accuracy
            pred = out.view(-1) > 0.5
            correct = pred == target.byte()
            total_acc += torch.sum(correct).item() / len(correct)

    return total_loss / len(loader), total_acc / len(loader)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    avg_loss = RunningAverage()
    avg_acc = RunningAverage()

    pbar = tqdm(train_dl, ascii=True, leave=False)
    for batch in pbar:
        inp, target = batch.to(device)
        model.zero_grad()
        out = model(inp.t())
        loss = criterion(output.view(-1), targets.float())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # upgrade stats
        avg_loss.update(loss.item())
        pred = out.view(-1) > 0.5
        correct = pred == target.byte()
        avg_acc.update(torch.sum(correct).item() / len(correct))

        pbar.set_postfix(loss=f'{avg_loss():05.3f}', acc=f'{avg_acc():05.2f}')


###############################################################################
# Actual training
###############################################################################

# Loop over epochs.
lr = args.lr
best_val_loss = None


for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    val_loss, val_acc = evaluate(test_dl)
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {time.time()-epoch_start_time:5.2f}s '
          f'| valid loss {val_loss:05.3f} | valid acc {val_acc:04.2f}')
    print('-' * 89)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0

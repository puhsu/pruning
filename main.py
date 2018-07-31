import argparse
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

import model
import utils
from utils import PadCollate, RunningAverage, TextDataset, TextSampler


parser = argparse.ArgumentParser(description='PyTorch IMDB LSTM classifier')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=5,
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
x_train, y_train = train_npz['texts'], train_npz['labels']
test_npz = np.load('data/test.npz')
x_test, y_test = test_npz['texts'], test_npz['labels']

train_ds = TextDataset(x=x_train, y=y_train)
test_ds = TextDataset(x=x_test, y=y_test)

eval_batch_size = 10
train_sp = TextSampler(train_ds, key=lambda i: len(x_train[i]), batch_size=args.batch_size)

train_dl = data.DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sp, collate_fn=PadCollate(max_len=200))
test_dl = data.DataLoader(test_ds, batch_size=eval_batch_size, shuffle=True, collate_fn=PadCollate(max_len=200))

with open('data/itos.pkl', 'rb') as f:
    itos = pickle.load(f)

print('Loaded train and test data.')


###############################################################################
# Create model
###############################################################################


ntokens = len(itos)
md = nn.Sequential(
    model.EncoderRNN(ntokens, args.emsize, args.nhid),
    model.LinearDecoder(args.nhid, 1)
).to(device)


print(f'Created model with {utils.count_parameters(md)} parameters:')
print(md)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(md.parameters(), lr=args.lr, betas=(0.8, 0.99))


###############################################################################
# Training code
###############################################################################

def evaluate(loader):
    """Calculates loss and prediction accuracy given torch dataloader"""
    # Turn on evaluation mode which disables dropout.
    md.eval()
    hid = md[0].init_hidden(eval_batch_size)
    avg_loss = RunningAverage()
    avg_acc = RunningAverage()

    with torch.no_grad():
        for batch in loader:
            hid = model.repackage_hidden(hid)
            # run model
            inp, target = batch
            inp, target = inp.to(device), target.to(device)
            out, hid = md(inp.t(), hid)

            # calculate loss
            loss = criterion(out.view(-1), target.float())
            avg_loss.update(loss.item())
            
            # calculate accuracy
            pred = out.view(-1) > 0.5
            correct = pred == target.byte()
            avg_acc.update(torch.sum(correct).item() / len(correct))

    return avg_loss(), avg_acc()

def train():
    # Turn on training mode which enables dropout.
    md.train()
    hid = md[0].init_hidden(args.batch_size)
    avg_loss = RunningAverage()
    avg_acc = RunningAverage()

    pbar = tqdm(train_dl, ascii=True, leave=False)
    for batch in pbar:
        hid = model.repackage_hidden(hid)
        inp, target = batch
        inp, target = inp.to(device), target.to(device)
        md.zero_grad()
        out = md((inp.t(), hid))
        loss = criterion(out.view(-1), target.float())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(md.parameters(), args.clip)
        optimizer.step()

        # upgrade stats
        avg_loss.update(loss.item())
        pred = out.view(-1) > 0.5
        correct = pred == target.byte()
        avg_acc.update(torch.sum(correct).item() / len(correct))

        pbar.set_postfix(loss=f'{avg_loss():05.3f}', acc=f'{avg_acc():05.2f}')
    
    return avg_loss(), avg_acc()


###############################################################################
# Actual training
###############################################################################

# Loop over epochs.
lr = args.lr
best_val_loss = None


for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    trn_loss, trn_acc = train()
    val_loss, val_acc = evaluate(test_dl)
    print('-' * 100)
    print(f'| end of epoch {epoch:3d} | time: {time.time()-epoch_start_time:5.2f}s '
          f'| train/valid loss {trn_loss:05.3f}/{val_loss:05.3f} | train/valid acc {trn_acc:04.2f}/{val_acc:04.2f}')
    print('-' * 100)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(md, f)
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0

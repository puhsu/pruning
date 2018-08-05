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
parser.add_argument('--load', action='store_true', help='Load dataset from disk')
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


train_ds = TextDataset(load=args.load, train=True)
test_ds = TextDataset(load=args.load, train=False)

train_dl = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=PadCollate(max_len=200))
test_dl = data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, collate_fn=PadCollate(max_len=200))

with open('data/dataset/itos.pkl', 'rb') as f:
    itos = pickle.load(f)

print('Loaded train and test data.')


###############################################################################
# Create model
###############################################################################


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
    
    def forward(self, sentences):
        embedding = self.word_embeddings(sentences)
        out, hidden = self.rnn(embedding)
        res = self.hidden2label(out[-1])
        return torch.sigmoid(res)

ntokens = len(itos)
# md = nn.Sequential(
#     model.EncoderRNN(ntokens, args.emsize, args.nhid),
#     model.LinearDecoder(args.nhid, 1)
# ).to(device)

md = LSTMClassifier(args.emsize, args.nhid, ntokens, 1).to(device)


print(f'Created model with {utils.count_parameters(md)} parameters:')
print(md)

criterion = nn.BCELoss(reduction='sum')
optimizer = torch.optim.Adam(md.parameters(), lr=args.lr, betas=(0.8, 0.99))

###############################################################################
# Training code
###############################################################################

def evaluate(loader):
    """Calculates loss and prediction accuracy given torch dataloader"""
    # Turn on evaluation mode which disables dropout.
    md.eval()
    avg_loss = RunningAverage()
    avg_acc = RunningAverage()

    with torch.no_grad():
        pbar = tqdm(loader, ascii=True, leave=False)
        for batch in pbar:
            # run model
            inp, target = batch
            inp, target = inp.to(device), target.to(device)            
            out = md(inp.t())

            # calculate loss
            loss = criterion(out.view(-1), target.float())
            avg_loss.update(loss.item())
            
            # calculate accuracy
            pred = out.view(-1) > 0.5
            correct = pred == target.byte()
            avg_acc.update(torch.sum(correct).item() / len(correct))

            pbar.set_postfix(loss=f'{avg_loss():05.3f}', acc=f'{avg_acc():05.2f}')


    return avg_loss(), avg_acc()

def train():
    # Turn on training mode which enables dropout.
    md.train()
    avg_loss = RunningAverage()
    avg_acc = RunningAverage()

    pbar = tqdm(train_dl, ascii=True, leave=False)
    for batch in pbar:
        inp, target = batch
        inp, target = inp.to(device), target.to(device)
        # run model
        md.zero_grad()
        out = md(inp.t())
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

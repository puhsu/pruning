import html
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.utils.data


def partition(a, sz):
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def partition_by_cores(a):
    return partition(a, len(a) // os.cpu_count() + 1)


###################################################################################
##  Classiffication
###################################################################################


class TextDataset(torch.utils.data.Dataset):
    # TODO transfer functions from `prepare.py` to this class
    def __init__(self, x, y, load=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TextSampler(torch.utils.data.Sampler):
    """Sampler for sequence data. Samples by sequence length, so that batches 
    have roughly same length sequences."""

    def __init__(self, data_source, key, batch_size):
        self.data_source = data_source
        self.key = key
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))
        # sort chunks by size
        chunk_size = self.batch_size * 50
        chunk_idxs = [idxs[i:i+chunk_size] for i in range(0, len(idxs), chunk_size)]
        idxs = np.concatenate([sorted(ck, key=self.key, reverse=True) for ck in chunk_idxs])
        
        # sort smaller chunks by size
        chunk_size = self.batch_size
        chunk_idxs = [idxs[i:i+chunk_size] for i in range(0, len(idxs), chunk_size)]
        # move chunk with longest key to the beggining
        max_chunk = np.argmax([self.key(ck[0]) for ck in chunk_idxs])
        chunk_idxs[0], chunk_idxs[max_chunk] = chunk_idxs[max_chunk], chunk_idxs[0]
        
        # get final order of elements
        idxs = np.concatenate(np.random.permutation(chunk_idxs[1:]))
        idxs = np.concatenate((chunk_idxs[0], idxs))
        return iter(idxs)


class PadCollate: # TODO more general implementation
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences given as numpy arrays. 

    Note:
        Should only be used with data of form [(seq, target)], 
        where seq is one dimentional array.

    Example:
        dl = DataLoader(ds, ..., collate_fn=PadCollate(pad_idx=0))
    """

    def __init__(self, pad_idx=1, max_len=None):
        self.pad_idx = pad_idx
        self.max_len = max_len

    def pad_sequence(self, seq, max_len):
        res = np.full(max_len, self.pad_idx)
        seq = seq[:max_len]
        res[-len(seq):] = seq
        return torch.tensor(res)

    def pad_collate(self, batch):
        """
        args:
            batch - list of (ndarray, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """

        # find longest sequence and pad according to it length
        max_len = self.max_len
        if max_len is None:
            max_len = max(map(lambda x: len(x[0]), batch))

        xs = torch.stack(tuple(self.pad_sequence(x, max_len) for x, y in batch))
        ys = torch.LongTensor(tuple(y for x, y in batch))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)




###################################################################################
##  Training models
###################################################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def __call__(self):
        return self.total / float(self.steps)

    def update(self, val):
        self.total += val
        self.steps += 1




###################################################################################
##  Language modeling
###################################################################################

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

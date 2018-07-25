import numpy as np
import torch

class LanguageModelLoader():
    """ 
    Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    """
    def __init__(self, nums, bs, bptt, device=torch.device('cpu')):
        self.bs, self.bptt = bs, bptt
        self.device = device
        self.data = self.batchify(nums)
        self.i = 0
        self.n = len(self.data)

    def __iter__(self):
        self.i, self.iter = 0, 0
        while self.i < self.n - 1 and self.iter < len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self):
        return self.n // self.bptt - 1

    def batchify(self, data):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs])
        data = data.reshape(self.bs, -1).T
        return data

    def get_batch(self, i, seq_len):
        seq_len = min(seq_len, len(source) - 1 - i)
        data = self.data[i:i+seq_len]
        target = self.data[i+1:i+1+seq_len].view(-1)
        return data.to(self.device), target.to(self.device)


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
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total / float(self.steps)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
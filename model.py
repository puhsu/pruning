import torch
import torch.nn as nn

###################################################################################
##  RNN models
###################################################################################
class EncoderRNN(nn.Module):
    def __init__(self, ntoken, ninp, nhid, padding_idx=1):
        super().__init__()
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=padding_idx)
        self.rnn = nn.LSTM(ninp, nhid, dropout=0.5)

        self.init_weights()
        
        self.ntoken = ntoken
        self.ninp = ninp
        self.nhid = nhid

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, inp):
        inp, hid = inp
        emb = self.encoder(inp)
        out, hid = self.rnn(emb, hid)
        return out, hid

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, self.nhid),
                weight.new_zeros(1, bsz, self.nhid))


class LinearDecoder(nn.Module):
    def __init__(self, ninp, bias=False):
        super().__init__()
        self.decoder = nn.Linear(ninp, 1, bias)

    def forward(self, inp):
        inp, hid = inp
        decoded = self.decoder(inp[-1])
        return decoded.view(-1, decoded.size(1))


###################################################################################
##  Internals
###################################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class RecurrentDropout(nn.Module):
    """Implements dropout with same mask for each time step."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, p=0.5):
        """
        Forward step. x has following dimensions: 
            (time, samples, input_dim)
        """
        if not p or not self.training:
            return x
        
        mask = torch.empty(1, x.size(1), x.size(2)).bernoulli_(1 - p) / (1 - p)
        mask = mask.expand_as(x)
        if x.is_cuda:
            mask = mask.to('cuda')
        return mask * x

class EmbeddingWithDropout(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, inp, p=0.5):
        if p and self.training:
            size = (self.weight.size(0), 1)
            mask = torch.empty(size).bernoulli_(1 - p) / (1 - p)
            mask = mask.expand_as(self.weight)
            if inp.is_cuda:
                mask = mask.to('cuda')
            dropout_weight = mask * self.weight
        else:
            dropout_weight = self.weight

        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1
        
        x = nn.functional.embedding(inp, dropout_weight, padding_idx, 
                                    self.max_norm, self.norm_type,
                                    self.scale_grad_by_freq, self.sparse)
        return x

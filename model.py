import torch.nn as nn
import torch

class RecurrentDropout(nn.Module):
    """
    Implements dropout with same mask for each time step
    """
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
            mask.to('cuda')
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
                mask.to('cuda')
            dropout_weight = mask * self.weight
        else:
            dropout_weight = self.weight

        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1
        
        x = torch.nn.functional.embedding(inp, dropout_weight, padding_idx, 
                                          self.max_norm, self.norm_type,
                                          self.scale_grad_by_freq, self.sparse)

        return x


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, dropouto=0.5, tie_weights=False):
        super(RNNModel, self).__init__()

        self.drop = RecurrentDropout()
        self.encoder = EmbeddingWithDropout(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)
        rnns = []
        for i in range(nlayers):
            if i == 0:
                rnns.append(nn.LSTM(ninp, nhid, num_layers=1))
            else:
                rnns.append(nn.LSTM(nhid, nhid, num_layers=1))
        self.rnns = nn.ModuleList(rnns)

        self.dropoute = dropoute
        self.dropouth = dropouth
        self.dropouti = dropouti
        self.dropouto = dropouto
        
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input), p=self.dropouti)
        output = emb
        outputs, new_hidden = [], []

        for l, rnn in enumerate(self.rnns):
            output, new_h = rnn(output, hidden[l])
            new_hidden.append(new_h)
            
            if l != self.nlayers - 1:
                output = self.drop(output, p=self.dropouth)
                outputs.append(output)

        output = self.drop(output, p=self.dropouto)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), new_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.nhid).zero_(),
                 weight.new(1, bsz, self.nhid).zero_())
                    for l in range(self.nlayers)]
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

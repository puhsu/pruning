class EncoderRNN(nn.Module):
    def __init__(self, ntoken, ninp, nhid, dropoute=0.1, dropouti=0.65, padding_idx=1):
        super().__init__()
        self.drop = RecurrentDropout()
        self.encoder = EmbeddingWithDropout(ntoken, ninp, padding_idx=padding_idx)
        self.rnn = nn.LSTM(ninp, nhid)

        self.init_weights()

        self.dropoute = dropoute
        self.dropouti = dropouti
        self.ntoken = ntoken
        self.ninp = ninp
        self.nhid = nhid

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp, hid):
        emb = self.encoder(inp, p=self.dropoute)
        emb = self.drop(emb, p=self.dropouti)
        out, hid = self.rnn(emb, hid)
        return out, hid

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, self.nhid),
                weight.new_zeros(1, bsz, self.nhid))
    
class ClassifierRNN(EncoderRNN):
    """Wrapper over encoder RNN. Which loops through entire sequence in chunks
    of size <= bptt
    """
    # TODO implement thing with average pooling
    def __init__(self, bptt, *args, **kwargs):
        self.bptt = bptt
        super().__init__(*args, **kwargs)

    def forward(self, inp):
        sl, bsz = inp.size()
        hid = super().init_hidden(bsz)

        for i in range(0, sl, self.bptt):
            hid = repackage_hidden(hid)
            out, hid = super().forward(inp[i:min(i+self.bptt, sl)], hid)

        return out[-1]


class LinearDecoder(nn.Module):
    """Simple Decoder.
    """
    # TODO implement thing with average pooling
    def __init__(self, ninp, nout, dropout=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(ninp, nout)
        self.dropout = dropout

    def forward(self, inp):
        inp = self.drop(inp)
        decoded = self.decoder(inp)
        return decoded.view(-1, decoded.size(1))

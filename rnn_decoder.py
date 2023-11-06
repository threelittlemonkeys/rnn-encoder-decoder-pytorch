from utils import *
from embedding import *

class rnn_decoder(nn.Module):

    def __init__(self, x_wti, y_wti):

        super().__init__()
        self.M = None # encoder hidden states
        self.H = None # decoder hidden states
        self.h = None # decoder output

        # architecture
        self.embed = embed(DEC_EMBED, None, y_wti, batch_first = True)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim + HIDDEN_SIZE * (1 + COPY),
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.attn = attn()
        if ATTN:
            self.Wc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
            self.Wo = nn.Linear(HIDDEN_SIZE, len(y_wti))
            self.softmax = nn.LogSoftmax(1)
        if COPY:
            self.Wo = nn.Linear(HIDDEN_SIZE, len(y_wti))
            self.copy = copy(x_wti, y_wti)

    def forward(self, xw, yi, mask):

        h = self.embed(None, None, yi)

        if ATTN:
            h = torch.cat([h, self.h], 2) # input feeding
            h, self.H = self.rnn(h, self.H)
            self.attn(self.M, h, mask)
            self.h = self.Wc(torch.cat([self.attn.V, h], 2)).tanh()
            h = self.Wo(self.h).squeeze(1) # [B, V]
            yo = self.softmax(h)

        if COPY:
            _M = self.M[:, :-1] # remove EOS token [B, L' = L - 1]
            self.attn(self.M, self.h, mask) # attentive read
            self.copy.attn(_M) # selective read
            h = torch.cat([h, self.attn.V, self.copy.R], 2)
            self.h, self.H = self.rnn(h, self.H)
            g = self.Wo(self.h).squeeze(1) # generation scores [B, V]
            c = self.copy.score(_M, self.h, mask) # copy scores [B, L']
            yo = self.copy.mix(xw, g, c) # [B, V']

        return yo

class attn(nn.Module): # attention mechanism (Luong et al 2015)

    def __init__(self):

        super().__init__()

        # architecture
        self.W = None # attention weights
        self.V = None # context vector

    def forward(self, hs, ht, mask):

        a = ht.bmm(hs.transpose(1, 2)) # [B, 1, H] @ [B, H, L] = [B, 1, L]
        a = a.masked_fill(mask.unsqueeze(1), -10000)
        self.W = F.softmax(a, 2) # [B, 1, L]
        self.V = self.W.bmm(hs) # [B, 1, L] @ [B, L, H] = [B, 1, H]

class copy(nn.Module): # copying mechanism (Gu et al 2016)

    def __init__(self, x_wti, y_wti):

        super().__init__()
        self.xyi = {i: y_wti[w] for w, i in x_wti.items() if w in y_wti}
        self.yxi = {i: x_wti[w] for w, i in y_wti.items() if w in x_wti}

        # architecture
        self.R = None # selective read
        self.W = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) # copy weights

    def attn(self, hs): # selective read

        self.R = self.R.unsqueeze(1).bmm(hs) # [B, 1, L'] @ [B, L', H] = [B, 1, H]

    def score(self, hs, ht, mask): # copy scores

        c = self.W(hs).tanh() # [B, L', H]
        c = ht.bmm(c.transpose(1, 2)) # [B, 1, H] @ [B, H, L'] = [B, 1, L']
        c = c.squeeze(1).masked_fill(mask[:, :-1], -10000) # [B, L']

        self.P = None # generation and copy probabilities
        self.R = F.softmax(c, 1) # selective read weights [B, L']

        return c

    def map(self, xw, vocab_size): # source to target index mapping

        idx = []
        oov = {}

        for i in xw.tolist():
            idx.append([])
            for j in i:
                if j in self.xyi:
                    j = self.xyi[j]
                else:
                    if j not in oov:
                        oov[j] = vocab_size + len(oov)
                    j = oov[j]
                idx[-1].append(j)

        idx = LongTensor(idx) # [B, L']
        oov = Tensor(len(xw), len(oov)).fill_(1e-6) # [B, OOV]
        ohv = zeros(*xw.size(), vocab_size + oov.size(1)) # [B, L', V' = V + OOV]
        ohv = ohv.scatter(2, idx.unsqueeze(2), 1) # one hot vector

        return ohv, oov

    def mix(self, xw, g, c):

        z = F.softmax(torch.cat([g, c], 1), 1) # normalization
        self.P = g, c = z.split([g.size(1), c.size(1)], 1)

        ohv, oov = self.map(xw[:, :-1], g.size(1))
        g = torch.cat([g, oov], 1) # [B, V']
        c = c.unsqueeze(1).bmm(ohv) # [B, 1, L'] @ [B, L', V'] = [B, 1, V']
        z = (g + c.squeeze(1)).log() # mixed probabilities [B, V']

        return z

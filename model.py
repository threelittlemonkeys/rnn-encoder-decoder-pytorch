import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT = "word" # unit for tokenization (char, word)
BATCH_SIZE = 64
EMBED_SIZE = 300
HIDDEN_SIZE = 1000
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
TEACHER_FORCING = 0.5
VERBOSE = False
MAX_ITER = 50 # maximum number of decoding iterations
SAVE_EVERY = 10

PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.rnn = nn.GRU( # LSTM or GRU
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = BIDIRECTIONAL
        )

        if CUDA:
            self = self.cuda()

    def init_hidden(self, rnn_type): # initialize hidden states
        h = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # hidden states
        if rnn_type == "LSTM":
            c = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # cell states
            return (h, c)
        return h

    def forward(self, x, mask):
        self.hidden = self.init_hidden("GRU") # LSTM or GRU
        x = self.embed(x)
        x = nn.utils.rnn.pack_padded_sequence(x, mask[1], batch_first = True)
        h, _ = self.rnn(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        return h

class decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.feed_input = True # input feeding

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.rnn = nn.GRU( # LSTM or GRU
            input_size = EMBED_SIZE + (HIDDEN_SIZE if self.feed_input else 0),
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = BIDIRECTIONAL
        )
        self.attn = attn()
        self.out = nn.Linear(HIDDEN_SIZE, vocab_size)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def forward(self, dec_in, enc_out, t, mask):
        x = self.embed(dec_in)
        if self.feed_input:
            x = torch.cat((x, self.attn.hidden), 2)
        h, _ = self.rnn(x, self.hidden)
        if self.attn:
            h = self.attn(h, enc_out, t, mask)
        h = self.out(h).squeeze(1)
        y = self.softmax(h)
        return y

class attn(nn.Module): # attention layer (Luong et al 2015)
    def __init__(self):
        super().__init__()
        self.type = "global" # global, local-m, local-p
        self.method = "dot" # dot, general, concat
        self.hidden = None # attentional hidden state for input feeding
        self.Va = None # attention weights

        # architecture
        if self.type[:5] == "local":
            self.window_size = 5
            if self.type[-1] == "p":
                self.Wp = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
                self.Vp = nn.Linear(HIDDEN_SIZE, 1)
        if self.method == "general":
            self.Wa = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        elif self.method  == "concat":
            pass # TODO
        self.softmax = nn.Softmax(2)
        self.Wc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)

    def window(self, ht, hs, t, mask): # for local attention
        if self.type[-1] == "m": # monotonic
            p0 = max(0, min(t - self.window_size, hs.size(1) - self.window_size))
            p1 = min(hs.size(1), t + 1 + self.window_size)
            return hs[:, p0:p1], mask[0][:, p0:p1]
        if self.type[-1] == "p": # predicative
            S = Tensor(mask[1]) # source sequence length
            pt = S * torch.sigmoid(self.Vp(torch.tanh(self.Wp(ht)))).view(-1) # aligned position
            hs_w = []
            mask_w = []
            k = [] # truncated Gaussian distribution as kernel function
            for i in range(BATCH_SIZE):
                p = int(S[i].item())
                seq_len = mask[1][i]
                min_len = mask[1][-1]
                p0 = max(0, min(p - self.window_size, seq_len - self.window_size))
                p1 = min(seq_len, p + 1 + self.window_size)
                if min_len < p1 - p0:
                    p0 = 0
                    p1 = min_len
                hs_w.append(hs[i, p0:p1])
                mask_w.append(mask[0][i, p0:p1])
                sd = (p1 - p0) / 2 # standard deviation
                v = [torch.exp(-(j - pt[i]).pow(2) / (2 * sd ** 2)) for j in range(p0, p1)]
                k.append(torch.cat(v))
            hs_w = torch.cat(hs_w).view(BATCH_SIZE, -1, HIDDEN_SIZE)
            mask_w = torch.cat(mask_w).view(BATCH_SIZE, -1)
            k = torch.cat(k).view(BATCH_SIZE, 1, -1)
            return hs_w, mask_w, pt, k

    def align(self, ht, hs, mask, k):
        if self.method == "dot":
            a = ht.bmm(hs.transpose(1, 2))
        elif self.method == "general":
            a = ht.bmm(self.Wa(hs).transpose(1, 2))
        elif self.method == "concat":
            pass # TODO
        a = a.masked_fill(mask.unsqueeze(1), -10000) # masking in log space
        a = self.softmax(a) # [B, 1, H] @ [B, H, L] = [B, 1, L]
        if self.type == "local-p":
            a = a * k
        return a # alignment weights

    def forward(self, ht, hs, t, mask):
        if self.type == "local-p":
            hs, mask, pt, k = self.window(ht, hs, t, mask)
        else:
            if self.type == "local-m":
                hs, mask = self.window(ht, hs, t, mask)
            else:
                mask = mask[0]
            k = None
        a = self.Va = self.align(ht, hs, mask, k) # alignment vector
        c = a.bmm(hs) # context vector [B, 1, H]
        h = torch.cat((c, ht), 2)
        self.hidden = torch.tanh(self.Wc(h)) # attentional vector
        return self.hidden

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def maskset(x):
    mask = x.data.eq(PAD_IDX)
    return (mask, x.size(1) - mask.sum(1)) # set of mask and lengths

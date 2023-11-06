from utils import *
from rnn_encoder import *
from rnn_decoder import *

class rnn_encoder_decoder(nn.Module):

    def __init__(self, x_cti, x_wti, y_wti):

        super().__init__()

        # architecture
        self.enc = rnn_encoder(x_cti, x_wti)
        self.dec = rnn_decoder(x_wti, y_wti)
        if CUDA: self = self.cuda()

    def init_state(self, b):

        self.dec.h = zeros(b, 1, HIDDEN_SIZE)
        self.dec.attn.W = zeros(b, 1, self.dec.M.size(1))
        self.dec.attn.V = zeros(b, 1, HIDDEN_SIZE)

        if COPY:
            self.dec.copy.R = zeros(b, self.dec.M.size(1) - 1)

    def forward(self, xc, xw, y0): # for training

        self.zero_grad()
        b = len(xw) # batch size
        mask, lens = maskset(xw)

        self.dec.M, self.dec.H = self.enc(xc, xw, lens)
        self.init_state(b)
        yi = LongTensor([SOS_IDX] * b)
        yo = []

        for t in range(y0.size(1)):
            yo.append(self.dec(xw, yi.unsqueeze(1), mask))
            yi = y0[:, t] # teacher forcing

        yo = torch.stack(yo).transpose(0, 1).flatten(0, 1)
        loss = F.nll_loss(yo, y0.view(-1), ignore_index = PAD_IDX)

        return loss

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

    def forward(self, xc, xw, y0): # for training

        self.zero_grad()
        b = len(xw) # batch size
        loss = zeros(b)
        mask, lens = maskset(xw)

        self.dec.M, self.dec.H = self.enc(xc, xw, lens)
        self.dec.h = zeros(b, 1, HIDDEN_SIZE)
        yi = LongTensor([SOS_IDX] * b)

        for t in range(y0.size(1)):
            yo = self.dec(xw, yi.unsqueeze(1), mask)
            yi = y0[:, t] # teacher forcing
            loss += F.nll_loss(yo, yi, ignore_index = PAD_IDX)

        loss /= y0.size(1) # average over timesteps

        return loss.mean(), loss

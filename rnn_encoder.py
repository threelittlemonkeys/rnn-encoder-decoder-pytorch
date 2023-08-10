from utils import *
from embedding import embed

class rnn_encoder(nn.Module):

    def __init__(self, cti, wti):

        super().__init__()

        # architecture
        self.embed = embed(ENC_EMBED, len(cti), len(wti))
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )

    def init_state(self, b): # initialize states

        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "GRU":
            return hs
        cs = zeros(n, b, h) # LSTM cell state

        return (hs, cs)

    def forward(self, xc, xw, lens):

        s = self.init_state(len(xw))
        x = self.embed(xc, xw)
        x = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first = True)
        h, s = self.rnn(x, s)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)

        return h, s

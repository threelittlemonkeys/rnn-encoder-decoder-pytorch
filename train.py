import sys
import re
import time
from model import *
from utils import *
from os.path import isfile

def load_data():
    data = []
    batch_src = []
    batch_tgt = []
    batch_len_src = 0
    batch_len_tgt = 0
    print("loading data...")
    vocab_src = load_vocab(sys.argv[2], "src")
    vocab_tgt = load_vocab(sys.argv[3], "tgt")
    fo = open(sys.argv[4], "r")
    for line in fo:
        line = line.strip()
        src, tgt = line.split("\t")
        src = [int(i) for i in src.split(" ")] + [EOS_IDX]
        tgt = [int(i) for i in tgt.split(" ")] + [EOS_IDX]
        # src.reverse() # reversing source sequence
        if len(src) > batch_len_src:
            batch_len_src = len(src)
        if len(tgt) > batch_len_tgt:
            batch_len_tgt = len(tgt)
        batch_src.append(src)
        batch_tgt.append(tgt)
        if len(batch_src) == BATCH_SIZE:
            for seq in batch_src:
                seq.extend([PAD_IDX] * (batch_len_src - len(seq)))
            for seq in batch_tgt:
                seq.extend([PAD_IDX] * (batch_len_tgt - len(seq)))
            data.append((LongTensor(batch_src), LongTensor(batch_tgt)))
            batch_src = []
            batch_tgt = []
            batch_len_src = 0
            batch_len_tgt = 0
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, vocab_src, vocab_tgt

def train():
    print("cuda: %s" % CUDA)
    num_epochs = int(sys.argv[5])
    data, vocab_src, vocab_tgt = load_data()
    if VERBOSE:
        itow_src = [w for w, _ in sorted(vocab_src.items(), key = lambda x: x[1])]
        itow_tgt = [w for w, _ in sorted(vocab_tgt.items(), key = lambda x: x[1])]
    enc = encoder(len(vocab_src))
    dec = decoder(len(vocab_tgt))
    enc_optim = torch.optim.SGD(enc.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    dec_optim = torch.optim.SGD(dec.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    epoch = load_checkpoint(sys.argv[1], enc, dec) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print(enc)
    print(dec)
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time.time()
        for x, y in data:
            loss = 0
            mask = maskset(x)
            enc.zero_grad()
            dec.zero_grad()
            if VERBOSE:
                pred = [[] for _ in range(BATCH_SIZE)]
            enc_out = enc(x, mask)
            dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
            dec.hidden = enc.hidden
            if dec.feed_input:
                dec.attn.hidden = zeros(BATCH_SIZE, 1, HIDDEN_SIZE)
            for t in range(y.size(1)):
                dec_out = dec(dec_in, enc_out, t, mask)
                loss += F.nll_loss(dec_out, y[:, t], size_average = False, ignore_index = PAD_IDX)
                dec_in = y[:, t].unsqueeze(1) # teacher forcing
                if VERBOSE:
                    for i, j in enumerate(dec_out.data.topk(1)[1]):
                        pred[i].append(scalar(j))
            loss /= y.data.gt(0).sum() # divide by the number of unpadded tokens
            loss.backward()
            enc_optim.step()
            dec_optim.step()
            loss = scalar(loss)
            loss_sum += loss
        timer = time.time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", "", "", ei, loss_sum, timer)
        else:
            if VERBOSE:
                for x, y in zip(x, pred):
                    print(" ".join([itow_src[scalar(i)] for i in x if scalar(i) != PAD_IDX]))
                    print(" ".join([itow_tgt[i] for i in y if i != PAD_IDX]))
            save_checkpoint(filename, enc, dec, ei, loss_sum, timer)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src vocab.tgt training_data num_epoch" % sys.argv[0])
    train()

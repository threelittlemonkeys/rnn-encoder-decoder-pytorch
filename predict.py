import sys
import re
from model import *
from utils import *

def load_model():
    src_vocab = load_vocab(sys.argv[2], "src")
    tgt_vocab = load_vocab(sys.argv[3], "tgt")
    tgt_vocab = [x for x, _ in sorted(tgt_vocab.items(), key = lambda x: x[1])]
    enc = encoder(len(src_vocab))
    dec = decoder(len(tgt_vocab))
    enc.eval()
    dec.eval()
    print(enc)
    print(dec)
    load_checkpoint(sys.argv[1], enc, dec)
    return enc, dec, src_vocab, tgt_vocab

def run_model(enc, dec, tgt_vocab, data):
    batch = []
    z = len(data)
    eos = [0 for _ in range(z)] # number of EOS tokens in the batch
    while len(data) < BATCH_SIZE:
        data.append(["", [EOS_IDX], []])
    data.sort(key = lambda x: len(x[1]), reverse = True)
    batch_len = len(data[0][1])
    batch = LongTensor([x[1] + [PAD_IDX] * (batch_len - len(x[1])) for x in data])
    mask = maskset(batch)
    enc_out = enc(batch, mask)
    dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
    dec.hidden = enc.hidden
    if dec.feed_input:
        dec.attn.hidden = zeros(BATCH_SIZE, 1, HIDDEN_SIZE)
    t = 0
    while sum(eos) < z:
        dec_out = dec(dec_in, enc_out, t, mask)
        dec_in = dec_out.data.topk(1)[1]
        y = dec_in.view(-1).data.tolist()
        for i in range(z):
            if eos[i]:
                continue
            if y[i] == EOS_IDX:
                eos[i] = 1
            else:
                data[i][2].append(tgt_vocab[y[i]])
        t += 1
    return [(x[0], x[2]) for x in data[:z]]

def predict():
    data = []
    enc, dec, src_vocab, tgt_vocab = load_model()
    fo = open(sys.argv[4])
    for line in fo:
        line = line.strip()
        x = tokenize(line, "word")
        x = [src_vocab[i] if i in src_vocab else UNK_IDX for i in x] + [EOS_IDX]
        data.append([line, x, []])
        if len(data) == BATCH_SIZE:
            result = run_model(enc, dec, tgt_vocab, data)
            for x in result:
                print(x)
            data = []
    fo.close()
    if len(data):
        result = run_model(enc, dec, tgt_vocab, data)
        for x in result:
            print(x)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model vocab.src vocab.tgt test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    predict()

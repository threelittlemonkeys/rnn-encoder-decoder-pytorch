import sys
from model import *
from utils import tokenize

def load_data():
    data = []
    src_vocab = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    tgt_vocab = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    fo = open(sys.argv[1])
    for line in fo:
        src, tgt = line.split("\t")
        src_tokens = tokenize(src, UNIT)
        tgt_tokens = tokenize(tgt, UNIT)
        if len(src_tokens) < MIN_LEN or len(src_tokens) > MAX_LEN:
            continue
        if len(tgt_tokens) < MIN_LEN or len(tgt_tokens) > MAX_LEN:
            continue
        src_seq = []
        tgt_seq = []
        for w in src_tokens:
            if w not in src_vocab:
                src_vocab[w] = len(src_vocab)
            src_seq.append(str(src_vocab[w]))
        for w in tgt_tokens:
            if w not in tgt_vocab:
                tgt_vocab[w] = len(tgt_vocab)
            tgt_seq.append(str(tgt_vocab[w]))
        data.append((src_seq, tgt_seq))
    fo.close()
    data.sort(key = lambda x: -len(x[0])) # sort by source sequence length
    return data, src_vocab, tgt_vocab

def save_data(data):
    fo = open(sys.argv[1] + ".csv", "w")
    for seq in data:
        fo.write(" ".join(seq[0]) + "\t" + " ".join(seq[1]) + "\n")
    fo.close()

def save_vocab(vocab, ext):
    fo = open(sys.argv[1] + ".vocab." + ext, "w")
    for w, _ in sorted(vocab.items(), key = lambda x: x[1]):
        fo.write("%s\n" % w)
    fo.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, src_vocab, tgt_vocab= load_data()
    save_data(data)
    save_vocab(src_vocab, "src")
    save_vocab(tgt_vocab, "tgt")

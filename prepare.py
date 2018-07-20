import sys
from model import SOS, EOS, PAD, SOS_IDX, EOS_IDX, PAD_IDX
from utils import tokenize

MIN_LENGTH = 3
MAX_LENGTH = 50

def load_data():
    data = []
    vocab_src = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
    vocab_tgt = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
    fo = open(sys.argv[1])
    for line in fo:
        src, tgt = line.split("\t")
        tokens_src = tokenize(src, "word")
        tokens_tgt = tokenize(tgt, "word")
        if len(tokens_src) < MIN_LENGTH or len(tokens_src) > MAX_LENGTH:
            continue
        if len(tokens_tgt) < MIN_LENGTH or len(tokens_tgt) > MAX_LENGTH:
            continue
        seq_src = []
        seq_tgt = []
        for word in tokens_src:
            if word not in vocab_src:
                vocab_src[word] = len(vocab_src)
            seq_src.append(str(vocab_src[word]))
        for word in tokens_tgt:
            if word not in vocab_tgt:
                vocab_tgt[word] = len(vocab_tgt)
            seq_tgt.append(str(vocab_tgt[word]))
        data.append((seq_src, seq_tgt))
    data.sort(key = lambda x: len(x[0]), reverse = True) # sort by source sequence length
    fo.close()
    return data, vocab_src, vocab_tgt

def save_data(data):
    fo = open(sys.argv[1] + ".csv", "w")
    for seq in data:
        fo.write(" ".join(seq[0]) + "\t" + " ".join(seq[1]) + "\n")
    fo.close()

def save_vocab(vocab, ext):
    fo = open(sys.argv[1] + ".vocab." + ext, "w")
    for word, _ in sorted(vocab.items(), key = lambda x: x[1]):
        fo.write("%s\n" % word)
    fo.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, vocab_src, vocab_tgt= load_data()
    save_data(data)
    save_vocab(vocab_src, "src")
    save_vocab(vocab_tgt, "tgt")

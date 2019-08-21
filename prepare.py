from utils import *

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, src_vocab, tgt_vocab = load_data()
    save_data(sys.argv[1] + ".csv", data)
    save_vocab(sys.argv[1] + ".vocab.src", src_vocab)
    save_vocab(sys.argv[1] + ".vocab.tgt", tgt_vocab)

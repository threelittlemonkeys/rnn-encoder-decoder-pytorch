from utils import *

def lineiter(fo):

    for line in fo:
        x, y = line.split("\t")
        x = tokenize(x, UNIT)
        y = tokenize(y, UNIT)
        if len(x) < MIN_LEN or len(x) > MAX_LEN:
            continue
        if len(y) < MIN_LEN or len(y) > MAX_LEN:
            continue
        yield x, y

def dict_to_tti(tti, vocab_size = 0):

    tokens = [PAD, SOS, EOS, UNK] # predefined tokens
    tti = sorted(tti, key = lambda x: -tti[x])
    if vocab_size:
        tti = tti[:vocab_size]
    return {w: i for i, w in enumerate(tokens + tti)}

def load_data():

    data = []
    x_cti = defaultdict(int)
    x_wti = defaultdict(int)
    y_wti = defaultdict(int)

    fo = open(sys.argv[1])
    for x, y in lineiter(fo):
        for w in x:
            for c in w:
                x_cti[c] += 1
            x_wti[w] += 1
        for w in y:
            y_wti[w] += 1

    x_cti = dict_to_tti(x_cti)
    x_wti = dict_to_tti(x_wti, SRC_VOCAB_SIZE)
    y_wti = dict_to_tti(y_wti, TGT_VOCAB_SIZE)

    fo.seek(0)
    for x, y in lineiter(fo):
        x = ["+".join(str(x_cti[c]) for c in w) + ":%d" % x_wti.get(w, UNK_IDX) for w in x]
        y = [str(y_wti.get(w, UNK_IDX)) for w in y]
        data.append((x, y))

    fo.close()
    data = sorted(data, key = lambda x: -len(x[0])) # sort by source sequence length

    return data, x_cti, x_wti, y_wti

def save_data(filename, data):

    fo = open(filename, "w")
    for seq in data:
        if not seq:
            print(file = fo)
            continue
        print(*seq[0], end = "\t", file = fo)
        print(*seq[1], file = fo)
    fo.close()

def save_tkn_to_idx(filename, tti):

    fo = open(filename, "w")
    for tkn, _ in sorted(tti.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tkn)
    fo.close()

if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])

    data, x_cti, x_wti, y_wti = load_data()
    save_data(sys.argv[1] + ".csv", data)
    save_tkn_to_idx(sys.argv[1] + ".src.char_to_idx", x_cti)
    save_tkn_to_idx(sys.argv[1] + ".src.word_to_idx", x_wti)
    save_tkn_to_idx(sys.argv[1] + ".tgt.word_to_idx", y_wti)

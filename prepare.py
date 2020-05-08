from utils import *

def load_data():
    data = []
    x_cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    x_wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    y_wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    fo = open(sys.argv[1])
    for line in fo:
        x, y = line.split("\t")
        x = tokenize(x, UNIT)
        y = tokenize(y, UNIT)
        if len(x) < MIN_LEN or len(x) > MAX_LEN:
            continue
        if len(y) < MIN_LEN or len(y) > MAX_LEN:
            continue
        src_seq = []
        tgt_seq = []
        for w in x:
            for c in w:
                if c not in x_cti:
                    x_cti[c] = len(x_cti)
            if w not in x_wti:
                x_wti[w] = len(x_wti)
        for w in y:
            if w not in y_wti:
                y_wti[w] = len(y_wti)
        x = ["+".join(str(x_cti[c]) for c in w) + ":%d" % x_wti[w] for w in x]
        y = [str(y_wti[w]) for w in y]
        data.append((x, y))
    fo.close()
    data.sort(key = lambda x: -len(x[0])) # sort by source sequence length
    return data, x_cti, x_wti, y_wti

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, x_cti, x_wti, y_wti = load_data()
    save_data(sys.argv[1] + ".csv", data)
    save_tkn_to_idx(sys.argv[1] + ".src.char_to_idx", x_cti)
    save_tkn_to_idx(sys.argv[1] + ".src.word_to_idx", x_wti)
    save_tkn_to_idx(sys.argv[1] + ".tgt.word_to_idx", y_wti)

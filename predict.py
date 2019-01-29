import sys
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

def greedy_search(dec, tgt_vocab, data, eos, dec_out, heatmap):
    p, dec_in = dec_out.topk(1)
    y = dec_in.view(-1).tolist()
    for i in range(len(eos)):
        if eos[i]:
            continue
        data[i][3].append(y[i])
        data[i][4] += p[i]
        eos[i] = y[i] == EOS_IDX
        heatmap[i].append([tgt_vocab[y[i]]] + dec.attn.Va[i][0].tolist())
    return dec_in

def beam_search(dec, tgt_vocab, data, t, eos, dec_out, heatmap):
    p, y = dec_out[:len(eos)].topk(BEAM_SIZE)
    p += Tensor([-10000 if b else a[4] for a, b in zip(data, eos)]).unsqueeze(1)
    p = p.view(len(eos) // BEAM_SIZE, -1)
    y = y.view(len(eos) // BEAM_SIZE, -1)
    if t == 0:
        p = p[:, :BEAM_SIZE]
        y = y[:, :BEAM_SIZE]
    for i, (p, y) in enumerate(zip(p, y)):
        j = i * BEAM_SIZE
        d1, m1 = [], [] # data and heatmap to be updated
        if VERBOSE >= 2:
            print("beam[%d][%d] =" % (t, i))
            for k in range(0, len(p), BEAM_SIZE):
                for a, b in zip(y[k:k + BEAM_SIZE], p[k:k + BEAM_SIZE]):
                    print(((tgt_vocab[a]), round(b.item(), 4)), end = ", ")
                print("\n")
        for p, k in zip(*p.topk(BEAM_SIZE)):
            d1.append(data[j + k // BEAM_SIZE].copy())
            d1[-1][3] = d1[-1][3] + [y[k]]
            d1[-1][4] = p
            m1.append(heatmap[j + k // BEAM_SIZE].copy())
            m1[-1].append([tgt_vocab[y[k]]] + dec.attn.Va[i][0].tolist())
        for k in filter(lambda x: eos[j + x], range(BEAM_SIZE)):
            d1.append(data[j + k])
            m1.append(heatmap[j + k])
        if VERBOSE >= 2:
            print("output[%d][%d] =" % (t, i))
        x = sorted(zip(d1, m1), key = lambda x: -x[0][4])[:BEAM_SIZE]
        for k, (a, b) in enumerate(x):
            k += j
            data[k] = a
            eos[k] = a[3][-1] == EOS_IDX
            heatmap[k] = b
            if VERBOSE >= 2:
                print([tgt_vocab[x] for x in a[3]] + [round(a[4].item(), 4)])
        if VERBOSE >= 2:
            print()
    dec_in = [x[3][-1] if len(x[3]) else SOS_IDX for x in data]
    dec_in = LongTensor(dec_in).unsqueeze(1)
    return dec_in

def run_model(enc, dec, tgt_vocab, data):
    t = 0
    eos = [False for _ in data] # number of completed sequences in the batch
    while len(data) < BATCH_SIZE:
        data.append([-1, [], [EOS_IDX], [], 0])
    data.sort(key = lambda x: -len(x[2]))
    batch_len = len(data[0][2])
    batch = LongTensor([x[2] + [PAD_IDX] * (batch_len - len(x[2])) for x in data])
    mask = maskset(batch)
    enc_out = enc(batch, mask)
    dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
    dec.hidden = enc.hidden
    if dec.feed_input:
        dec.attn.hidden = zeros(BATCH_SIZE, 1, HIDDEN_SIZE)
    heatmap = [[[""] + x[1] + [EOS]] for x in data[:len(eos)]]
    while sum(eos) < len(eos) and t < MAX_LEN:
        dec_out = dec(dec_in, enc_out, t, mask)
        if BEAM_SIZE == 1:
            dec_in = greedy_search(dec, tgt_vocab, data, eos, dec_out, heatmap)
        else:
            dec_in = beam_search(dec, tgt_vocab, data, t, eos, dec_out, heatmap)
        t += 1
    data, heatmap = zip(*sorted(zip(data, heatmap), key = lambda x: (x[0][0], -x[0][4])))
    if VERBOSE >= 1:
        for i in range(len(heatmap)):
            if VERBOSE < 2 and i % BEAM_SIZE:
                continue
            print("heatmap[%d] =" % i)
            print(mat2csv(heatmap[i], rh = True))
    data = [x for i, x in enumerate(data) if not i % BEAM_SIZE]
    return [(x[1], [tgt_vocab[x] for x in x[3][:-1]], x[4].item()) for x in data]

def predict():
    idx = 0
    data = []
    result = []
    enc, dec, src_vocab, tgt_vocab = load_model()
    fo = open(sys.argv[4])
    for line in fo:
        tkn = tokenize(line, UNIT)
        x = [src_vocab[i] if i in src_vocab else UNK_IDX for i in tkn] + [EOS_IDX]
        data.extend([[idx, tkn, x, [], 0] for _ in range(BEAM_SIZE)])
        if len(data) == BATCH_SIZE:
            result.extend(run_model(enc, dec, tgt_vocab, data))
            data = []
        idx += 1
    fo.close()
    if len(data):
        result.extend(run_model(enc, dec, tgt_vocab, data))
    for x in result:
        print(x)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model vocab.src vocab.tgt test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    print("batch size: %d" % BATCH_SIZE)
    with torch.no_grad():
        predict()

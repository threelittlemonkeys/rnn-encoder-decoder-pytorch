from model import *
from utils import *

def load_model():
    src_vocab = load_vocab(sys.argv[2])
    tgt_vocab = load_vocab(sys.argv[3])
    tgt_vocab = [x for x, _ in sorted(tgt_vocab.items(), key = lambda x: x[1])]
    model = rnn_enc_dec(len(src_vocab), len(tgt_vocab))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, src_vocab, tgt_vocab

def greedy_search(dec, itw, batch, eos, dec_out, heatmap):
    p, dec_in = dec_out.topk(1)
    y = dec_in.view(-1).tolist()
    for i in range(len(eos)):
        if eos[i]:
            continue
        batch[i][3].append(y[i])
        batch[i][4] += p[i]
        eos[i] = (y[i] == EOS_IDX)
        heatmap[i].append([itw[y[i]]] + dec.attn.a[i][0].tolist())
    return dec_in

def beam_search(dec, itw, batch, t, eos, dec_out, heatmap):
    bp, by = dec_out[:len(eos)].topk(BEAM_SIZE) # [B * BEAM_SIZE, BEAM_SIZE]
    bp += Tensor([-10000 if b else a[4] for a, b in zip(batch, eos)]).unsqueeze(1) # update
    bp = bp.view(-1, BEAM_SIZE ** 2) # [B, BEAM_SIZE * BEAM_SIZE]
    by = by.view(-1, BEAM_SIZE ** 2)
    if t == 0: # remove non-first duplicate beams
        bp = bp[:, :BEAM_SIZE]
        by = by[:, :BEAM_SIZE]
    for i, (p, y) in enumerate(zip(bp, by)): # for each sequence
        j = i * BEAM_SIZE
        b1, m1 = [], [] # batch and heatmap to be updated
        if VERBOSE >= 2:
            for k in range(0, len(p), BEAM_SIZE): # for each beam
                q = j + k // BEAM_SIZE
                w = [(next(reversed(batch[q][3]), SOS_IDX), batch[q][4])] # previous token
                w += list(zip(y, p))[k:k + BEAM_SIZE] # current candidates
                w = [(itw[a], round(b.item(), 4)) for a, b in w]
                print("beam[%d][%d][%d] =" % (i, t, q), w[0], "->", w[1:])
        for p, k in zip(*p.topk(BEAM_SIZE)): # for each n-best candidate
            q = j + k // BEAM_SIZE
            b1.append(batch[q].copy())
            b1[-1][3] = b1[-1][3] + [y[k]] # word
            b1[-1][4] = p # probability
            m1.append(heatmap[q].copy())
            m1[-1].append([itw[y[k]]] + dec.attn.a[q][0][:len(batch[j][1]) + 1].tolist())
        for k in filter(lambda x: eos[j + x], range(BEAM_SIZE)): # append completed sequences
            b1.append(batch[j + k])
            m1.append(heatmap[j + k])
        topk = sorted(zip(b1, m1), key = lambda x: -x[0][4])[:BEAM_SIZE]
        for k, (b1, m1) in enumerate(topk, j):
            batch[k] = b1
            eos[k] = (b1[3][-1] == EOS_IDX)
            heatmap[k] = m1
            if VERBOSE >= 2:
                print("output[%d][%d][%d] = " % (i, t, k), end = "")
                print(([itw[x] for x in b1[3]], round(b1[4].item(), 4)))
    if VERBOSE >= 2:
        print()
    return LongTensor([next(reversed(x[3]), SOS_IDX) for x in batch]).unsqueeze(1)

def run_model(model, tgt_vocab, batch):
    t = 0
    eos = [False for _ in batch] # number of completed sequences in the batch
    while len(batch) < BATCH_SIZE:
        batch.append([-1, [], [EOS_IDX], [], 0])
    batch.sort(key = lambda x: -len(x[2]))
    _, bx = batchify(None, [x[2] for x in batch], eos = True)
    mask = maskset(bx)
    enc_out = model.enc(bx, mask)
    dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
    model.dec.hidden = model.enc.hidden
    if model.dec.feed_input:
        model.dec.attn.h = zeros(BATCH_SIZE, 1, HIDDEN_SIZE)
    heatmap = [[[""] + x[1] + [EOS]] for x in batch[:len(eos)]]
    while sum(eos) < len(eos) and t < MAX_LEN:
        dec_out = model.dec(dec_in, enc_out, t, mask)
        if BEAM_SIZE == 1:
            dec_in = greedy_search(model.dec, tgt_vocab, batch, eos, dec_out, heatmap)
        else:
            dec_in = beam_search(model.dec, tgt_vocab, batch, t, eos, dec_out, heatmap)
        t += 1
    batch, heatmap = zip(*sorted(zip(batch, heatmap), key = lambda x: (x[0][0], -x[0][4])))
    if VERBOSE >= 1:
        for i in range(len(heatmap)):
            if VERBOSE < 2 and i % BEAM_SIZE:
                continue
            print("heatmap[%d] =" % i)
            print(mat2csv(heatmap[i], rh = True))
    batch = [x for i, x in enumerate(batch) if not i % BEAM_SIZE]
    return [(x[1], [tgt_vocab[x] for x in x[3][:-1]], x[4].item()) for x in batch]

def predict(filename, model, src_vocab, tgt_vocab):
    data = []
    result = []
    fo = open(filename)
    for idx, line in enumerate(fo):
        tkn = tokenize(line, UNIT)
        x = [src_vocab[i] if i in src_vocab else UNK_IDX for i in tkn]
        data.extend([[idx, tkn, x, [], Tensor([0])] for _ in range(BEAM_SIZE)])
    fo.close()
    with torch.no_grad():
        model.eval()
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            for y in run_model(model, tgt_vocab, batch):
                yield y

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model vocab.src vocab.tgt test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    result = predict(sys.argv[4], *load_model())
    for x, y, p in result:
        print((x, y))

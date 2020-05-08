from parameters import *

def greedy_search(dec, batch, itw, eos, lens, yo):
    p, yi = yo.topk(1)
    y = yi.view(-1).tolist()
    for i, _ in filter(lambda x: not x[1], enumerate(eos)):
        eos[i] = (y[i] == EOS_IDX)
        batch.y1[i].append(y[i])
        batch.prob[i] += p[i]
        batch.attn[i].append([itw[y[i]], *dec.attn.w[i][0][:lens[i]]])
    return yi

def beam_search(dec, batch, itw, eos, lens, yo, t):
    bp, by = yo[::1 if t else BEAM_SIZE].topk(BEAM_SIZE) # [B * BEAM_SIZE, BEAM_SIZE]
    if t: # update probabilities and reshape into [B, BEAM_SIZE * BEAM_SIZE]
        bp += Tensor([-10000 if e else p for p, e in zip(batch.prob, eos)]).unsqueeze(1)
        bp, by = bp.view(-1, BEAM_SIZE ** 2), by.view(-1, BEAM_SIZE ** 2)
    for i, (bp, by) in enumerate(zip(bp, by)): # for each sequence
        j, _y1, _prob, _attn = i * BEAM_SIZE, [], [], []
        if VERBOSE >= 2:
            for k in range(0, len(bp), BEAM_SIZE): # for each previous beam
                q = j + k // BEAM_SIZE
                w = [(batch.prob[q], *(batch.y1[q][-1:] or [SOS_IDX]))] # previous token
                w += list(zip(bp, by))[k:k + BEAM_SIZE] # current candidates
                w = [(round(p.item(), NUM_DIGITS), itw[y]) for p, y in w]
                print("beam[%d][%d][%d] = %s ->" % (t, i, k // BEAM_SIZE, w[0]), *w[1:])
        for p, k in zip(*bp.topk(BEAM_SIZE)): # append n-best candidates
            q = j + k // BEAM_SIZE
            _y1.append(batch.y1[q] + [by[k]])
            _prob.append(p)
            _attn.append(batch.attn[q] + [[itw[by[k]], *dec.attn.w[q][0][:lens[j]]]])
        for k in filter(lambda x: eos[x], range(j, j + BEAM_SIZE)): # append completed sequences
            _y1.append(batch.y1[k][:])
            _prob.append(batch.prob[k])
            _attn.append(batch.attn[k][:])
        topk = sorted(zip(_y1, _prob, _attn), key = lambda x: -x[1])[:BEAM_SIZE]
        for k, (_y1, _prob, _attn) in enumerate(topk, j):
            batch.y1[k] = _y1
            batch.prob[k] = _prob
            batch.attn[k] = _attn
            eos[k] = (_y1[-1] == EOS_IDX)
            if VERBOSE >= 2:
                print("output[%d][%d][%d] = " % (t, i, k - j), end = "")
                print(([itw[y] for y in _y1], round(_prob.item(), 4)))
        if VERBOSE >= 2:
            print()
    return LongTensor([y[-1] for y in batch.y1]).unsqueeze(1)

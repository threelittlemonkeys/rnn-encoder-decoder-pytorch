from utils import *
from dataloader import *
from rnn_encoder_decoder import *
from beamsearch import *

def load_model():

    x_cti = load_tkn_to_idx(sys.argv[2])
    x_wti = load_tkn_to_idx(sys.argv[3])
    y_wti = load_tkn_to_idx(sys.argv[4])
    y_itw = load_idx_to_tkn(sys.argv[4])

    model = rnn_encoder_decoder(x_cti, x_wti, y_wti)
    print(model)

    load_checkpoint(sys.argv[1], model)

    return model, x_cti, x_wti, y_itw

def run_model(model, data, itw):

    with torch.no_grad():
        model.eval()

        for batch in data.split(BATCH_SIZE):

            xc, xw, _, lens = batch.sort()
            xc, xw = data.tensor(xc, xw, lens, eos = True)
            eos = [False for _ in xw] # EOS states
            mask, lens = maskset(xw)

            model.dec.M, model.dec.H = model.enc(xc, xw, lens)
            model.dec.h = zeros(len(xw), 1, HIDDEN_SIZE)
            yi = LongTensor([[SOS_IDX]] * len(xw))

            batch.y1 = [[] for _ in xw]
            batch.prob = [zeros(1) for _ in xw]
            batch.attn = [[["", *batch.x1[i], EOS]] for i in batch.idx]
            batch.copy = [[["", *batch.x1[i]]] for i in batch.idx]

            t = 0
            while t < MAX_LEN and sum(eos) < len(eos):
                yo = model.dec(xw, yi, mask)
                args = (model.dec, batch, itw, eos, lens, yo)
                yi = beam_search(*args, t) if BEAM_SIZE > 1 else greedy_search(*args)
                t += 1
            batch.unsort()

            if VERBOSE:
                for i in range(0, len(batch.y1), BEAM_SIZE):
                    i //= BEAM_SIZE
                    print("attn[%d] =" % i)
                    print(mat2csv(batch.attn[i]), end = "\n\n")
                    if COPY:
                        print("copy[%d] =" % i)
                        print(mat2csv(batch.copy[i][:-1]), end = "\n\n")

            for i, (x0, y0, y1) in enumerate(zip(batch.x0, batch.y0, batch.y1)):
                if not i % BEAM_SIZE: # use the best candidate from each beam
                    y1 = [itw[y] for y in y1[:-1]]
                    yield x0, y0, y1

def predict(filename, model, x_cti, x_wti, y_itw):

    data = dataloader(batch_first = True)
    fo = open(filename)

    for x0 in fo:

        x0, y0 = x0.strip(), []
        if x0.count("\t") == 1:
            x0, y0 = x0.split("\t")
        x1 = tokenize(x0, UNIT)
        xc = [[x_cti.get(c, UNK_IDX) for c in w] for w in x1]
        xw = [x_wti.get(w, UNK_IDX) for w in x1]

        for _ in range(BEAM_SIZE):
            data.append_row()
            data.append_item(x0 = x0, x1 = x1, xc = xc, xw = xw, y0 = y0)

    fo.close()

    return run_model(model, data, y_itw)

if __name__ == "__main__":

    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src.char_to_idx vocab.src.word_to_idx vocab.tgt.word_to_idx test_data" % sys.argv[0])

    for x, y0, y1 in predict(sys.argv[5], *load_model()):
        print((x, y0, y1) if y0 else (x, y1))

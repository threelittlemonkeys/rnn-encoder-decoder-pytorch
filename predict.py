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

def run_model(model, data, y_itw):

    with torch.no_grad():
        model.eval()

        for batch in data.split(BATCH_SIZE):

            xc, xw, _, lens = batch.sort()
            xc, xw = data.tensor(xc, xw, lens, eos = True)
            eos = [False for _ in xw] # EOS states
            b, t = len(xw), 0
            mask, lens = maskset(xw)

            model.dec.M, model.dec.H = model.enc(xc, xw, lens)
            model.init_state(b)
            yw = LongTensor([[SOS_IDX]] * b)

            batch.y1 = [[] for _ in xw]
            batch.prob = [0 for _ in xw]
            batch.attn = [[["", *batch.x1[i], EOS]] for i in batch.idx]
            batch.copy = [[["", *batch.x1[i]]] for i in batch.idx]

            while t < MAX_LEN and sum(eos) < len(eos):
                yo = model.dec(xw, yw, mask)
                args = (model.dec, batch, y_itw, eos, lens, yo)
                yw = beam_search(*args, t) if BEAM_SIZE > 1 else greedy_search(*args)
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
                    y1 = [y_itw[y] for y in y1[:-1]]
                    yield x0, y0, y1

def predict(model, x_cti, x_wti, y_itw, filename):

    data = dataloader(batch_first = True)
    fo = open(filename)

    for line in fo:
        data.append_row()

        x0, y0 = line.strip(), []
        if x0.count("\t") == 1:
            x0, y0 = x0.split("\t")
        x1 = tokenize(x0, UNIT)
        xc = [[x_cti.get(c, UNK_IDX) for c in w] for w in x1]
        xw = [x_wti.get(w, UNK_IDX) for w in x1]
        data.append_item(x0 = x0, x1 = x1, xc = xc, xw = xw, y0 = y0)

        for _ in range(BEAM_SIZE - 1):
            data.clone_row()

    fo.close()

    return run_model(model, data, y_itw)

if __name__ == "__main__":

    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src.char_to_idx vocab.src.word_to_idx vocab.tgt.word_to_idx test_data" % sys.argv[0])

    for x, y0, y1 in predict(*load_model(), sys.argv[5]):
        if y0:
            print((x, y0))
        print((x, y1))

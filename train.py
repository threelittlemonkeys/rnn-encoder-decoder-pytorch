from model import *
from utils import *
# from evaluate import *

def load_data():
    data = []
    bx = [] # source sequence batch
    by = [] # target sequence batch
    src_vocab = load_vocab(sys.argv[2])
    tgt_vocab = load_vocab(sys.argv[3])
    print("loading %s" % sys.argv[4])
    fo = open(sys.argv[4], "r")
    for line in fo:
        x, y = line.strip().split("\t")
        x = [int(i) for i in x.split(" ")]
        y = [int(i) for i in y.split(" ")]
        # x.reverse() # reversing source sequence
        bx.append(x)
        by.append(y)
        if len(by) == BATCH_SIZE:
            _, bx = batchify(None, bx, eos = True)
            _, by = batchify(None, by, eos = True)
            data.append((bx, by))
            bx = []
            by = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, src_vocab, tgt_vocab

def train():
    print("cuda: %s" % CUDA)
    num_epochs = int(sys.argv[5])
    data, src_vocab, tgt_vocab = load_data()
    enc = encoder(len(src_vocab))
    dec = decoder(len(tgt_vocab))
    enc_optim = torch.optim.Adam(enc.parameters(), lr = LEARNING_RATE)
    dec_optim = torch.optim.Adam(dec.parameters(), lr = LEARNING_RATE)
    epoch = load_checkpoint(sys.argv[1], enc, dec) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print(enc)
    print(dec)
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        ii = 0
        loss_sum = 0
        timer = time()
        for x, y in data:
            ii += 1
            loss = 0
            enc.zero_grad()
            dec.zero_grad()
            mask = maskset(x)
            enc_out = enc(x, mask)
            dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
            dec.hidden = enc.hidden
            if dec.feed_input:
                dec.attn.h = zeros(BATCH_SIZE, 1, HIDDEN_SIZE)
            for t in range(y.size(1)):
                dec_out = dec(dec_in, enc_out, t, mask)
                loss += F.nll_loss(dec_out, y[:, t], ignore_index = PAD_IDX, reduction = "sum")
                dec_in = y[:, t].unsqueeze(1) # teacher forcing
            loss /= y.data.gt(0).sum().float() # divide by the number of unpadded tokens
            loss.backward()
            enc_optim.step()
            dec_optim.step()
            loss = loss.item()
            loss_sum += loss
            # print("epoch = %d, iteration = %d, loss = %f" % (ei, ii, loss))
        timer = time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, enc, dec, ei, loss_sum, timer)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src vocab.tgt training_data num_epoch" % sys.argv[0])
    train()

from model import *
from utils import *
# from evaluate import *

def load_data():
    data = []
    bx = [] # source sequence batch
    by = [] # target sequence batch
    src_vocab = load_vocab(sys.argv[2])
    tgt_vocab = load_vocab(sys.argv[3])
    print("loading %s..." % sys.argv[4])
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
    model = rnn_enc_dec(len(src_vocab), len(tgt_vocab))
    enc_optim = torch.optim.Adam(model.enc.parameters(), lr = LEARNING_RATE)
    dec_optim = torch.optim.Adam(model.dec.parameters(), lr = LEARNING_RATE)
    print(model)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time()
        for x, y in data:
            loss = model(x, y) # forward pass and compute loss
            loss.backward() # compute gradients
            enc_optim.step() # update encoder parameters
            dec_optim.step() # update decoder parameters
            loss_sum += loss.item()
        timer = time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src vocab.tgt training_data num_epoch" % sys.argv[0])
    train()

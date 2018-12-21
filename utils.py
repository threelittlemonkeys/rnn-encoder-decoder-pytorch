import re
from model import *

def normalize(x):
    # x = re.sub("[^ ,.?!a-zA-Z0-9\u3131-\u318E\uAC00-\uD7A3]+", " ", x)
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x, unit):
    x = normalize(x)
    if unit == "char":
        # x = re.sub(" ", "", x)
        return list(x)
    if unit == "word":
        return x.split(" ")

def load_vocab(filename, ext):
    print("loading vocab.%s..." % ext)
    vocab = {}
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        vocab[line] = len(vocab)
    fo.close()
    return vocab

def load_checkpoint(filename, enc = None, dec = None):
    print("loading model...")
    checkpoint = torch.load(filename)
    if enc:
        enc.load_state_dict(checkpoint["encoder_state_dict"])
    if dec:
        dec.load_state_dict(checkpoint["decoder_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, enc, dec, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and enc and dec:
        print("saving model...")
        checkpoint = {}
        checkpoint["encoder_state_dict"] = enc.state_dict()
        checkpoint["decoder_state_dict"] = dec.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)

def mat2csv(m, delim ="\t", n = 6):
    k = 10 ** -n
    csv = delim.join([x for x in m[0]]) + "\n"
    for v in m[1:]:
        csv += v[0] + delim
        if n:
            csv += delim.join([str(round(x, n)) if x > k else "" for x in v[1:]]) + "\n"
        else:
            csv += delim.join([str(x) for x in v[1:]]) + "\n"
    return csv

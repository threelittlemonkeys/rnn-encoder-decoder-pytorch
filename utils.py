import sys
import re
from time import time
from os.path import isfile
from parameters import *
from collections import defaultdict

def normalize(x):
    # x = re.sub("[^ ,.?!a-zA-Z0-9\u3131-\u318E\uAC00-\uD7A3]+", " ", x)
    x = re.sub("(?=[,.?!])", " ", x)
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x, unit):
    x = normalize(x)
    if unit == "char":
        return list(x)
    if unit == "word":
        return x.split(" ")

def save_data(filename, data):
    fo = open(filename, "w")
    for seq in data:
        fo.write(" ".join(seq[0]) + "\t" + " ".join(seq[1]) + "\n")
    fo.close()

def load_idx_to_tkn(filename):
    print("loading %s" % filename)
    idx_to_tkn = []
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        idx_to_tkn.append(line)
    fo.close()
    return idx_to_tkn

def load_tkn_to_idx(filename):
    print("loading %s" % filename)
    tkn_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        tkn_to_idx[line] = len(tkn_to_idx)
    fo.close()
    return tkn_to_idx

def save_tkn_to_idx(filename, tkn_to_idx):
    fo = open(filename, "w")
    for tkn, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tkn)
    fo.close()

def load_vocab(filename):
    print("loading %s..." % filename)
    vocab = {}
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        vocab[line] = len(vocab)
    fo.close()
    return vocab

def save_vocab(filename, vocab):
    fo = open(filename, "w")
    for w, _ in sorted(vocab.items(), key = lambda x: x[1]):
        fo.write("%s\n" % w)
    fo.close()

def load_checkpoint(filename, model = None):
    print("loading model...")
    checkpoint = torch.load(filename)
    if model:
        model.enc.load_state_dict(checkpoint["encoder_state_dict"])
        model.dec.load_state_dict(checkpoint["decoder_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f\n" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        print("saving model...")
        checkpoint = {}
        checkpoint["encoder_state_dict"] = model.enc.state_dict()
        checkpoint["decoder_state_dict"] = model.dec.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)

def cudify(f):
    return lambda *x: f(*x).cuda() if CUDA else f(*x)

Tensor = cudify(torch.Tensor)
LongTensor = cudify(torch.LongTensor)
zeros = cudify(torch.zeros)

def batchify(bxc, bxw, sos = False, eos = False, minlen = 0):
    bxw_len = max(minlen, max(len(x) for x in bxw))
    if bxc:
        bxc_len = max(minlen, max(len(w) for x in bxc for w in x))
        pad = [[PAD_IDX] * (bxc_len + 2)]
        bxc = [[[SOS_IDX, *w, EOS_IDX, *[PAD_IDX] * (bxc_len - len(w))] for w in x] for x in bxc]
        bxc = [(pad if sos else []) + x + (pad * (bxw_len - len(x) + eos)) for x in bxc]
        bxc = LongTensor(bxc)
    sos = [SOS_IDX] if sos else []
    eos = [EOS_IDX] if eos else []
    bxw = [sos + list(x) + eos + [PAD_IDX] * (bxw_len - len(x)) for x in bxw]
    return bxc, LongTensor(bxw)

def maskset(x):
    mask = x.eq(PAD_IDX)
    return (mask, x.size(1) - mask.sum(1)) # set of mask and lengths

def mat2csv(m, ch = True, rh = False, nd = 4, delim ="\t"):
    f = "%%.%df" % nd
    if ch: # column header
        csv = delim.join([x for x in m[0]]) + "\n" # source sequence
    for row in m[ch:]:
        if rh: # row header
            csv += str(row[0]) + delim # target sequence
        csv += delim.join([f % x for x in row[rh:]]) + "\n"
    return csv

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0

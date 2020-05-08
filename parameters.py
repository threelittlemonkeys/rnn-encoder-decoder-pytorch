import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT = "word" # unit of tokenization (char, word)
MIN_LEN = 1 # minimum sequence length for training
MAX_LEN = 50 # maximum sequence length for training and decoding
RNN_TYPE = "LSTM" # LSTM or GRU
NUM_DIRS = 2 # unidirectional: 1, bidirectional: 2
NUM_LAYERS = 2
BATCH_SIZE = 64 * 3 # BATCH_SIZE * BEAM_SIZE
HRE = False # (UNIT == "sent") # hierarchical recurrent encoding
ENC_EMBED = {"lookup": 300} # encoder embedding (char-cnn, char-rnn, lookup, sae)
DEC_EMBED = {"lookup": 300} # decoder embedding (lookup only)
HIDDEN_SIZE = 1000
DROPOUT = 0.5
LEARNING_RATE = 2e-4
BEAM_SIZE = 1
VERBOSE = 0 # 0: None, 1: attention heatmap, 2: beam search
EVAL_EVERY = 10
SAVE_EVERY = 10

PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

CUDA = torch.cuda.is_available()
torch.manual_seed(0) # for reproducibility
# torch.cuda.set_device(0)

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
zeros = lambda *x: torch.zeros(*x).cuda() if CUDA else torch.zeros

NUM_DIGITS = 4 # number of decimal places to print
assert BATCH_SIZE % BEAM_SIZE == 0

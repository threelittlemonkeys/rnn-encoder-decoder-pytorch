import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT = "word" # unit of tokenization (char, word)
MIN_LEN = 1 # minimum sequence length for training
MAX_LEN = 50 # maximum sequence length for training and inference
SRC_VOCAB_SIZE = 50000 # source vocabulary size (0: limitless)
TGT_VOCAB_SIZE = 50000 # target vocabulary size (0: limitless)

RNN_TYPE = "GRU" # GRU, LSTM
NUM_DIRS = 2 # number of directions (1: unidirectional, 2: bidirectional)
NUM_LAYERS = 2
HRE = False # (UNIT == "sent") # hierarchical recurrent encoding
ENC_EMBED = {"lookup": 50} # encoder embedding (char-cnn, char-rnn, lookup, sae)
DEC_EMBED = {"lookup": 50} # decoder embedding (lookup only)
HIDDEN_SIZE = 300
DROPOUT = 0.5
LEARNING_RATE = 2e-4

ATTN = True # attention mechanism
COPY = False # copying mechanism

BEAM_SIZE = 1
BATCH_SIZE = 64 * BEAM_SIZE

VERBOSE = 0 # 0: None, 1: attention heatmap, 2: beam search
EVAL_EVERY = 10
SAVE_EVERY = 10
SAVE_LOSS = False
NUM_DIGITS = 4 # number of decimal places to print

PAD, PAD_IDX = "<PAD>", 0 # padding
SOS, SOS_IDX = "<SOS>", 1 # start of sequence
EOS, EOS_IDX = "<EOS>", 2 # end of sequence
UNK, UNK_IDX = "<UNK>", 3 # unknown token

CUDA = torch.cuda.is_available()
torch.manual_seed(0) # for reproducibility
# torch.cuda.set_device(0)

assert ATTN != COPY
assert BATCH_SIZE % BEAM_SIZE == 0

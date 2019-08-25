import math
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
EMBED = {"lookup": 300} # embeddings (char-cnn, char-rnn, lookup, sae)
HIDDEN_SIZE = 1000
DROPOUT = 0.5
LEARNING_RATE = 1e-4
BEAM_SIZE = 3
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

assert BATCH_SIZE % BEAM_SIZE == 0

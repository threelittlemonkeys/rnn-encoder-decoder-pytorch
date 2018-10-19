# The RNN Encoder-Decoder in PyTorch

A PyTorch implementation of the RNN Encoder-Decoder for sequence to sequence learning, adapted from [the PyTorch tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

Supported features:
- Mini-batch training with CUDA
- Global and local attention (Luong et al 2015)
- Input feeding (Luong et al 2015)
- Vectorized computation of alignment scores in the attention layer

## Usage

Training data should be formatted as below:
```
source_sequence \t target_sequence
source_sequence \t target_sequence
...
```

To prepare data:
```
python prepare.py training_data
```

To train:
```
python train.py model vocab.src vocab.tgt training_data.csv num_epoch
```

To predict:
```
python predict.py model.epochN vocab.src vocab.tgt test_data
```

## References

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. 2015. [Neural Machine Translation by Jointly Learning to Align and Translate.](https://arxiv.org/abs/1409.0473) arXiv:1409.0473.

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio. 2014. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.](https://arxiv.org/abs/1406.1078) arXiv:1406.1078.

Jiatao Gu, Zhengdong Lu, Hang Li, Victor O.K. Li. 2016. [Incorporating Copying Mechanism in Sequence-to-Sequence Learning.](https://arxiv.org/abs/1603.06393) arXiv:1603.06393.

Minh-Thang Luong, Hieu Pham, Christopher D. Manning. 2015. [Effective Approaches to Attention-based Neural Machine Translation.](https://arxiv.org/abs/1508.04025) arXiv:1507.04025.

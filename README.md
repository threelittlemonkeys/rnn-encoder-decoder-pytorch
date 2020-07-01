# RNN Encoder-Decoder in PyTorch

A minimal PyTorch implementation of RNN Encoder-Decoder for sequence to sequence learning.

Supported features:
- Mini-batch training with CUDA
- Lookup, CNNs, RNNs and/or self-attentive encoding in the embedding layer
- Attention mechanism (Luong et al 2015)
- Input feeding (Luong et al 2015)
- CopyNet, copying mechanism (Gu et al 2016)
- Beam search decoding
- Attention visualization

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

Denny Britz, Anna Goldie, Minh-Thang Luong, Quoc Le. 2017. [Massive Exploration of Neural Machine Translation Architectures.](https://arxiv.org/abs/1703.03906) arXiv:1703.03906.

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio. 2014. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.](https://arxiv.org/abs/1406.1078) arXiv:1406.1078.

Jiatao Gu, Zhengdong Lu, Hang Li, Victor O.K. Li. 2016. [Incorporating Copying Mechanism in Sequence-to-Sequence Learning.](https://arxiv.org/abs/1603.06393) arXiv:1603.06393.

Jiwei Li. 2017. [Teaching Machines to Converse.](https://github.com/jiweil/Jiwei-Thesis/blob/master/thesis.pdf) Doctoral dissertation. Stanford University.

Junyang Lin, Xu Sun, Xuancheng Ren, Muyu Li, Qi Su. 2018. [Learning When to Concentrate or Divert Attention: Self-Adaptive Attention Temperature for Neural Machine Translation.](https://arxiv.org/abs/1808.07374) arXiv:1808.07374.

Minh-Thang Luong, Hieu Pham, Christopher D. Manning. 2015. [Effective Approaches to Attention-based Neural Machine Translation.](https://arxiv.org/abs/1508.04025) arXiv:1507.04025.

Chan Young Park, Yulia Tsvetkov. [Learning to Generate Word- and Phrase-Embeddings for Efficient Phrase-Based Neural Machine Translation.](https://www.aclweb.org/anthology/D19-5626.pdf) In Proceedings of the 3rd Workshop on Neural Generation and Translation.

Sam Wiseman, Alexander M. Rush. [Sequence-to-Sequence Learning as Beam-Search Optimization.](https://arxiv.org/abs/1606.02960) arXiv:1606.02960.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, Jeffrey Dean. 2016. [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation.](https://arxiv.org/abs/1609.08144) arXiv:1609.08144.

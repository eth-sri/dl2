## Training with DL2

This directory constrains different examples of training with DL2.
The dl2 library allows to express logical constraints as a loss over numerical terms.
This loss can be used directly in training, as we do here in the semi-supervised and unsupervised experiments.
When convex domains for projection are identified, the PGD-style training algorithm discussed in the paper can be used. We apply this in the supervised experiments.

`setup.sh` downloads additional requirements  and should be run before running any training.

### Supervised Learning
The examples from the paper can be run via the command line switches on `cegis_train_constraint.py`.
`run.sh` shows example invocations.

### Semi-supervised Learning
The code is based on [wide-resnet](https://github.com/meliketoy/wide-resnet.pytorch) and [PyTorch-Cifar](https://github.com/kuangliu/pytorch-cifar) slightly modified.
The examples from the paper can be run via the command line switches on `main.py`.

### Unsupervised Learning
The code is based on [pygcn](https://github.com/tkipf/pygcn).
The examples from the paper can be run via the command line switches on `train.py`.

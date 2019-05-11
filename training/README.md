## Training with DL2

This directory constrains different examples of training neural networks with DL2 constraints.
The dl2 library allows to express logical constraints as a loss over numerical terms.

This loss can be used directly in training, as we do here in the semi-supervised and unsupervised experiments.
For supervised training we use DL2 to specify constraints on the network involving the dataset. This allows us to find specific training examples inside and outside the training data to optimize the network for the constraint. Please see the paper for details.

### Supervised Learning
The experiments from the paper can be run via the command line switches on `main.py`.
`run.sh` contains the exact invocations.

### Semi-supervised Learning
The code is based on [wide-resnet](https://github.com/meliketoy/wide-resnet.pytorch) and [PyTorch-Cifar](https://github.com/kuangliu/pytorch-cifar) slightly modified.
The experiments from the paper can be run via the command line switches on `main.py`.
`run.sh` contains the exact invocations.

### Unsupervised Learning
The code is based on [pygcn](https://github.com/tkipf/pygcn).
`setup.py` needs to be run before first usage.
The experiments from the paper can be run via the command line switches on `main.py`.
`run.sh` contains the exact invocations.

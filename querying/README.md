## DL2 querying

DL2 allows to previously query trained neural networks for their inputs inputs.

`train_models.sh` downloads and trains several neural networks on common classification tasks.
Many of the models use existing implementations and datasets. See section "Used Models" below for details.

Querying can be done with two different APIs:
- the Querying DSL, introduced in the paper, and
- the Querying API.

Both expose the same features and the DSL is implemented in terms of the API.
Depending on the task one or the other can be easier to use.

### Querying DSL
This uses the same syntax as in the paper to query networks.
`run.py` implements the evaluation section of the paper.

#### Example usage
``` python
import dl2lib as dl2
import dl2lib.query as q
qtext = """FIND i[10]
S.T. i[0] + i[3] < NN(i[4])""" # actualy query
context = dict() # a python dict that defines models and named constants
context['NN'] = myPytrochModel
parser = ArgumentParser(description='DL2 Querying')
parser = dl2.add_default_parser_args(parser, query=True)
args = parser.parse_args() # additionaly arguments such as timeout etc.
success, result, time = q.Query(qtext, context=context, args=args).run()
```

### Querying API
`run_additional.py` performs the experiments found in the appendix.
It uses the querying API for q1 and q2.

#### Example usage
``` python
import dl2lib as dl2
import dl2lib.query as q
parser = ArgumentParser(description='DL2 Querying')
parser = dl2.add_default_parser_args(parser, query=True)
args = parser.parse_args() # additionaly arguments such as timeout etc.
NN = q.Model(myPytrochModel)
i = q.Variable('i', (10,))
a = i[0] + i[3]
b = NN(i[4])
success, r, t = q.solve(a < b, return_values=[i])
```



### Used Models
To obtain models to query, the `train_models.sh` script will download and train several different models:
- MNIST Classifiers found in `models/mnist`, adapted from [PyTroch Examples](https://github.com/pytorch/examples/tree/master/mnist)
- FashionMNIST Classifiers found in `models/mnist`, adapted from [PyTroch Examples](https://github.com/pytorch/examples/tree/master/mnist)
- CIFAR Classifiers found in `models/cifar` adapted from [PyTorch-Cifar](https://github.com/kuangliu/pytorch-cifar) slightly modified
- GTSRB Classifiers found in `models/gtsrb` are also from based on the above CIFAR classifier
- DCGAN for various models found in `models/dcgan` are from [DCGAN](https://github.com/pytorch/examples/tree/master/dcgan)

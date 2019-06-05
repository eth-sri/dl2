## DL2 querying

DL2 allows to query previously trained neural networks (or other models implemented in PyTorch).

Querying can be done with two different APIs:
- the Querying DSL, introduced in the paper, and
- the Querying API.

Both expose the same features and the DSL is implemented in terms of the API.
Depending on the task at hand one or the other can be easier to use.

### Querying DSL
This uses the same syntax as in the paper to query networks.
The snipped below shows how to run such a query and the file `run.py` uses the DSL to run the queries from the evaluation section of the paper.

#### Example usage
``` python
import dl2lib as dl2
import dl2lib.query as q
from configargparse import ArgumentParser

qtext = """FIND i[10]
S.T. i[0] + i[3] < NN(i[4])""" # actualy query
context = dict() # a python dict that defines models and named constants
context['NN'] = myPytrochModel
parser = ArgumentParser(description='DL2 Querying')
parser = dl2.add_default_parser_args(parser, query=True)
args = parser.parse_args() # additionaly arguments such as timeout etc.
success, result, time = q.Query(qtext, context=context, args=args).run()
```

Note that the `class(NN(x)) = y` constraint from the paper can also be written as `class(NN(x), y)`. Both versions are equivalent and the first is just syntactic sugar for the second.


### Querying API
The querying API allows a lower-level access to DL2 while providing access to the same constructs as the DSL.
For an example see the snipped below and queries 1 and 2 from the "Additional Experiments" in the Appendix of the paper in `run_additional.py`.

#### Example usage
``` python
import dl2lib as dl2
import dl2lib.query as q
from configargparse import ArgumentParser

parser = ArgumentParser(description='DL2 Querying')
parser = dl2.add_default_parser_args(parser, query=True)
args = parser.parse_args() # additionaly arguments such as timeout etc.
NN = q.Model(myPytrochModel)
i = q.Variable('i', (10,))
a = i[0] + i[3]
b = NN(i[4])
success, r, t = q.solve(a < b, return_values=[i])
```


### Installation and Experiments
`train_models.sh` downloads and trains several neural networks on common classification tasks. This might take a long time.
Many of the models use existing implementations and datasets. See section "Used Models" below for details.

Alternatively you can also skip this step and [download](https://drive.google.com/file/d/17br5Y9Ta-dxoFqFw83Qtcpdhr1JLGfa_/view?usp=sharing) our provided models. We recommend placing the file `models.zip` in this folder and running `unzip models.zip` from the command line. Additionally you will need to run `bash download_data.sh` to download the GTSRB data. Note, that we can not provide you with Imagenet data samples. You will need to follow the instructions in `train_models.sh` to download them yourself.

Once this has been run and the models are successfully trained  `run.py` can be used to perform the experiments from the evaluation section of the paper.
`run_additional.py` performs the experiments found in the appendix of the paper.


### Used Models
To obtain models to query, the `train_models.sh` script will download and train several different models:
- MNIST Classifiers found in `models/mnist`, adapted from [PyTroch Examples](https://github.com/pytorch/examples/tree/master/mnist)
- FashionMNIST Classifiers found in `models/mnist`, adapted from [PyTroch Examples](https://github.com/pytorch/examples/tree/master/mnist)
- CIFAR Classifiers found in `models/cifar` adapted from [PyTorch-Cifar](https://github.com/kuangliu/pytorch-cifar) slightly modified
- GTSRB Classifiers found in `models/gtsrb` are also from based on the above CIFAR classifier
- DCGAN for various models found in `models/dcgan` are from [DCGAN](https://github.com/pytorch/examples/tree/master/dcgan)

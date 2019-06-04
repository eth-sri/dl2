# DL2: Training and Querying Neural Networks with Logic

<img width="100" alt="portfolio_view" align="left" src="https://www.sri.inf.ethz.ch/assets/systems/dl2-logo.png"><a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a><br />




DL2 is a framework that allows training neural networks with logical constraints over numerical values in the network (e.g. inputs, outputs, weights) and to query networks for inputs fulfilling a logical formula. An example query is shown [below](#example-query). For more details read [training/README.md](https://github.com/eth-sri/dl2/tree/master/training) and [querying/README.md](https://github.com/eth-sri/dl2/tree/master/querying).

This is implementation of DL2 can be used as a library compatible with PyTorch and can be used to reproduce the results of the DL2 [research paper](https://www.sri.inf.ethz.ch/publications/fischer2019dl2).

## Example query

```
FIND i[100]
WHERE i[:] in [-1, 1],
      class(NN1(GEN(i)), 1),
      class(NN2(GEN(i)), 2),
RETURN GEN(i)
```

This example query, spans 3 networks: a generator `GEN` and two classifiers `NN1` and `NN2`. It looks for a noise input (a 100-dimensional vector where all values are between -1 and 1) to the generator such, that it creates an input that gets classifies to class 1 by `NN1` and class 2 by `NN2`. Finally the generated input is returned.

## Structure

 ```
.
├── README.md              - this  file
├── dl2lib                 - DL2 Library
├── training               - the experiments for training networks
│   ├── README.md          - more details on training networks with DL2
│   ├── semisupservised
│   │   ├── main.py        - script to run the semi-supervised experiments
│   │   └── run.sh         - replicates the experiments from the paper
│   ├── supervised
│   │   ├── main.py        - script to run the supervised experiments
│   │   ├── results.py     - creates the tables and plots for the paper
│   │   └── run.sh         - replicates the experiments from the paper
│   └── unsupervised
│       ├── setup.sh       - installs prerequisite libraries
│       ├── run.sh         - replicates the experiments from the paper
│       └── train_DL2.py   - script to run the unsupervised experiments
├── querying               - the experiments for querying networks
│   ├── README.md          - more details on querying networks with DL2
│   ├── run.py             - runs the querying experiments from the paper
│   ├── run_additional.py  - runs the additional querying experiments from the appendix
│   └── train_models.sh    - downloads and trains the models required for the queries
└── requirements.txt       - pip requirements

```

Some files omitted.

## Installation
DL2 was developed and tested with with `python 3.6`, but should also work with newer versions.
All requirements can be installed via `pip install -r requirements.txt`.
Afterwards the folder dl2lib can be imported as a python library (for details see examples).

## Reproducing Results and Examples
For examples see the files in`training` and `querying`, which implement the experiments from the paper.
Each folder contains it's own `README.md` with further instructions on how to use the library and on how to reproduce the results.

## Paper
[website](https://www.sri.inf.ethz.ch/publications/fischer2019dl2)
[pdf](https://files.sri.inf.ethz.ch/website/papers/icml19-dl2.pdf)
```
@inproceedings{fischer2019dl2, title={DL2: Training and Querying Neural Networks with Logic}, author={Marc Fischer, Mislav Balunovic, Dana Drachsler-Cohen, Timon Gehr, Ce Zhang, Martin Vechev}, booktitle={International Conference on Machine Learning}, year={2019}}
```
If you are using the library please also use the above citation to reference this work.


## Contributors

- [Marc Fischer](https://www.sri.inf.ethz.ch/people/marc)
- [Mislav Balunović](https://www.sri.inf.ethz.ch/people/mislav)
- [Dana Drachsler-Cohen](https://www.sri.inf.ethz.ch/people/dana)
- [Timon Gehr](https://www.sri.inf.ethz.ch/people/timon)
- [Ce Zhang](https://ds3lab.org/people/czhang.html)
- [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)

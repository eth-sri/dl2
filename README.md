# DL2: Training and Querying Neural Networks with Logic

This is the implementation of DL2, that can be used as a library compatible with pytorch and that can be used to reproduce the results of the paper.
The library allows training neural networks with logical constraints over numerical values in the network (e.g. inputs, outputs, weights) and to query networks for inputs fulfilling a logical formula.

## Structure

 ```
.
├── README.md              - this  file
├── dl2lib                 - DL2 Library containing
├── training               - the experiments for training networks
│   ├── README.md          - more details on training networks with DL2
│   ├── semisupservised
│   │   ├── main.py        - script to run the semi-supervised experiments
│   │   └── run.sh         - calls the main.py script with the arguments to replicate the experiments from the paper
│   ├── supervised
│   │   ├── main.py        - script to run the supervised experiments
│   │   ├── results.py     - takes the results-logs resulting from main.py and creates the tables and plots for the paper
│   │   └── run.sh         - calls the other scripts with the arguments to replicate the experiments from the paper
│   └── unsupervised
│       ├── setup.sh       - installs prerequisite libraries
│       ├── run.sh         - calls the train.py script with the arguments to replicate the experiments from the paper
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
```
@inproceedings{fischer2019dl2, title={DL2: Training and Querying Neural Networks with Logic}, author={Marc Fischer, Mislav Balunovic, Dana Drachsler-Cohen, Timon Gehr, Ce Zhang, Martin Vechev}, booktitle={International Conference on Machine Learning}, year={2019}}
```
If you are using the library please also use the above citation to reference this work.


## Contributors

- [Marc Fischer](https://marcfischer.at) 
- [Mislav Balunović](https://www.sri.inf.ethz.ch/people/mislav)

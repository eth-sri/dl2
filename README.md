# Code for DL2

We present a DL2, a system for training and querying neural networks with logical constraints.
This is implementation of DL2, that can be used as a library compatible with pytorch.
It allows training neural networks with logical constraints over numerical values in the network (e.g. inputs, outputs, weights) and to query networks for inputs fulfilling a logical formula.

## Structure

``` 
.
├── README.md        - This readme file
├── dl2lib           - Library containing the DL2 loss
├── training         - the experiments for training networks
├── querying         - the experiments for querying networks
└── requirements.txt - pip requirements
```

## Insallation
Run with `python 3.6`.
Install requirements `pip install -r requirements.txt`.

Afterwards the folder dl2lib can be imported as a python library.

## Examples
For examples the files in`training` and `querying`, which implement the experiments from the paper.
Each folder contains it's own README.md with further instructions.

## Contributors

- [Marc Fischer](https://marcfischer.at) 
- [Mislav Balunovic](https://www.sri.inf.ethz.ch/people/mislav)

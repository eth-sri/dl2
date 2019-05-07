# DL2: Training and Querying Neural Networks with Logic

This is implementation of DL2, that can be used as a library compatible with pytorch and that can be used to reproduce the results of the paper.
The library allows training neural networks with logical constraints over numerical values in the network (e.g. inputs, outputs, weights) and to query networks for inputs fulfilling a logical formula.

## Structure

``` 
.
├── README.md        - This readme file
├── dl2lib           - Library containing the DL2 loss
├── training         - the experiments for training networks
├── querying         - the experiments for querying networks
└── requirements.txt - pip requirements
```

## Installation
Run with `python 3.6`.
Install requirements `pip install -r requirements.txt`.

Afterwards the folder dl2lib can be imported as a python library.

## Reproducing Results and Examples
For examples see the files in`training` and `querying`, which implement the experiments from the paper.
Each folder contains it's own README.md with further instructions on how to use the library and on how to reproduce the results.

## Paper
```
@inproceedings{fischer2019dl2, title={DL2: Training and Querying Neural Networks with Logic}, author={Marc Fischer, Mislav Balunovic, Dana Drachsler-Cohen, Timon Gehr, Ce Zhang, Martin Vechev}, booktitle={International Conference on Machine Learning}, year={2019}}
```
If you are using the library please also use the above citation to reference this work.


## Contributors

- [Marc Fischer](https://marcfischer.at) 
- [Mislav Balunovic](https://www.sri.inf.ethz.ch/people/mislav)

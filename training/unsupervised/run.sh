#!/bin/bash

# Run DL2
python train_DL2.py --epochs 1500 --lr 0.0001 --hidden 1000 --dropout 0.3 --n_train 300 --n_valid 150

# Run supervised baseline
python train_DL2.py --epochs 1500 --lr 0.0001 --hidden 1000 --dropout 0.3 --n_train 300 --n_valid 150 --baseline True

# Both versions also evaluate the naive baseline

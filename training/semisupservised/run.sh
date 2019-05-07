#!/bin/bash

python main.py --lr 0.001 --net_type vgg --constraint none --epochs 1600  --num_labeled 100 --dataset cifar100 --exp_name baseline --constraint-weight 0
python main.py --lr 0.001 --net_type vgg --constraint none --epochs 1600  --num_labeled 100 --dataset cifar100 --exp_name dl2_06 --constraint-weight 0.6

#!/bin/bash

# train model
python main.py --lr 0.001 --net_type vgg --constraint none --epochs 1600  --num_labeled 100 --dataset cifar100 --exp_name baseline --constraint-weight 0
python main.py --lr 0.001 --net_type vgg --constraint none --epochs 1600  --num_labeled 100 --dataset cifar100 --exp_name dl2_06 --constraint-weight 0.6

# run evaluation
python main.py --net_type vgg --dataset cifar100 --resume_from vgg_42_baseline_overall_  --testOnly
python main.py --net_type vgg --dataset cifar100 --resume_from vgg_42_dl2_06_overall_  --testOnly

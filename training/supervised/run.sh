#!/bin/bash

# MNIST DL2
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.2 --dataset mnist  --constraint "RobustnessT(eps1=7.8, eps2=2.9)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.2 --dataset mnist  --constraint "RobustnessG(eps=0.3, delta=0.52)" --report-dir . --num-iters 50
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.1 --dataset mnist  --constraint "LipschitzT(L=0.1)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.2 --dataset mnist  --constraint "LipschitzG(L=0.1, eps=0.3)" --report-dir . --num-iters 5
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.01 --dataset mnist  --constraint "SegmentG(eps=100.0, delta=1.5)" --report-dir . --num-iters 5 --embed

# MNIST Baselines
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset mnist  --constraint "RobustnessT(eps1=7.8, eps2=2.9)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset mnist  --constraint "RobustnessG(eps=0.3, delta=0.52)" --report-dir . --num-iters 50
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset mnist  --constraint "LipschitzT(L=0.1)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset mnist  --constraint "LipschitzG(L=0.1, eps=0.3)" --report-dir . --num-iters 5
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset mnist  --constraint "SegmentG(eps=100.0, delta=1.5)" --report-dir . --num-iters 5 --embed

# FMNIST DL2
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.2 --dataset fashion  --constraint "RobustnessT(eps1=7.8, eps2=2.9)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.2 --dataset fashion  --constraint "RobustnessG(eps=0.3, delta=0.52)" --report-dir . --num-iters 50
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.1 --dataset fashion  --constraint "LipschitzT(L=0.1)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.2 --dataset fashion  --constraint "LipschitzG(L=0.1, eps=0.3)" --report-dir . --num-iters 50
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.01 --dataset fashion --constraint "SegmentG(eps=100.0, delta=1.5)" --report-dir . --num-iters 5 --embed

# FMNIST Baselines
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset fashion  --constraint "RobustnessT(eps1=7.8, eps2=2.9)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset fashion  --constraint "RobustnessG(eps=0.3, delta=0.52)" --report-dir . --num-iters 50
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset fashion  --constraint "LipschitzT(L=0.1)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset fashion  --constraint "LipschitzG(L=0.1, eps=0.3)" --report-dir . --num-iters 50
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset fashion --constraint "SegmentG(eps=100.0, delta=1.5)" --report-dir . --num-iters 5 --embed

# CIFAR DL2
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.04 --dataset cifar10  --constraint "RobustnessT(eps1=13.8, eps2=0.9)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.1 --dataset cifar10  --constraint "RobustnessG(eps=0.3, delta=0.52)" --report-dir . --num-iters 7
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.1 --dataset cifar10  --constraint "LipschitzT(L=1.0)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.1 --dataset cifar10  --constraint "LipschitzG(L=1.0, eps=0.3)" --report-dir . --num-iters 5
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.2 --dataset cifar10  --constraint "CSimilarityT(delta=0.01)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0.2 --dataset cifar10  --constraint "CSimilarityG(delta=1.0, eps=0.3)" --report-dir . --num-iters 10

# CIFAR Baselines
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "RobustnessT(eps1=13.8, eps2=0.9)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "RobustnessG(eps=0.3, delta=0.52)" --report-dir . --num-iters 7
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "LipschitzT(L=1.0)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "LipschitzG(L=1.0, eps=0.3)" --report-dir . --num-iters 5
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "CSimilarityT(delta=0.01)" --report-dir .
python cegis_train_constraint.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "CSimilarityG(delta=1.0, eps=0.3)" --report-dir . --num-iters 10


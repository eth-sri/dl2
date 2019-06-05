#!/bin/bash

bash download_data.sh

cd models


# MNIST classifier
cd mnist
python main.py --save-model --net 1 --dataset mnist > log_mnist1.txt
python main.py --save-model --net 2 --dataset mnist > log_mnist2.txt
cd ..


# FashionMNIST classifier
cd mnist
python main.py --save-model --net 1 --dataset fashionmnist > log_fashionmnist1.txt
python main.py --save-model --net 2 --dataset fashionmnist > log_fashionmnist2.txt
cd ..

# GTSRB classifier
cd gtsrb
python sort_data.py
python gtsrb.py --net vgg16 --lr 0.1 --epochs 150
python gtsrb.py --net vgg16 --lr 0.01 --epochs 100 --resume
python gtsrb.py --net vgg16 --lr 0.001 --epochs 100 --resume
python gtsrb.py --net resnet18 --lr 0.1 --epochs 150
python gtsrb.py --net resnet18 --lr 0.01 --epochs 100 --resume
python gtsrb.py --net resnet18 --lr 0.001 --epochs 100 --resume
cd ..


# Cifar classifier
cd cifar
python main.py --net vgg16 --lr 0.1 --epochs 150
python main.py --net vgg16 --lr 0.01 --epochs 100 --resume
python main.py --net vgg16 --lr 0.001 --epochs 100 --resume
python main.py --net resnet18 --lr 0.1 --epochs 150
python main.py --net resnet18 --lr 0.01 --epochs 100 --resume
python main.py --net resnet18 --lr 0.001 --epochs 100 --resume
cd ..

# DCGAN for MNIST, FashionMNIST, CIFAR
cd dcgan
rm -rf *.pth
rm -rf *.png
python main.py --dataset cifar10 --dataroot ../../../data/cifar --cuda > log_gen_cifar10.txt
mkdir -p cifar
mv *.pth cifar
mv *.png cifar
python main.py --dataset mnist  --dataroot ../../../data/mnist --cuda > log_gen_mnist.txt
mkdir -p mnist
mv *.pth mnist
mv *.png mnist
python main.py --dataset fashionmnis  --dataroot ../../../data/fashionmnist --cuda > log_gen_fashionmnist.txt
mkdir -p fashionmnist
mv *.pth fashionmnist
mv *.png fashionmnist
python main.py --dataset folder  --dataroot ../../../data/GTSRB/Final_Training --cuda > log_gen_gtsrb.txt
mkdir -p gtsrb
mv *.pth gtsrb
mv *.png gtsrb
cd ..


echo "Since we can not download ImageNet for you. You will need to download ImageNet yourself and place a selection of it in the folder models/Imagenet_selection. See file Imagenet.md for the correct layout:"
cat Imagenet.md

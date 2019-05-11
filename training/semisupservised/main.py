from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import itertools
import os
import sys
import time
import argparse
import datetime
import pickle
import numpy as np
import sys
import math
sys.path.append('wide-resnet.pytorch')
import config as cf

from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append('../../')
import dl2lib as dl2
import random
from resnet import ResNet18
from vgg import VGG

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser = dl2.add_default_parser_args(parser)
parser.add_argument('--seed', default=42, type=int, help='Random seed to use.')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--growing', default=0, type=int, help='epochs')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--exp_name', default='', type=str, help='experiment name')
parser.add_argument('--resume_from', type=str, default=None, help='resume from checkpoint')
parser.add_argument('--testOnly', action='store_true', help='Test mode with the saved model')
parser.add_argument('--constraint', type=str, choices=['DL2', 'none'], default='none', help='constraint system to use')
parser.add_argument('--constraint-weight', '--constraint_weight', type=float, default=0.6, help='weight for constraint loss')
parser.add_argument('--num_labeled', default=1000, type=int, help='Number of labeled examples (per class!).')
parser.add_argument('--skip_labled', default=0, type=int, help='Number of labeled examples (per class!).')
parser.add_argument('--decrease-eps-weight', default=1.0, type=float, help='Number of labeled examples (per class!).')
parser.add_argument('--c-eps', default=0.05, type=float, help='Number of labeled examples (per class!).')
parser.add_argument('--increase-constraint-weight', default=1.0, type=float, help='Number of labeled examples (per class!).')

args = parser.parse_args()
args.growing = bool(args.growing)
args.skip_labled = bool(args.skip_labled)
torchvision.datasets.CIFAR100(root='../../data/cifar100', train=True, download=True)
meta = pickle.load(open('../../data/cifar100/cifar-100-python/meta', 'rb'))
coarse = meta['coarse_label_names']
fine = meta['fine_label_names']

label_idx = {label:i for i, label in enumerate(fine)}
group_idx = {label:i for i, label in enumerate(coarse)}
g = {}
group = [0 for i in range(100)]
pairs = []

print(group_idx)

with open('groups.txt') as f:
    for line in f:
        tokens = line[:-1].split('\t')
        large_group = tokens[0]
        tokens[1] = tokens[1].replace(',', '').strip()
        labels = tokens[1].split(' ')
        assert len(labels) == 5, labels
        
        for label in labels:
            assert label in fine, label
            group[label_idx[label]] = group_idx[large_group]

        g[group_idx[large_group]] = [label_idx[label] for label in labels]
        
        for x in labels:
            for y in labels:
                if x != y:
                    pairs.append((label_idx[x], label_idx[y]))

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
print(use_cuda)
best_acc = 0
best_model = None
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
num_epochs = args.epochs
# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='../../data/cifar100', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../../data/cifar100', train=False, download=False, transform=transform_test)
    num_classes = 100

num_train = len(trainset)

per_class = [[] for _ in range(100)]
for i in range(num_train):
    per_class[trainset[i][1]].append(i)

train_lab_idx = []
train_unlab_idx = []
valid_idx = []
    
np.random.seed(args.seed)
torch.manual_seed(args.seed)
for i in range(100):
    np.random.shuffle(per_class[i])
    split = int(np.floor(0.2 * len(per_class[i])))
    train_lab_idx += per_class[i][split:split+args.num_labeled]
    train_unlab_idx += per_class[i][split+args.num_labeled:]
    valid_idx += per_class[i][:split]

print('Total train[labeled]: ',len(train_lab_idx))
print('Total train[unlabeled]: ',len(train_unlab_idx))
print('Total valid: ',len(valid_idx))

train_labeled_sampler = SubsetRandomSampler(train_lab_idx)
train_unlabeled_sampler = SubsetRandomSampler(train_unlab_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

unlab_batch = batch_size if args.constraint != 'none' else 1

trainloader_lab = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, sampler=train_labeled_sampler, num_workers=2)
trainloader_unlab = torch.utils.data.DataLoader(
    trainset, batch_size=unlab_batch, sampler=train_unlabeled_sampler, num_workers=2)
validloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, sampler=valid_sampler, num_workers=2)

def getNetwork(args):
    if args.net_type == 'resnet':
        net = ResNet18(100)
        file_name = 'resnet18'
    elif args.net_type == 'vgg':
        net = VGG('VGG16', 100)
        file_name = 'vgg'
    else:
        assert False
    file_name += '_' + str(args.seed) + '_' + args.exp_name
    return net, file_name

# Test only option
if (args.testOnly):
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/' + args.resume_from + '.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    constraint_correct = 0
    total = 0

    conf_mat = np.zeros((100, 100))
    group_ok = 0

    np.set_printoptions(threshold=np.inf)
    softmax = torch.nn.Softmax()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        
        probs = softmax(outputs)
        eps = 0.05
        dl2_one_group = []
        for i in range(20):
            gsum = 0
            for j in g[i]:
                gsum += probs[:, j]
            dl2_one_group.append(dl2.Or([dl2.GT(gsum, 1.0 - eps), dl2.LT(gsum, eps)]))
        constraint = dl2.And(dl2_one_group)
        constraint_correct += constraint.satisfy(args).sum()
        
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        conf_mat += confusion_matrix(targets.data.cpu().numpy(), predicted.cpu().numpy(), labels=np.arange(100))

        n_batch = predicted.size()[0]
        for i in range(n_batch):
            if group[predicted.cpu()[i]] == group[targets.cpu().data[i]]:
                group_ok += 1

    #rint('Confusion matrix:')
    #print(conf_mat)
        
    acc = 100.0*float(correct)/total
    c_acc = 100.0*float(constraint_correct)/total
    group_acc = 100.0*float(group_ok)/total
    print("| Test Result\tAcc@1: %.2f%%" %(acc))
    print("| Test Result\tCAcc: %.2f%%" %(c_acc))
    print("| Test Result\tGroupAcc: %.2f%%" %(group_acc))
    
    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume_from is not None:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/' + args.resume_from + '.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    # net.apply(conv_init)


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    softmax = torch.nn.Softmax()

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, args.lr))
    if args.skip_labled:
        tl = [None] * 200000
    else:
        tl = trainloader_lab
    for batch_idx, (lab, ulab) in enumerate(zip(tl, trainloader_unlab)):
        inputs_u, targets_u = ulab
        inputs_u, targets_u = Variable(inputs_u), Variable(targets_u)
        n_u = inputs_u.size()[0]
        if use_cuda:
            inputs_u, targets_u = inputs_u.cuda(), targets_u.cuda() # GPU settings

        if lab is None:
            n = 0
            all_outputs = net(inputs_u)
        else:
            inputs, targets = lab
            inputs, targets = Variable(inputs), Variable(targets)
            n = inputs.size()[0]
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
            all_outputs = net(torch.cat([inputs, inputs_u], dim=0))

        optimizer.zero_grad()
        outputs_u = all_outputs[n:,]
        logits_u = F.log_softmax(outputs_u)
        probs_u = softmax(outputs_u)

        outputs = all_outputs[:n,]
        if args.skip_labled:
            ce_loss = 0
        else:
            outputs = all_outputs[:n,]
            ce_loss = criterion(outputs, targets)  # Loss
        
        constraint_loss = 0
        if args.constraint == 'DL2':
            eps = args.c_eps * args.decrease_eps_weight**epoch
            dl2_one_group = []
            for i in range(20):
                gsum = 0
                for j in g[i]:
                    gsum += probs_u[:,j]
                dl2_one_group.append(dl2.Or([dl2.EQ(gsum, 1.0), dl2.EQ(gsum, 0.0)]))
            dl2_one_group = dl2.And(dl2_one_group)
            dl2_loss = dl2_one_group.loss(args).mean()
            constraint_loss = dl2_loss
            loss = ce_loss + (args.constraint_weight * args.increase_constraint_weight**epoch) * dl2_loss
        else:
            loss = ce_loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update
        
        train_loss += loss.item()
        if args.skip_labled:

            total = 1
            correct = 0
        else:
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tCE Loss: %.4f, Constraint Loss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(train_lab_idx)//batch_size)+1, loss, constraint_loss, 100.*float(correct)/total))
        sys.stdout.flush()
    return 100.*float(correct)/total

def save(acc, e, net, best=False):
    state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if best:
        e = int(100* math.floor(( float(epoch) / 100)) )
        save_point = './checkpoint/' + file_name + '_' + str(e) + '_best' + '.t7'
    else:
        save_point = './checkpoint/' + file_name + '_' + str(e) + '_' + '.t7'
    torch.save(state, save_point)
    return net
    
        
def test(epoch):
    global best_acc, best_model
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(validloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*float(correct)/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

    if acc > best_acc:
        #print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        best_model = save(acc, num_epochs, net, best=True)
        best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    acc = train(epoch)
    if epoch % 100 == 0:
        save(acc, epoch, net)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

if best_model is not None:
    print('.')
    save(best_acc, 'overall',  best_model)
    
print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))

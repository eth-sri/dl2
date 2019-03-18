import torch
import torch.nn as nn
import torch.nn.functional as F
import models.mnist.main as mnist
import torchvision.models as models
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import models.dcgan.main as dcgan
import models.cifar.models as cifarmodels
import glob
import PIL


def get_context(args):
    context = dict()
    # Load MNIST classifiers

    class MNIST_Normalize(nn.Module):
        def forward(self, x):
            return (x - 0.1307) / 0.3081

    curr_dir = os.path.dirname(__file__)
    M_NN1 = mnist.Net1()
    M_NN1.load_state_dict(torch.load(os.path.join(curr_dir, 'models/mnist/mnist_1.pt'), map_location='cpu'))
    M_NN1 = nn.Sequential(MNIST_Normalize(), M_NN1)
    M_NN1.eval()

    M_NN2 = mnist.Net2()
    M_NN2.load_state_dict(torch.load(os.path.join(curr_dir, 'models/mnist/mnist_2.pt'), map_location='cpu'))
    M_NN2 = nn.Sequential(MNIST_Normalize(), M_NN2)
    M_NN2.eval()
    
    if args.cuda:
        M_NN1.to('cuda:0')
        M_NN2.to('cuda:0')
    
    context['M_NN1'] = M_NN1
    context['M_NN2'] = M_NN2

    # Load MNIST generator/discriminator

    class MNIST_Upsample(nn.Module):
        def forward(self, x):
            x =  F.interpolate(x, size=[64, 64], mode='bilinear')
            return x

    class MNIST_Downsample(nn.Module):
        def forward(self, x):
            return F.interpolate(x, size=[28, 28], mode='bilinear')

    mnistD = dcgan.Discriminator(0, nc=1)
    mnistD.load_state_dict(torch.load(os.path.join(curr_dir, 'models/dcgan/mnist/netD_epoch_13.pth'), map_location='cpu'))
    mnistD = nn.Sequential(MNIST_Upsample(), mnistD)
    mnistD.eval()

    
    mnistG = dcgan.Generator(0, nc=1)
    mnistG.load_state_dict(torch.load(os.path.join(curr_dir, 'models/dcgan/mnist/netG_epoch_13.pth'), map_location='cpu'))
    mnistG = nn.Sequential(mnistG, MNIST_Downsample())
    mnistG.eval()

    if args.cuda:
        mnistD.to('cuda:0')
        mnistG.to('cuda:0')

    context['M_D'] = mnistD
    context['M_G'] = mnistG


    # Load MNIST variables

    mask = np.ones((1, 1, 28, 28), dtype=bool)
    mask[0, 0, 16:20, 9:16] = False
    mask[0, 0, 20:27, 9:16] = False
    context['M_mask'] = np.logical_not(mask)
    context['M_not_mask'] = mask

    keys = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    loaded = [False for k in keys]
    data = datasets.MNIST('../data/mnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    data_iter = iter(data)

    while not all(loaded):
        img, label = next(data_iter)
        label = int(label)
        if not loaded[label]:
            name = "M_" + keys[int(label)]
            context[name] = img.numpy().reshape((1, 1, 28, 28))
            context[name + "_not_mask"] = context[name][context['M_not_mask']]
            context[name + "_mask"] = context[name][context['M_mask']]
            loaded[label] = True



    # # Load FashionMNIST classifiers
    FM_NN1 = mnist.Net1()
    FM_NN1.load_state_dict(torch.load(os.path.join(curr_dir, 'models/mnist/fashionmnist_1.pt'), map_location='cpu'))
    FM_NN1.eval()

    FM_NN2 = mnist.Net2()
    FM_NN2.load_state_dict(torch.load(os.path.join(curr_dir, 'models/mnist/fashionmnist_2.pt'), map_location='cpu'))
    FM_NN2.eval()

    if args.cuda:
        FM_NN1.to('cuda:0')
        FM_NN2.to('cuda:0')
    
    context['FM_NN1'] = FM_NN1
    context['FM_NN2'] = FM_NN2

    # Load FashionMNIST variables
    context['FM_mask'] = context['M_mask']
    context['FM_not_mask'] = context['M_not_mask']

    keys = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']
    loaded = [False for k in keys]
    data = datasets.FashionMNIST('../data/fashionmnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    data_iter = iter(data)

    while not all(loaded):
        img, label = next(data_iter)
        label = int(label)
        if not loaded[label]:
            name = "FM_" + keys[int(label)]
            context[name] = img.numpy().reshape((1, 1, 28, 28))
            context[name + "_not_mask"] = context[name][context['FM_not_mask']]
            context[name + "_mask"] = context[name][context['FM_mask']]
            loaded[label] = True

    # Load FashionMNIST generator/discriminator
    fmnistD = dcgan.Discriminator(0, nc=1)
    fmnistD.load_state_dict(torch.load(os.path.join(curr_dir, 'models/dcgan/fashionmnist/netD_epoch_13.pth'), map_location='cpu'))
    fmnistD = nn.Sequential(MNIST_Upsample(), fmnistD)
    fmnistD.eval()
    fmnistG = dcgan.Generator(0, nc=1)
    fmnistG.load_state_dict(torch.load(os.path.join(curr_dir, 'models/dcgan/fashionmnist/netG_epoch_13.pth'), map_location='cpu'))
    fmnistG = nn.Sequential(fmnistG, MNIST_Downsample())
    fmnistG.eval()

    if args.cuda:
        fmnistD.to('cuda:0')
        fmnistG.to('cuda:0')
    
    context['FM_D'] = fmnistD
    context['FM_G'] = fmnistG
   
    # Load Imagenet data
    mask = np.zeros((1, 3, 224, 224), dtype=bool)
    mask[0, :, 0:112, 0:112] = True
    context['I_not_mask'] = np.logical_not(mask)
    context['I_mask'] = mask

    images = glob.glob('./models/Imagenet_selection/*/*.jpeg')
    for image in images:
        image_data = PIL.Image.open(image).convert('RGB')
        image_data = transforms.CenterCrop((224, 224))(image_data)
        image_data = transforms.ToTensor()(image_data).numpy()
        key, _ = os.path.split(image)
        _, key = os.path.split(key)
        name = "I_" + key
        context[name] = image_data.reshape((1, 3, 224, 224))
        context[name + "_not_mask"] = context[name][context['I_not_mask']]
        context[name + "_mask"] = context[name][context['I_mask']]
        loaded[label] = True
   

    # Load Imagenet classifiers
    class Imagenet_Normalize(nn.Module):

        def forward(self, x):
            bias = torch.ones_like(x)
            bias[:, 0, :, :] *= 0.485
            bias[:, 1, :, :] *= 0.456
            bias[:, 2, :, :] *= 0.406
            std = torch.ones_like(x)
            std[:, 0, :, :] *= 0.229
            std[:, 1, :, :] *= 0.224
            std[:, 2, :, :] *= 0.225
            return (x - bias) / std

    resnet50 = models.resnet50(pretrained=True)
    resnet50 = nn.Sequential(Imagenet_Normalize(), resnet50)
    resnet50.eval()
    vgg16 = models.vgg16(pretrained=True)
    vgg16 = nn.Sequential(Imagenet_Normalize(), vgg16)
    vgg16.eval()
    vgg19 = models.vgg19(pretrained=True)
    vgg19 = nn.Sequential(Imagenet_Normalize(), vgg19)
    vgg19.eval()

    if args.cuda:
        vgg16.to('cuda:0')
        vgg19.to('cuda:0')
        resnet50.to('cuda:0')
    
    context['I_VGG16'] = vgg16
    context['I_VGG19'] = vgg19
    context['I_R50'] = resnet50

    # # Load CIFAR classifiers
    class CIFAR_Normalize(nn.Module):
        def forward(self, x):
            bias = torch.ones_like(x)
            bias[:, 0, :, :] *= 0.4914
            bias[:, 1, :, :] *= 0.4822
            bias[:, 2, :, :] *= 0.4465
            std = torch.ones_like(x)
            std[:, 0, :, :] *= 0.2023
            std[:, 1, :, :] *= 0.1994
            std[:, 2, :, :] *= 0.2010
            return (x - bias) / std

    cifar_vgg = cifarmodels.VGG('VGG16')
    state = torch.load(os.path.join(curr_dir, 'models/cifar/checkpoint/vgg16.pt'), map_location='cpu')
    state_dict = dict([(a.replace('module.', ''), b) for a, b in state['net'].items()])
    cifar_vgg.load_state_dict(state_dict)
    cifar_vgg = nn.Sequential(CIFAR_Normalize(), cifar_vgg)
    cifar_vgg.eval()
    resnet = cifarmodels.ResNet18()
    state = torch.load(os.path.join(curr_dir, 'models/cifar/checkpoint/resnet18.pt'), map_location='cpu')
    state_dict = dict([(a.replace('module.', ''), b) for a, b in state['net'].items()])
    resnet.load_state_dict(state_dict)
    resnet = nn.Sequential(CIFAR_Normalize(), resnet)
    resnet.eval()

    if args.cuda:
        cifar_vgg.to('cuda:0')
        resnet.to('cuda:0')

    context['C_VGG'] = cifar_vgg
    context['C_RESNET'] = resnet
    
    # Load CIFAR variables

    keys = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    mask = np.zeros((1, 3, 32, 32), dtype=bool)
    mask[0, :, 0:16, 0:16] = True
    context['C_not_mask'] = np.logical_not(mask)
    context['C_mask'] = mask

    loaded = [False for k in keys]
    data = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    data_iter = iter(data)

    while not all(loaded):
        img, label = next(data_iter)
        label = int(label)
        if not loaded[label]:
            name = "C_" + keys[label]
            context[name] = img.numpy().reshape((1, 3, 32, 32))
            context[name + "_not_mask"] = context[name][context['C_not_mask']]
            context[name + "_mask"] = context[name][context['C_mask']]
            loaded[label] = True

    # Load CIFAR generator/discriminator

    class CIFAR_Upsample(nn.Module):
        def forward(self, x):
            x =  F.interpolate(x, size=[64, 64], mode='bilinear')
            return x

    class CIFAR_Downsample(nn.Module):
        def forward(self, x):
            return F.interpolate(x, size=[32, 32], mode='bilinear')

    cifarD = dcgan.Discriminator(0, nc=3)
    cifarD.load_state_dict(torch.load(os.path.join(curr_dir, 'models/dcgan/cifar10/netD_epoch_13.pth'), map_location='cpu'))
    cifarD = nn.Sequential(CIFAR_Upsample(), cifarD)
    cifarD.eval()
    cifarG = dcgan.Generator(0, nc=3)
    cifarG.load_state_dict(torch.load(os.path.join(curr_dir, 'models/dcgan/cifar10/netG_epoch_13.pth'), map_location='cpu'))
    cifarG = nn.Sequential(cifarG, CIFAR_Downsample())
    cifarG.eval()

    if args.cuda:
        cifarD.to('cuda:0')
        cifarG.to('cuda:0')
    
    context['C_D'] = cifarD
    context['C_G'] = cifarG


    # Load GTSRB classifiers
    gtsrb_vgg = cifarmodels.VGG('VGG16', num_classes=43)
    state = torch.load(os.path.join(curr_dir, 'models/gtsrb/checkpoint/vgg16.pt'), map_location='cpu')
    state_dict = dict([(a.replace('module.', ''), b) for a, b in state['net'].items()])
    gtsrb_vgg.load_state_dict(state_dict)
    gtsrb_vgg.eval()
    resnet = cifarmodels.ResNet18(num_classes=43)
    state = torch.load(os.path.join(curr_dir, 'models/gtsrb/checkpoint/resnet18.pt'), map_location='cpu')
    state_dict = dict([(a.replace('module.', ''), b) for a, b in state['net'].items()])
    resnet.load_state_dict(state_dict)
    resnet.eval()

    if args.cuda:
        gtsrb_vgg.to('cuda:0')
        resnet.to('cuda:0')

    context['G_VGG'] = gtsrb_vgg
    context['G_RESNET'] = resnet


    # Load gtsrb variables
    # label list from https://github.com/magnusja/GTSRB-caffe-model/blob/master/labeller/main.py
    keys = ['20_speed', '30_speed', '50_speed', '60_speed', '70_speed', '80_speed', '80_lifted', '100_speed', '120_speed', 'no_overtaking_general', 'no_overtaking_trucks', 'right_of_way_crossing', 'right_of_way_general', 'give_way', 'stop', 'no_way_general', 'no_way_trucks', 'no_way_one_way', 'attention_general', 'attention_left_turn', 'attention_right_turn', 'attention_curvy', 'attention_bumpers', 'attention_slippery', 'attention_bottleneck', 'attention_construction', 'attention_traffic_light', 'attention_pedestrian', 'attention_children', 'attention_bikes', 'attention_snowflake', 'attention_deer', 'lifted_general', 'turn_right', 'turn_left', 'turn_straight', 'turn_straight_right', 'turn_straight_left', 'turn_right_down', 'turn_left_down', 'turn_circle', 'lifted_no_overtaking_general', 'lifted_no_overtaking_trucks']
    mask = np.zeros((1, 3, 32, 32), dtype=bool)
    mask[0, :, 0:16, 0:16] = True
    context['G_not_mask'] = np.logical_not(mask)
    context['G_mask'] = mask

    loaded = [False for k in keys]
    transform_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    testset = datasets.ImageFolder('../data/GTSRB/Final_Test/Images', transform=transform_test)
    data_iter = iter(testset)

    while not all(loaded):
        img, label = next(data_iter)
        label = int(label)
        if not loaded[label]:
            name = "G_" + keys[label]
            context[name] = img.numpy().reshape((1, 3, 32, 32))
            context[name + "_not_mask"] = context[name][context['G_not_mask']]
            context[name + "_mask"] = context[name][context['G_mask']]
            loaded[label] = True


    # Load GTSRB generator/discriminator
    gtsrbD = dcgan.Discriminator(0, nc=3)
    gtsrbD.load_state_dict(torch.load(os.path.join(curr_dir, 'models/dcgan/gtsrb/netD_epoch_23.pth'), map_location='cpu'))
    gtsrbD = nn.Sequential(CIFAR_Upsample(), gtsrbD)
    gtsrbD.eval()
    gtsrbG = dcgan.Generator(0, nc=3)
    gtsrbG.load_state_dict(torch.load(os.path.join(curr_dir, 'models/dcgan/gtsrb/netG_epoch_23.pth'), map_location='cpu'))
    gtsrbG = nn.Sequential(gtsrbG, CIFAR_Downsample())
    gtsrbG.eval()

    if args.cuda:
        gtsrbD.to('cuda:0')
        gtsrbG.to('cuda:0')
    
    context['G_D'] = gtsrbD
    context['G_G'] = gtsrbG
    return context

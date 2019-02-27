'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from fer import RAF
from torch.autograd import Variable
from models import *
import torch.optim.lr_scheduler as lr_scheduler
from models.xception import xception
from models.resnext import resnext50
# from models.resnetall import resnet18,resnet34,resnet50,resnet101
from ArcModel.resnet import *
from ArcModel.focal_loss import *
from ArcModel.metrics import *
from models.densenet import densenet121,densenet161,densenet169,densenet201
from visdom import Visdom
import numpy as np

from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch RAF CNN Training')
parser.add_argument('--model', type=str, default='Resnet34', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='RAF', help='CNN architecture')
parser.add_argument('--num_class', default=7, type=int, help='batch size')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--valbs', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 50  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 250

# train_path = 'data/secondTrain/raf_fer_train/'
# train_txt = 'data/secondTrain/raf_fer_train.txt'
train_path = 'data/firstTrain/aligned48/'
train_txt = 'data/firstTrain/train.txt'
# train_path = 'data/thirdTrain/aligned100/'
# train_txt = 'data/thirdTrain/raf_train.txt'


# val_path = 'data/secondTrain/raf_val/'
# val_txt = 'data/secondTrain/raf_val.txt'
val_path = 'data/firstTrain/aligned48/'
val_txt = 'data/firstTrain/test.txt'
# val_path = 'data/thirdTrain/aligned100/'
# val_txt = 'data/thirdTrain/raf_test.txt'

# test_path = 'data/secondTrain/fer_test/'
# test_txt = 'data/secondTrain/fer_test.txt'

path = os.path.join(opt.dataset + '_arc_' + opt.model)

# Data aug
# Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
print('==> Preparing data..')

normMean = [0.485, 0.456, 0.406]
normStd = [0.229, 0.224, 0.225]
normTransform = transforms.Normalize(normMean, normStd)
# transforms.RandomRotation(10),  # 在（-10， 10）范围内旋转
# transforms.ColorJitter()
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    # transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # normTransform
])
transform_test = transforms.Compose([
    transforms.CenterCrop(cut_size),
    transforms.ToTensor(),
    # transforms.FiveCrop(cut_size),
    # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    # transforms.Lambda(lambda crops: torch.stack([normTransform(crop) for crop in crops])),
    # normTransform
])



trainset = RAF(train_path,train_txt, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)

valset = RAF(val_path,val_txt, transform=transform_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.valbs, shuffle=False, num_workers=1)

# testset = RAF(test_path,test_txt, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=opt.valbs, shuffle=False, num_workers=1)


# trainset = FER2013(split = 'Training', transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
#
# PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
# PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
#
# PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
# PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)

# Model
if opt.model == 'VGG16':
    net = VGG(opt.model)
    print(net)
elif opt.model  == 'Resnet18':
    net = resnet18(num_classes=7)
elif opt.model == 'Resnet34':
    net = resnet34()#num_classes=7
    print(net)
elif opt.model == 'Resnet50':
    net = resnet50(num_classes=7)
    print(net)
elif opt.model == 'densenet':
    net = densenet121(num_class=7)
elif opt.model == 'xception':
    net = xception(class_nums=7)
    print(net)
elif opt.model == 'Resnext50':
    net = resnext50(class_names=7)
    print(net)

metric_fc = ArcMarginProduct(512, opt.num_class, s=30, m=0.5, easy_margin=False)

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')


if use_cuda:
    net.cuda()
    metric_fc.cuda()
    # summary(net, input_size=(3, 48, 48))
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(gamma=2)
# optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.SGD([{'params': net.parameters()}, {'params': metric_fc.parameters()}],
                                lr=opt.lr, momentum=0.9,weight_decay=5e-4)
# optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': metric_fc.parameters()}],
#                              lr=opt.lr, weight_decay=5e-4)
sheduler = lr_scheduler.StepLR(optimizer,50,gamma=0.1)
# optimizer = optim.Adam(net.parameters(), lr=opt.lr)
# optim = optim.Adam(self.model.parameters(), lr=opts.lr)
# optim = optim.RMSprop(self.model.parameters(), lr=opts.lr)
# Training
def train(epoch):
    print('learning_rate: %s' % str(sheduler.get_lr()))
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        feature = net(inputs)
        outputs = metric_fc(feature, targets)
        # print("output_size", outputs.size())
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_loss += loss.data[0]
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Train_loss = train_loss/(batch_idx+1)
    print('avg Train loss :%.4f'%Train_loss)
    Train_acc = 100.*correct/total
    return Train_loss,Train_acc

def RAFTest_48(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)#, volatile=True
            feature = net(inputs)
            outputs = metric_fc(feature, targets)
            loss = criterion(outputs, targets)
            PublicTest_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(valloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
                               % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    avg_loss = PublicTest_loss / (batch_idx + 1)
    print('val_avg_loss: %.4f'%avg_loss)
    PublicTest_acc = 100.*correct/total
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch
    return avg_loss,PublicTest_acc

def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            bs, ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)#volatile=True
            feature = net(inputs)
            outputs = metric_fc(feature, targets)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets)
            PublicTest_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(valloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
                               % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    avg_loss = PublicTest_loss / (batch_idx + 1)
    print('val_avg_loss: %.4f'%avg_loss)
    PublicTest_acc = 100.*correct/total

    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch
    return avg_loss,PublicTest_acc

def plot_init(viz):
    x,y,z=0,0,0
    win_loss = viz.line(
        X=np.array([x]),
        Y=np.column_stack((np.array([y]),np.array([z]))),
        opts=dict(title='Loss', legend=['train', 'test'], showlegend=True))
    win_acc = viz.line(
        X=np.array([x]),
        Y=np.column_stack((np.array([y]),np.array([z]))),
        opts=dict(title='Acc', legend=['train', 'test'], showlegend=True))
    return win_loss,win_acc

if __name__ == '__main__':

    ##visdom_online
    viz = Visdom(env='RAF_resnet34-arc48')
    win_loss, win_acc = plot_init(viz)

    for epoch in range(start_epoch, total_epoch):
        sheduler.step()
        train_loss,train_acc = train(epoch)
        # val_loss, val_acc =  PublicTest(epoch)
        val_loss, val_acc = RAFTest_48(epoch)
        # test_loss, test_acc = PublicTest_48(epoch)

        # PrivateTest(epoch)
        viz.line(
            X=np.array([epoch]),
            Y=np.column_stack((np.array([train_loss]), np.array([val_loss]))),
            opts=dict(title='Loss', xlable='epoch',ylabel='Loss',legend=['train', 'test'], showlegend=True),
            win=win_loss,  # win要保持一致
            update='append')
        viz.line(
            X=np.array([epoch]),
            Y=np.column_stack((np.array([train_acc]), np.array([val_acc]))),
            opts=dict(title='Acc',xlable='epoch',ylabel='Acc', legend=['train', 'test'], showlegend=True),
            win=win_acc,  # win要保持一致
            update='append')
    print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
    print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
    # print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
    # print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)

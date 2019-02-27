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
import torch.optim.lr_scheduler as lr_scheduler
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from fer import RAF
from torch.autograd import Variable
from models import *
from models.xception import xception
from ArcModel.focal_loss import *
# from models.resnetall import resnet18,resnet34,resnet50,resnet101
from models.imagenetresnet import resnet34
from models.densenet import densenet121,densenet161,densenet169,densenet201
from centerLoss import CenterLoss
from visdom import Visdom
import numpy as np

from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch RAF CNN Training')
parser.add_argument('--model', type=str, default='Resnet34', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='RAF', help='CNN architecture')
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

cut_size = 112
total_epoch = 100

# train_path = 'data/secondTrain/raf_fer_train/'
# train_txt = 'data/secondTrain/raf_fer_train.txt'
# train_path = 'data/firstTrain/aligned48/'
# train_txt = 'data/firstTrain/train.txt'
train_path = 'data/thirdTrain/aligned100/'
train_txt = 'data/thirdTrain/raf_train.txt'


# val_path = 'data/secondTrain/raf_val/'
# val_txt = 'data/secondTrain/raf_val.txt'
# val_path = 'data/firstTrain/aligned48/'
# val_txt = 'data/firstTrain/test.txt'
val_path = 'data/thirdTrain/aligned100/'
val_txt = 'data/thirdTrain/raf_test.txt'

# test_path = 'data/secondTrain/fer_test/'
# test_txt = 'data/secondTrain/fer_test.txt'

path = os.path.join(opt.dataset + '_' + opt.model)

# Data aug
# Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
print('==> Preparing data..')

normMean = [0.485, 0.456, 0.406]
normStd = [0.229, 0.224, 0.225]
normTransform = transforms.Normalize(normMean, normStd)
# transforms.RandomRotation(10),  # 在（-10， 10）范围内旋转
# transforms.ColorJitter()
transform_train = transforms.Compose([
    transforms.Resize(120),
    transforms.RandomResizedCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    # transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # normTransform
])

transform_test = transforms.Compose([
    transforms.Resize(120),
    # transforms.CenterCrop(cut_size),
    # transforms.ToTensor(),
    transforms.FiveCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
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
    # net = resnet18(num_classes=7)
    print('aa')
elif opt.model == 'Resnet34':
    net = resnet34(num_classes=7)
    print(net)
elif opt.model  == 'Resnet50':
    print('aa')
    # net = resnet50(num_classes=7)
    # print(net)
elif opt.model == 'densenet':
    net = densenet121(num_class=7)
elif opt.model == 'xception':
    net = xception(class_nums=7)
    print(net)

# if opt.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir(path), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))
#
#     net.load_state_dict(checkpoint['net'])
#     best_PublicTest_acc = checkpoint['best_PublicTest_acc']
#     best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
#     best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
#     best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
#     start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
# else:
#     print('==> Building model..')

device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    net.cuda()
    # summary(net, input_size=(3, 48, 48))
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(gamma=2)
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
sheduler = lr_scheduler.StepLR(optimizer,20,gamma=0.5)
# optimizer = optim.Adam(net.parameters(), lr=opt.lr)
# optim = optim.Adam(self.model.parameters(), lr=opts.lr)
# optim = optim.RMSprop(self.model.parameters(), lr=opts.lr)
loss_weight = 0.03
centerloss = CenterLoss(num_classes=7,feat_dim=512).to(device)
optimzer_center = optim.SGD(centerloss.parameters(), lr =0.5)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
    #     frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
    #     decay_factor = learning_rate_decay_rate ** frac
    #     current_lr = opt.lr * decay_factor
    #     utils.set_lr(optimizer, current_lr)  # set the decayed rate
    # else:
    #     current_lr = opt.lr
    # print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs,featurelast = net(inputs)
        loss = criterion(outputs, targets) + loss_weight * centerloss(targets,featurelast)

        optimizer.zero_grad()
        optimzer_center.zero_grad()

        loss.backward()
        # utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        for param in centerloss.parameters():
            param.grad.data *= (1. / loss_weight)  # 权重回归
        optimzer_center.step()

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
            outputs,feature = net(inputs)
            loss = criterion(outputs, targets) + loss_weight * centerloss(targets,feature)
            PublicTest_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(valloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
                               % (PublicTest_loss / (batch_idx + 1),  float(correct) / total*100.0, correct, total))

    # Save checkpoint.
    avg_loss = PublicTest_loss / (batch_idx + 1)
    print('val_avg_loss: %.4f'%avg_loss)
    PublicTest_acc = float(correct)/total * 100.0
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
            outputs,featurelast = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets) + loss_weight * centerloss(targets,featurelast)
            PublicTest_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(valloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1),  float(correct) / total *100.0, correct, total))

    # Save checkpoint.
    avg_loss = PublicTest_loss / (batch_idx + 1)
    print('val_avg_loss: %.4f'%avg_loss)
    PublicTest_acc = float(correct)/total * 100.0

    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.4f" % PublicTest_acc)
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

# def plot_online():
#     viz = Visdom(env='Expression')
#     x,y,z=0,0,10
#     win = viz.line(
#         X=np.array([x]),
#         Y=np.column_stack((np.array([y]),np.array([z]))),
#         opts=dict(title='two_lines', legend=['train', 'loss'], showlegend=True))
#     for i in range(100):
#         x+=i
#         y+=i
#         z+=i*10
#         viz.line(
#             X=np.array([x]),
#             Y=np.column_stack((np.array([y]),np.array([z]))),
#             opts=dict(title='two_lines', legend=['train','loss'],showlegend=True),
#             win=win,#win要保持一致
#             update='append')
def plot_init(viz):
    x,y,z=0,0,0
    win_loss = viz.line(
        X=np.array([x]),
        Y=np.column_stack((np.array([y]),np.array([z]))),
        opts=dict(title='Loss', xlable='epoch',ylabel='Loss',legend=['train', 'test'], showlegend=True))
    win_acc = viz.line(
        X=np.array([x]),
        Y=np.column_stack((np.array([y]),np.array([z]))),
        opts=dict(title='Acc', xlable='epoch',ylabel='Loss',legend=['train', 'test'], showlegend=True))
    return win_loss,win_acc

if __name__ == '__main__':

    ##visdom_online
    viz = Visdom(env='RAF_modifyresnet34-112')
    win_loss, win_acc = plot_init(viz)

    for epoch in range(start_epoch, total_epoch):
        sheduler.step()
        print('learning_rate: %s' % str(sheduler.get_lr()))
        train_loss,train_acc = train(epoch)
        val_loss, val_acc =  PublicTest(epoch)
        # val_loss, val_acc = RAFTest_48(epoch)
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

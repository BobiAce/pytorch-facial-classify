"""
plot confusion_matrix of PublicTest and PrivateTest
"""

import itertools
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from fer import FER2013
from fer import RAF

from torch.autograd import Variable
import torchvision
import transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from models import *
from torchsummary import summary

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG11', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--split', type=str, default='PrivateTest', help='split')
opt = parser.parse_args()

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

# class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
RAF_class_names = ['Surprise','Fear','Disgust', 'Happy', 'Sad','Angry', 'Neutral']
# Model
if opt.model == 'VGG11':
    net = VGG(opt.model)
    print(net)
elif opt.model  == 'Resnet18':
    net = ResNet18()

# path = os.path.join(opt.dataset + '_' + opt.model)
path = 'RAF_VGG11/'#   RAF_VGG16_1128_first
modelpath = os.path.join(path,'PublicTest_model.t7')
checkpoint = torch.load(modelpath)#,map_location='cpu'

net.load_state_dict(checkpoint['net'])
net.cuda()
summary(net, input_size=(3, 44, 44))
net.eval()
# Testset = FER2013(split = opt.split, transform=transform_test)
# Testloader = torch.utils.data.DataLoader(Testset, batch_size=128, shuffle=False, num_workers=1)
test_path = 'data/firstTrain/aligned48/'
test_txt = 'data/firstTrain/test.txt'
testset = RAF(test_path,test_txt, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)

correct = 0
total = 0
all_target = []
for batch_idx, (inputs, targets) in enumerate(testloader):

    bs, ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
    outputs = net(inputs)

    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
    _, predicted = torch.max(outputs_avg.data, 1)

    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()
    if batch_idx == 0:
        all_predicted = predicted
        all_targets = targets
    else:
        all_predicted = torch.cat((all_predicted, predicted),0)
        all_targets = torch.cat((all_targets, targets),0)

acc = 100. * correct / total
print("accuracy: %0.3f" % acc)
y_true = all_targets.data.cpu().numpy()
y_pred = all_predicted.cpu().numpy()
macro = f1_score(y_true, y_pred, average=None)
print('F1_macro = %s'%macro)
# Compute confusion matrix
matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=RAF_class_names, normalize=True,
                      title= 'Confusion Matrix (Accuracy: %0.3f%%)' %acc)
# plt.savefig(os.path.join(path, 'test_cm.png'))
plt.close()

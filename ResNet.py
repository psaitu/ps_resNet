import os
import numpy as np
from PIL import Image
import time
from shutil import copyfile
from os.path import isfile, join, abspath, exists, isdir, expanduser
from os import listdir, makedirs, getcwd, remove
from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler
import pandas as pd
import sys

batch_size = 64
num_workers = 8

mean = [0.485, 0.456, 0.406]
stdev = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=mean, std=stdev)

image_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

train_dataset = torchvision.datasets.ImageFolder(root='hw2p2_check/train_data/medium//', transform=image_transformations)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

dev_dataset = torchvision.datasets.ImageFolder(root='hw2p2_check/validation_classification/medium//', transform=image_transformations)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

dataset_sizes = {
    'train': len(train_dataloader.dataset), 
    'valid': len(dev_dataloader.dataset)
}

def resnet_block(n_channels, kernal_size=3, stride=1, padding=1):
    resblock = nn.Sequential(
        nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernal_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(num_features=n_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernal_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(num_features=n_channels))
    return resblock


class ResBlock(nn.Module):
    def __init__(self, n_channels, stride=1):
        super(ResBlock, self).__init__()
        
        kernal_size = 3
        padding = 1
        
        self.resnet = resnet_block(n_channels, kernal_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = x
        y = self.resnet(y)
        y = self.relu(y + x)
        return y

def conv2d(in_channel, out_channel, kernel_size=3, stride=2):
    conv2d = nn.Conv2d(
        in_channels = in_channel,
        out_channels = out_channel,
        kernel_size = kernel_size,
        stride = stride,
        bias=False)
    return conv2d


class Network(nn.Module):
    def __init__(self, features, hiddens, classes, dimensions=10):
        super(Network, self).__init__()
        
        self.hiddens = [features] + hiddens + [classes]
        self.layers = []
        
        for i, channels in enumerate(hiddens):
            self.layers.append(conv2d(in_channel=self.hiddens[i], out_channel=self.hiddens[i+1], kernel_size=3, stride=2))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(ResBlock(n_channels=channels))
            
        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(self.hiddens[-2], self.hiddens[-1], bias=False)
    
    def forward(self, x, evalMode=False):
        y = x
        y = self.layers(y)
            
        y = F.avg_pool2d(y, [y.size(2), y.size(3)], stride=1)
        y = y.reshape(y.shape[0], y.shape[1])
        
        label_output = self.linear_label(y)
        label_output = label_output / torch.norm(self.linear_label.weight, dim=1)

        return label_output

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=10):
    
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            running_batch = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                labels = labels.view(-1)
                
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                running_batch +=1

            epoch_loss = running_loss / running_batch
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            torch.cuda.empty_cache()
            del inputs
            del labels
            del loss

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model

#hyper params
numEpochs = 4
learningRate = 0.001
weightDecay = 5e-5
# ResNet
num_feats  = 3
use_gpu = torch.cuda.is_available()
hidden_sizes = [128, 256, 512, 1024]
num_classes = len(train_dataset.classes)

model = Network(num_feats, hidden_sizes, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
if use_gpu:
    model = model.cuda()

loaders = {'train':train_dataloader, 'valid':dev_dataloader}
model = train_model(loaders, model, criterion, optimizer, exp_lr_scheduler, num_epochs=15)

torch.save(model.state_dict(), "res_68.pth")

class OrderedImageDataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = self.transform(img)
        label = 0
        return img, label

test_images = []
for i in range(num_classes):
    test_images.append('hw2p2_check/test_classification/medium/' + str(i)+".jpg")
test_dataset = ImageDataset(test_images, transform=image_transformations)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

classifications = []
classifications_flattened = []
for batch_num, (inputs, labels) in enumerate(verify_dataloaders[i]):
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        classifications.append(preds.cpu().detach().numpy())

for sublist in classifications:
    for item in sublist:
        classifications_flattened.append(train_dataset.classes[item])


df = pd.DataFrame({'id': np.arange(0,4600),
                   'label': classifications_flattened})
df.to_csv('r68.csv', index=False)

test_trials = pd.read_csv("hw2p2_check/test_trials_verification_student.txt", header=None)
test_trials = test_trials.values.tolist()

images_list_1 = []
images_list_2 = []
for i in range(len(test_trials)):
    images_list_1.append("hw2p2_check/test_verification/" + test_trials[i][0].split()[0])
    images_list_2.append("hw2p2_check/test_verification/" + test_trials[i][0].split()[1])
    
verify_1 = OrderedImageDataset(images_list_1, transform=image_transformations)
verify_1_dataloader = torch.utils.data.DataLoader(verify_1, batch_size=batch_size, shuffle=False, num_workers=num_workers)

verify_2 = OrderedImageDataset(images_list_2, transform=image_transformations)
verify_2_dataloader = torch.utils.data.DataLoader(verify_2, batch_size=batch_size, shuffle=False, num_workers=num_workers)

flat_list1 = []
flat_list2 = []

verify_dataloaders = [verify_1_dataloader, verify_2_dataloader]

verify_outputs = {
    'outputs_0': [], 
    'outputs_1': []
}

verify_outputs_flattened = {
    'outputs_0': [], 
    'outputs_1': []
}

for i in range(len(verify_dataloaders)):
    for batch_num, (inputs, labels) in enumerate(verify_dataloaders[i]):
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        outputs = model(inputs)
        verify_outputs['outputs_' + str(i)].append(outputs.cpu().detach().numpy())
    
    for sublist in verify_outputs['outputs_' + str(i)]:
        for item in sublist:
            verify_outputs_flattened['outputs_' + str(i)].append(item)

score = torch.nn.functional.cosine_similarity(
    torch.FloatTensor(verify_outputs_flattened['outputs_0']),
    torch.FloatTensor(verify_outputs_flattened['outputs_1'])
)

f = open("hw2p2_verify.csv", "w")
f.write("trial,score\n")
for i in range(len(score)):
    f.write(test_trials[i][0] + "," + str(score.cpu().detach().numpy()[i])+"\n")
f.close()
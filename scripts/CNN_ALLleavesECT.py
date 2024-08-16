#!/usr/bin/env python
# coding: utf-8



import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as img
from PIL import Image
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn, optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

from tqdm import tqdm



import os
os.getcwd()





data_path = '../data/ALLleaves_ECT'

# # CNN
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001 #was 0.001
KERNEL_SIZE = 3

pad_size = math.floor(KERNEL_SIZE/2)



# Dict to map class names to indices
classes = []
for (dirpath, dirnames, filenames) in os.walk(data_path):
    classes.extend(dirnames)
    break
idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
num_classes = len(classes)
print('num_classes=',num_classes)





class NPYDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        x = x.unsqueeze(0)
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)



    
# load in images to dataset
numpy_data=[]
numpy_target=[]

for path, subdirs, files in os.walk(data_path):
    
    files = [f for f in files if not f[0] == '.']
    subdirs[:] = [d for d in subdirs if not d[0] == '.']
    
    for name in files:
        input_filedir = os.path.join(path, name)
        image = np.load(input_filedir)
        numpy_data.append(image)
        
        splitpath = os.path.normpath(input_filedir).split(os.path.sep)
        label = list(set(splitpath).intersection(classes))[0]
        numpy_target.append(class_to_idx[label])
        
numpy_data = np.array(numpy_data)
numpy_target = np.array(numpy_target)



# Training and validation loaders

train, valid_data, y_train, y_valid = train_test_split(numpy_data, numpy_target, test_size=0.2)
train_data = NPYDataset(train, y_train)
train_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=0)
valid_data = NPYDataset(valid_data, y_valid)
valid_loader = DataLoader(dataset = valid_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=0)



def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image[0,:,:]
    # FIX NORMALIZATION PARAMS (mean and std)
    if normalize:
        mean = [5]
        std = [0.25]
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image,cmap='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


""" trainimages, trainlabels = next(iter(train_loader))

fig, axes = plt.subplots(figsize=(12, 12), ncols=5)
print('training images')
for i in range(5):
    axe1 = axes[i] 
    imshow(trainimages[i], ax=axe1, normalize=False)
    plt.show() """





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.padC = [pad_size,pad_size,0,0]
        self.padZ = [0,0,pad_size,pad_size]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=KERNEL_SIZE)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=KERNEL_SIZE)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1540, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.pad(F.pad(input=x, pad=self.padC, mode='circular'), pad=self.padZ, mode='constant')
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x






model = CNN().to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)
expr_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3, 7], gamma=0.1)







loss_list = []
accuracy_list = []
iteration_list = []

def train(epoch):
    
    i = 0
    for features, labels in tqdm(train_loader):
        features, labels = Variable(features), Variable(labels)

        # zero out gradients from previous iteration
        optimizer.zero_grad()

        # forward propagation
        output = model(features)
        
        # calculate loss
        loss = criterion(output, labels)

        # backprop
        loss.backward()
        
        # update params (gradient descent)
        optimizer.step()
            
        i += 1

    expr_lr_scheduler.step()



def evaluate(data_loader, epoch=0):
    model.eval()
    loss = 0
    correct = 0
    
    for features, labels in tqdm(data_loader):
        with torch.no_grad():
            features, labels = Variable(features), Variable(labels)
            output = model(features)
            
        loss += F.cross_entropy(output, labels, size_average=False).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        
    loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    print('Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        epoch,
        loss, correct, len(data_loader.dataset),
        accuracy))
    loss_list.append(loss)
    accuracy_list.append(accuracy)
    iteration_list.append(epoch)



evaluate(train_loader)
for epoch in range(1,NUM_EPOCHS+1):
    train(epoch)
    evaluate(valid_loader, epoch=epoch)




    # visualize loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.title("Loss vs Number of epochs")
plt.show()

# visualize accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of epochs")
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sn


y_pred = []
y_true = []

# iterate over test data
for inputs, labels in valid_loader:
        output = model(inputs) 

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)

df_counts = pd.DataFrame(cf_matrix[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_counts, annot=True)
plt.savefig('cf_counts.png')

df_norm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_norm, annot=True)
plt.savefig('cf_norm.png')
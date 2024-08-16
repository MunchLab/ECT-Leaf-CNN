#!/usr/bin/env python
# coding: utf-8



import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as img
from PIL import Image

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



labels = pd.read_csv('data_ECT/labels.csv')

data_path = 'data_ECT/32dirs_48thresh_NPY/'





#plt.figure(figsize = (8,8))
#plt.pie(labels.groupby('species').size(), labels = labels["species"].unique(), autopct='%1.1f%%', shadow=False, startangle=90)
#plt.show()






# Dict to map class names to indices
classes = labels["species"].unique()
idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
num_classes = len(classes)
print('num_classes=',num_classes)


#ADD  SECT TRANSFORM


mean = [-2.377]
std = [3.71]
#normalize images: x = (x-mean)/std
train_transform = transforms.Compose([transforms.Normalize(mean,std)])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean,std)])

valid_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean,std)])

#alternative transforms for data augmentation
train_aug_transforms = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(224),
                                transforms.ColorJitter(),
                                transforms.RandomCrop(24),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((32,32)),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean,std)])


class LeafECTDataset(Dataset):
    def __init__(self, data, path , transform = None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        label = self.data[index][2]
        label = class_to_idx[label]
        
        filename = 'leaf_'+str(index)+'.png'
        img_path = os.path.join(data_path,filename)
        #image = img.imread(img_path)
        image = Image.open(img_path)
        image = np.array(image)
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# second versions of dataset class (for npy input data)
class NPYDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index].unsqueeze(dim=0)
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
# load in images to dataset
numpy_data = []
numpy_target = []
for index in range(len(labels)):
    filename = 'leaf_'+str(index)+'.npy'
    img_path = os.path.join(data_path,filename)
    image = np.load(img_path)
    numpy_data.append(image)
    
    index = int(filename[5:-4])
    label = labels['species'][index]
    numpy_target.append(class_to_idx[label])
    
numpy_data = np.array(numpy_data)
numpy_target = np.array(numpy_target)



# Training and validation loaders
#train, valid_data = train_test_split(labels, stratify=labels.species, test_size=0.2)
#train_data = LeafECTDataset(train, data_path, transform = train_transform)
#train_loader = DataLoader(dataset = train_data, batch_size = 10, shuffle=True, num_workers=0)

#valid_data = LeafECTDataset(valid_data, data_path, transform = train_transform)
#valid_loader = DataLoader(dataset = valid_data, batch_size = 10, shuffle=True, num_workers=0)

train, valid_data, y_train, y_valid = train_test_split(numpy_data, numpy_target, test_size=0.2)
train_data = NPYDataset(train, y_train, transform = train_transform)
train_loader = DataLoader(dataset = train_data, batch_size = 10, shuffle=True, num_workers=0)
valid_data = NPYDataset(valid_data, y_valid, transform = train_transform)
valid_loader = DataLoader(dataset = valid_data, batch_size = 10, shuffle=True, num_workers=0)



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


trainimages, trainlabels = next(iter(train_loader))

fig, axes = plt.subplots(figsize=(12, 12), ncols=5)
print('training images')
for i in range(5):
    axe1 = axes[i] 
    imshow(trainimages[i], ax=axe1, normalize=False)
    plt.show()

    print(trainimages[i].size())
    print(torch.min(trainimages[i]),torch.max(trainimages[i]), )







# # CNN
num_epochs = 40
batch_size = 10 # check this matches dataloader
learning_rate = 0.001 #was 0.001




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1200, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding =1)
        self.conv_drop = nn.Dropout2d(p=0.1)
        # linear layers
        self.fc1 = nn.Linear(1536, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes) 
        # dropout
        self.dropout = nn.Dropout(p=0.2)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv_drop(self.conv2(x))))
        # flattening the image
        x = x.view(x.shape[0],-1)
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x







model = CNN2().to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
expr_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3, 7], gamma=0.1)




""" # keeping-track-of-losses 
train_losses = []
valid_losses = []

for epoch in range(1, num_epochs + 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    valid_loss = 0.0
    
    # training-the-model
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU 
        data = data.to(device)
        target = target.to(device)
        
        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)
        
    # validate-the-model
    model.eval()
    for data, target in valid_loader:
        
        data = data.to(device)
        target = target.to(device)
        
        output = model(data)
        
        loss = criterion(output, target)
        
        # update-average-validation-loss 
        valid_loss += loss.item() * data.size(0)
    
    # calculate-average-losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
        
    # print-training/validation-statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))


# test-the-model
model.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
          
    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

# Save 
torch.save(model.state_dict(), 'model.ckpt')
 """




loss_list = []
accuracy_list = []
iteration_list = []

def train(epoch):
    expr_lr_scheduler.step()
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





def evaluate(data_loader):
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



n_epochs = 50

for epoch in range(n_epochs):
    train(epoch)
    evaluate(valid_loader)




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
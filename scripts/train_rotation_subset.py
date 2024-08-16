#Rotation Evaluation

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models import CNN, CNN_images, CNN_simple
from dataloaders import create_rotation_datasets, create_datasets
from torch.utils.data import DataLoader
from utils import save_model, save_plots, save_cf, SaveBestModel


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=40,
    help='number of epochs to train the network for')
parser.add_argument('-b', '--batch_size', type=int, default=32,
    help='batch_size for dataloaders')
parser.add_argument('-lr', '--lr', type=float, default= 1e-3,
    help='learning rate for training')
parser.add_argument('-t', '--type', type=str, default= 'ect',
    help='which input type to use for model: ect or image')

args = vars(parser.parse_args())

NUM_EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']
LEARNING_RATE = args['lr']


# device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# get the training, validation and test_datasets
# image_type specifies if model should use ect data, sect transform of ect data, or original images
train_dataset= create_rotation_datasets(image_type = args['type'], sub='train')
test_dataset= create_rotation_datasets(image_type = args['type'], sub='test0')
#train_dataset, test_dataset = create_datasets(image_type = args['type'])

# get the training and validaion data loaders
train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle=True)    
trainimages, trainlabels = next(iter(train_loader))

print(trainlabels)
print(trainimages.shape)
plt.style.use('default')
fig, axes = plt.subplots(figsize=(10,5), ncols=int(BATCH_SIZE/4))
print('training images')
for i in range(int(BATCH_SIZE/4)):
    ax = axes[i]
    ax.imshow(trainimages[i,0,:,:], cmap='gray')
    ax.set_title(trainlabels[i])
plt.show()

print('dataset says classes=',train_dataset.num_classes)
# model
if args['type']=='image':
    model = CNN_images(num_classes=train_dataset.num_classes).to(device)
else:
    model = CNN(num_classes=train_dataset.num_classes, num_channels=trainimages.shape[1]).to(device)
    #model = CNN_simple(num_classes=train_dataset.num_classes, num_channels=trainimages.shape[1]).to(device)
print(model)

# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")

# optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# loss function
lossfcn = nn.CrossEntropyLoss()
# initialize SaveBestModel class
save_best_model = SaveBestModel()


# function to train the  model
def train(model, train_loader, optimizer, lossfcn, output):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    outputs_list=[]
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        outputs_list.append(outputs.detach().cpu())
        # calculate the loss
        loss = lossfcn(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
        #print(outputs_list)
    #outputs_list = torch.cat(outputs_list)   
    #output.append(outputs_list)
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc, outputs_list


# function for validation
def test(model, valid_loader, lossfcn, output):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        outputs_list = []
        for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            outputs_list.append(outputs)
            # calculate the loss
            loss = lossfcn(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        #outputs_list = torch.cat(outputs_list)   
    #output.append(outputs_list) 
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))

    return epoch_loss, epoch_acc, output

# keeping track of losses and accuracies
train_loss, test_loss = [],[]
train_acc, test_acc = [],[]
train_outputs, test_outputs = [],[]

# loss/accuracy before training
train_loss_e0, train_acc_e0,_ = test(model,train_loader, lossfcn, train_outputs)
test_loss_e0, test_acc_e0,_ = test(model, test_loader, lossfcn, test_outputs)
train_loss.append(train_loss_e0)
test_loss.append(test_loss_e0)
train_acc.append(train_acc_e0)
test_acc.append(test_acc_e0)
print(f"Training loss: {train_loss_e0:.3f}, training acc: {train_acc_e0:.3f}")
print(f"Test loss: {test_loss_e0:.3f}, test acc: {test_acc_e0:.3f}")


# begin training
for epoch in range(1,NUM_EPOCHS+1):
    print(f"[INFO]: Epoch {epoch} of {NUM_EPOCHS}")
    train_epoch_loss, train_epoch_acc, outputs_TRAIN = train(model, train_loader, optimizer, lossfcn, train_outputs)
    test_epoch_loss, test_epoch_acc, outputs_TEST = test(model, test_loader, lossfcn, test_outputs)
    train_loss.append(train_epoch_loss)
    test_loss.append(test_epoch_loss)
    train_acc.append(train_epoch_acc)
    test_acc.append(test_epoch_acc)
    #outputs_TRAIN = torch.stack(outputs_TRAIN)
    #outputs_TEST = torch.stack(outputs_TEST)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Test loss: {test_epoch_loss:.3f}, test acc: {test_epoch_acc:.3f}")

    # save the best model up to current epoch, if we have the least loss in the current epoch
    save_best_model(
        test_epoch_loss, epoch, model, optimizer, lossfcn
    )
    print('-'*50)
#output_test = torch.stack(test_outputs)
#output_train = torch.stack(train_outputs)
#torch.save(output_train, f'results/{args["type"]}_train.pth')
#torch.save(output_test, f'results/{args["type"]}_test.pth')
  
# save the trained model weights for a final time
save_model(NUM_EPOCHS, model, optimizer, lossfcn)
# save the loss and accuracy plots
save_plots(train_acc, test_acc, train_loss, test_loss)



print('TRAINING COMPLETE.')


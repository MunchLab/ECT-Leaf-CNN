import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models import CNN, CNN_images, CNN_simple
from dataloaders import create_datasets, create_data_loaders, create_rotation_datasets
from utils import save_model, save_plots, save_cf, SaveBestModel

device = ('cuda' if torch.cuda.is_available() else 'cpu')
#model = CNN_simple(num_classes=6, num_channels=1)
model = CNN(num_classes=14, num_channels=1)

#train_dataset, valid_dataset = create_datasets(image_type = 'ect')
#train_dataset, valid_dataset = create_datasets(image_type = 'image')

#train_dataset = create_rotation_datasets(image_type = 'ect', sub='train')
#valid_dataset = create_rotation_datasets(image_type = 'ect', sub='test1')
train_dataset, valid_dataset = create_datasets(image_type = 'ect')

train_loader, valid_loader = create_data_loaders(train_dataset, valid_dataset)


state_dict = torch.load('outputs/best_model.pth')['model_state_dict']
model.load_state_dict(state_dict)
model.eval()
print('Using validation to compute confusion matrix')
valid_running_pred = []
valid_running_labels = []
counter = 0
with torch.no_grad():
    for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        counter += 1
        
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(image)
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)

        valid_running_pred.append(preds)
        valid_running_labels.append(labels)
    
# confusion matrix for the complete epoch
valid_running_pred = torch.cat(valid_running_pred)
valid_running_labels = torch.cat(valid_running_labels)

save_cf(valid_running_pred,valid_running_labels, valid_dataset.classes)



print('TESTING COMPLETE.')
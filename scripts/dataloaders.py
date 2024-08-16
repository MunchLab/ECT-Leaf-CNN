import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
os.getcwd()
import random
from utils import SECT, ROTATE_TRANSLATE, STACKED_TRANSFORMS

BATCH_SIZE = 32
VALID_SPLIT = 0.2
NUM_WORKERS = 0


""" data_path = '../data/ALLleaves_ECT'
# Dict to map class names to indices
classes = []
for (dirpath, dirnames, filenames) in os.walk(data_path):
    classes.extend(dirnames)
    break
idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
num_classes = len(classes)
print('num_classes===',num_classes) """




#Dataset class for numpy array data (ECT, SECT)
class NPYDataset(Dataset):
    def __init__(self, data, target, classes, transform=None):
        #self.data = torch.from_numpy(data).float()
        self.data = data
        self.target = torch.from_numpy(target).long()
        #self.target = torch.from_numpy(target).float()
        self.transform = transform
        self.num_classes = len(classes)
        self.classes = classes
        
    def __getitem__(self, index):
        
        x = self.data[index]
        #x = x.unsqueeze(0)
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        #x = x.unsqueeze(0)
        return x, y
    
    def __len__(self):
        return len(self.data)
    
class ImageDataset(Dataset):
    def __init__(self, data, target, classes, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        self.num_classes = len(classes)
        self.classes = classes
        
    def __getitem__(self, index):
        x = self.data[index]
        x = x.unsqueeze(0)
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

# Transforms included in utils.py
def rescale(arr):
    new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    return new_arr
    
# function to create the datasets
def create_datasets(dataset, image_type='ect'):
    """
    Function to load in images (ect or outline) to dataset
    """
    print('DATASET:', dataset)
    print('IMAGE TYPE:', image_type)
    if dataset=='leafoutline':
        if image_type=='image':
            #image_path = '../data/ALLleaves_images_small'
            image_path = '../data/ALLleaves_images'
        else:
            image_path ='../data/ALLleaves_ECT'
    elif dataset=='leafgraph':
        if image_type=='image':
            image_path = '../data/leafgraph_images_small'
        else:
            image_path ='../data/leafgraph_ECT'
    elif dataset=='mpeg7': 
        if image_type=='image':
            image_path = '../data/MPEG7shap_images_small'
        else:
            image_path ='../data/MPEG7shap_ECT'
    else:
        raise Exception(f"The dataset name {dataset=} passed is not valid.")

    print(image_path)
    # Dict to map class names to indices
    classes = []
    for (dirpath, dirnames, filenames) in os.walk(image_path):
        print(dirnames)
        classes.extend(dirnames)
        break
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    num_classes = len(classes)
    print('num_classes=',num_classes)


    numpy_data=[]
    numpy_target=[]
    classes_count=[0 for c in classes]
    for path, subdirs, files in os.walk(image_path):
        
        files = [f for f in files if not f[0] == '.']
        subdirs[:] = [d for d in subdirs if not d[0] == '.']
        
        for name in files:
            input_filedir = os.path.join(path, name)

            if image_type=='image':
                image = rescale(np.array(Image.open(input_filedir).convert('L')))#Converting to grayscale, using only 1 channel
            elif image_type=='ect' or 'sect':
                image = rescale(np.load(input_filedir))

            splitpath = os.path.normpath(input_filedir).split(os.path.sep)
            label = list(set(splitpath).intersection(classes))[0]
            
            #if classes_count[class_to_idx[label]]< 50:
            if True:
                numpy_data.append(image)
                numpy_target.append(class_to_idx[label])
                classes_count[class_to_idx[label]]+=1
                
            
    numpy_data = np.array(numpy_data)
    numpy_target = np.array(numpy_target)
   
    train, valid_data, y_train, y_valid = train_test_split(numpy_data, numpy_target, test_size=VALID_SPLIT, shuffle=True)


    if image_type=='sect':
        print('applying Smooth ECT transform')
        train_data = NPYDataset(train, y_train, classes, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE(), SECT()]))
        valid_data = NPYDataset(valid_data, y_valid, classes, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE(), SECT()]))
    elif image_type=='image':
        print('Using normalize transform on image data')
        #train_data = ImageDataset(train, y_train, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5)]))
        #valid_data = ImageDataset(valid_data, y_valid, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5)]))
        if dataset=='mpeg7':
            fill = -1
        else:
            fill = 1
        train_data = NPYDataset(train, y_train, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5),transforms.RandomRotation((0,360), fill=fill)]))
        valid_data = NPYDataset(valid_data, y_valid, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), transforms.RandomRotation((0,360), fill=fill)]))
    else:
        print('ECT data; using only normalize, rotation transforms on training data')
        train_data = NPYDataset(train, y_train, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE()]))   
        valid_data = NPYDataset(valid_data, y_valid, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE()]))
    return train_data, valid_data




# function to create the datasets
def create_stacked_datasets(dataset):
    """
    Function to load in images (ect and outline) to dataset
    """
    print('DATASET:', dataset)

    if dataset=='leafoutline':
        image_path = '../data/ALLleaves_images_small'
        ect_path ='../data/ALLleaves_ECT'
    elif dataset=='leafgraph':
        image_path = '../data/leafgraph_images_small'
        ect_path ='../data/leafgraph_ECT'
    elif dataset=='mpeg7': 
        image_path = '../data/MPEG7shap_images_small'
        ect_path ='../data/MPEG7_ECT/subset'
    else:
        raise Exception(f"The dataset name {dataset=} passed is not valid.")

    print('imagepath:',image_path)
    print('ectpath:',ect_path)

    # Dict to map class names to indices
    classes = []
    for (dirpath, dirnames, filenames) in os.walk(image_path):
        print(dirnames)
        classes.extend(dirnames)
        break
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    num_classes = len(classes)
    print('num_classes=',num_classes)


    numpy_data=[]
    numpy_target=[]
    classes_count=[0 for c in classes]
    for path, subdirs, files in os.walk(image_path):
        
        files = [f for f in files if not f[0] == '.']
        subdirs[:] = [d for d in subdirs if not d[0] == '.']
        
        for name in files:
            input_filedir1 = os.path.join(path, name)

            image = rescale(np.array(Image.open(input_filedir1).convert('L')))#Converting to grayscale, using only 1 channel

            end = os.path.join(path[31:], name[:-3]+'npy')
            input_filedir2 = os.path.join(path[:18]+'ECT', end)
            ect = rescale(np.load(input_filedir2)) #load the ECT image (single channel)


            splitpath = os.path.normpath(input_filedir1).split(os.path.sep)
            label = list(set(splitpath).intersection(classes))[0]
            
            #if classes_count[class_to_idx[label]]< 50:
            if True:
                image_stack = np.stack((image, ect,ect),axis=0) 
                numpy_data.append(image_stack)
                numpy_target.append(class_to_idx[label])
                classes_count[class_to_idx[label]]+=1
                
            
    numpy_data = np.array(numpy_data)
    numpy_target = np.array(numpy_target)

    
   
    train, valid_data, y_train, y_valid = train_test_split(numpy_data, numpy_target, test_size=VALID_SPLIT, shuffle=True)

    train_data = NPYDataset(train, y_train, classes, transform = transforms.Compose([STACKED_TRANSFORMS()]))   
    valid_data = NPYDataset(valid_data, y_valid, classes, transform = transforms.Compose([STACKED_TRANSFORMS()]))
    
    
    return train_data, valid_data



# function to create the datasets
def create_rotation_datasets(image_type = 'ect',sub='train'):
    """
    Function to load in images to dataset- for the subset dataset of rotations
    """
    if image_type =='ect':
        pathbase = '../data/ALLleaves_ECT_rotate/50eachclass/ect32.48/'
        
    else:
        pathbase = '../data/ALLleaves_ECT_rotate/50eachclass/outline32.48/'
    if sub=='train':
        image_path = pathbase + 'train'
    else:
        image_path = pathbase+str(sub)

    print(image_path)
    # Dict to map class names to indices
    classes = []
    for (dirpath, dirnames, filenames) in os.walk(image_path):
        classes.extend(dirnames)
        break
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    num_classes = len(classes)
    print('num_classes=',num_classes)
    print('classes:', classes)

    numpy_data=[]
    numpy_target=[]
    for path, subdirs, files in os.walk(image_path):
        
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
   
    #numpy_data, numpy_target = shuffle(numpy_data, numpy_target)
    X_tr, X_te, y_tr, y_te = train_test_split(numpy_data, numpy_target, test_size=.5)
    numpy_data = np.concatenate((X_tr,X_te), axis=0)
    numpy_target = np.concatenate((y_tr,y_te), axis=0)

    transform_list = transforms.Compose([transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE()])
    if sub=='train':
        #data = NPYDataset(numpy_data, numpy_target, classes, transform = transform_list)
        data = NPYDataset(numpy_data, numpy_target, classes, transform = transforms.Normalize(0.5,0.5))
    else:
        data = NPYDataset(numpy_data, numpy_target, classes, transform = transforms.Normalize(0.5,0.5))


    return data



def create_data_loaders(train_data, valid_data):
    """
    Function to build the data loaders.
    Parameters:
    :param train_data: The training dataset.
    :param valid_data: The validation dataset.
    """

    train_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    valid_loader = DataLoader(dataset = valid_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)    

    return train_loader, valid_loader


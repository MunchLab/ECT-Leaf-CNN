import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import find_numpy_files
import os
os.getcwd()
from utils import ROTATE_TRANSLATE

BATCH_SIZE = 4
VALID_SPLIT = 0.2
NUM_WORKERS = 0





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
    
    def get_label(self, index):
        if type(index) == torch.Tensor:
            index = index.numpy()
        return self.classes[index]
    
    def __len__(self):
        return len(self.data)
    


# Transforms included in utils.py
def rescale(arr):
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return arr
    return  ( (arr - min_val) / ( (max_val - min_val) * 255) ).astype('uint8')
    
# function to create the datasets
def create_datasets(data=os.getcwd()+'/example_data/ect_output/', valid_split=VALID_SPLIT, log_level='INFO'):
    """
    Function to load in images (ect or outline) to dataset
    """
    numpy_data=[]
    numpy_target=[]

    log_level = log_level == True or str(log_level).upper() == 'INFO'
    
    if type(data) == dict:
        classes = list(data.keys())
        for i, category in enumerate(data):
            for img in data[category]:
                numpy_data.append( img )
                numpy_target.append( i )
        numpy_data = np.float32(numpy_data)
        numpy_target = np.float32(numpy_target)
        train, valid_data, y_train, y_valid = train_test_split(numpy_data, numpy_target, test_size=valid_split, shuffle=True)
        if log_level:
            print('ECT data; using only normalize, rotation transforms on training data')
        train_data = NPYDataset(train, y_train, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE()]))   
        valid_data = NPYDataset(valid_data, y_valid, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE()]))
        return train_data, valid_data
    else:
        image_path = data

    # image_path = os.getcwd()+'/example_data/ect_output/'
    
    # Dict to map class names to indices
    classes = [ dir_name for dir_name in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, dir_name)) ]
    class_to_idx = {j:i for i, j in enumerate(classes)}
    num_classes = len(classes)
    if log_level:
        print('num_classes=',num_classes)

    np_files = { class_name: find_numpy_files( os.path.join(image_path, class_name) ) for class_name in classes }
    np_data = [ (rescale(np.load(np_file)), class_name) for class_name, file_list in np_files.items() for np_file in file_list ]
    

    numpy_data = np.float32( [i[0] for i in np_data] )
    numpy_target = np.float32( [class_to_idx[i[1]] for i in np_data] )
   
    train, valid_data, y_train, y_valid = train_test_split(numpy_data, numpy_target, test_size=valid_split, shuffle=True)

    if log_level:
        print('ECT data; using only normalize, rotation transforms on training data')
    train_data = NPYDataset(train, y_train, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE()]))   
    valid_data = NPYDataset(valid_data, y_valid, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE()]))

    return train_data, valid_data







def create_data_loaders(train_data, valid_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Function to build the data loaders.
    Parameters:
    :param train_data: The training dataset.
    :param valid_data: The validation dataset.
    """

    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=num_workers)

    valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=True, num_workers=num_workers)    

    return train_loader, valid_loader


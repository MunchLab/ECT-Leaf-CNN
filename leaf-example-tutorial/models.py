import torch
import torch.nn as nn
import torch.nn.functional as F

#KERNEL_SIZE = 3
KERNEL_SIZE = 3

# For now I was just working on making the dynamic ECT resolution work.
# I can go in and fix all the other classes if required.
# Also, we should probably make the input_resolution a required argument.

class CNN(nn.Module): 
    def __init__(self, num_classes, num_channels, **kwargs):
        '''
        A CNN model for ECT data.
        The model is designed to take in a 4D tensor of shape (batch_size, num_channels, height, width) and output a 2D tensor of shape (batch_size, num_classes).
        The model consists of two convolutional layers followed by two fully connected layers.
        The convolutional layers are followed by max pooling layers and ReLU activation functions.
        The fully connected layers are followed by ReLU activation functions.
        The first convolutional layer has 10 output channels and the second has 20 output channels.
        The first fully connected layer has 1024 output features.

        Required arguments:
            num_classes: int, the number of classes to predict.
            num_channels: int, the number of channels in the input data.
        Optional keyword arguments:
            kernel_size: int, the size of the convolutional kernels. Default is 3.
            pad_size: int, the size of the padding to apply to the input data. Default is half the kernel size.
            input_resolution: tuple of ints, the resolution of the input data. Default is (32, 48).
        '''
        super(CNN, self).__init__()
        kwargs = { i.lower(): kwargs[i] for i in kwargs }
        kernel_size = kwargs.get('kernel_size', KERNEL_SIZE)

        # I'm unsure if pad size NEEDS to be 1/2 of kernel size, so I'm leaving it as a variable for now.
        pad = kwargs.get('pad_size', int(kernel_size//2))

        self.padC = [pad, pad, 0, 0] #[pad_size,pad_size,0,0]
        self.padZ = [0, 0, pad, pad] #[0,0,pad_size,pad_size]
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d()

        # This seems like the resolution of pictures, look into it.
        # Currently this is arbritary to make it work and I figured out the number by trial and error.
        # This should be corrected to the actual dependent variables.
        # The first dimension is the number to be concerned about.
        input_resolution = kwargs.get('input_resolution', (32,48))
        fc1_input = 20 * (input_resolution[0]//4) * (input_resolution[1]//4)
        self.fc1 = nn.Linear(fc1_input, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.pad(F.pad(input=x, pad=self.padC, mode='circular'), pad=self.padZ, mode='constant')
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.min_pool2d(self.conv1(x), 2))
        x = F.pad(F.pad(input=x, pad=self.padC, mode='circular'), pad=self.padZ, mode='constant')
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = F.relu(F.min_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

#Same as CNN model, but with zero padding everywhere instead of cylinder padding
class CNN_images(nn.Module): 
    def __init__(self, num_classes):
        super(CNN_images, self).__init__()
        self.padZ = [pad_size,pad_size,pad_size,pad_size]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=KERNEL_SIZE)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=KERNEL_SIZE)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1920, 1024)
        #self.fc1 = nn.Linear(9680, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.pad(input=x, pad=self.padZ, mode='constant')
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #added additional pad for consistency 
        x = F.pad(input=x, pad=self.padZ, mode='constant')
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    


class CNN_simple(nn.Module): 
    def __init__(self, num_classes, num_channels):
        super(CNN_simple, self).__init__()
        self.padC = [pad_size,pad_size,0,0]
        self.padZ = [0,0,pad_size,pad_size]
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=4, kernel_size=KERNEL_SIZE)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=KERNEL_SIZE)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1540, 1024)
        #self.fc2 = nn.Linear(1024, num_classes)
        self.fc2 = nn.Linear(576, num_classes)

    def forward(self, x):
        #print(x.shape)
        x = F.pad(F.pad(input=x, pad=self.padC, mode='circular'), pad=self.padZ, mode='constant')
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #print(x.shape)
        x = F.pad(F.pad(input=x, pad=self.padC, mode='circular'), pad=self.padZ, mode='constant')
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x    
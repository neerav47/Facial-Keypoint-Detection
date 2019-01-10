## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size = (4,4))
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (2,2), stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0)
        
        #Pooling layer
        self.maxpool1 = nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 0)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        
        #Fullyconnected Layers
        self.fc1 = nn.Linear(in_features = 9216, out_features = 1000)
        self.fc2 = nn.Linear(in_features = 1000, out_features = 1000)
        self.fc3 = nn.Linear(in_features = 1000, out_features = 136)
        
        #Define miltiple dropout layers with various zeroout probability
        self.dropout1 = nn.Dropout(p = 0.1)
        self.dropout2 = nn.Dropout(p = 0.2)
        self.dropout3 = nn.Dropout(p = 0.3)
        self.dropout4 = nn.Dropout(p = 0.4)
        self.dropout5 = nn.Dropout(p = 0.5)
        self.dropout6 = nn.Dropout(p = 0.6)
       
        #Custom Weight Initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight = nn.init.uniform(module.weight, a = 0, b = 1)
            elif isinstance(module, nn.Linear):
                module.weight = nn.init.xavier_uniform(module.weight, gain = 1)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #Convolution and pooling layers
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.maxpool2(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        
        x = self.maxpool2(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        
        #Flatten feature maps for fully connected layers
        x = x.view(x.size(0), -1)
        #Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

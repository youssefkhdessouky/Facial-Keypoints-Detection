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
        #conv layers
        # 6 3x3 conv layers.
        #  o/p size = (W-F)/S + 1 # changed 
        self.conv1 = nn.Conv2d(1, 64, 3) #output size = (224-3)/1 +1 = 222 (64,222,222) -> after pooling (64,111,111)
        self.conv2 = nn.Conv2d(64,128, 3)#output size = (111-3)/1 +1 = 109 (128,109,109) -> after pooling (128,54,54)
        self.conv3 = nn.Conv2d(128,256, 3)#output size = (54-3)/1 +1 = 52 (256,52,52) -> after pooling (256,26,26)
        self.conv4 = nn.Conv2d(256,512, 3)#output size = (26-3)/1 +1 = 24 (512,24,24) -> after pooling (512,12,12)
        self.conv5 = nn.Conv2d(512,1024, 3)#output size = (12-3)/1 +1 = 10 (1024,10,10) -> after pooling (1024,5,5)
        self.conv6 = nn.Conv2d(1024,2048, 3)#output size = (5-3)/1 +1 = 3 (2048,3,3) -> after pooling (2048,1,1)
        #maxpooling layer
        self.pool = nn.MaxPool2d(2,2)
        
        #fc layers
        self.fc1 = nn.Linear(in_features = 2048,out_features = 1000) # 2048 *1*1
        self.fc2 = nn.Linear(in_features =1000,out_features =1000 )
        self.fc3 = nn.Linear(in_features =1000,out_features =136 )
        
        #dropout layer for fc 1 and fc 2
        self.drop = nn.Dropout(0.5)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #dropout in each step
        #max pooling&conv d1->d6
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        #flatten
        x = x.view(x.size(0),-1) 
        #2 [ELU] fc with dropout probability of 0.5 and 1 last activation [Linear activation] fc 
        
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

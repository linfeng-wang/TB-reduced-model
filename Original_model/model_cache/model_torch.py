#1 sample batch model

#%%
from pyexpat import model
import statistics
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import torchvision
import torchvision.transforms as transforms
from itertools import chain
from sklearn import metrics as met
import pickle
import icecream as ic

#%%
np.random.seed(42)

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

def conv_block1(in_f, out_f, kernel_size, conv_dropout_rate, padding):
    return nn.Sequential(
        nn.Dropout(p=conv_dropout_rate, inplace=False),
        nn.Conv1d(in_f, out_f, kernel_size = kernel_size, padding = padding),
        nn.BatchNorm1d(out_f),
        nn.ReLU(),
        nn.MaxPool1d(3, stride = 1),  #check here

        )
    
def dense_block1(in_f, out_f, dense_dropout_rate, *args, **kwargs):
    return nn.Sequential(
        nn.Dropout(p=dense_dropout_rate, inplace=False),
        nn.Linear(in_f, out_f, *args, **kwargs),
        nn.BatchNorm1d(1),
        nn.ReLU(),
        nn.Dropout(p=dense_dropout_rate, inplace=False),
        nn.Linear(out_f, out_f, *args, **kwargs),
        nn.BatchNorm1d(1),
        nn.ReLU()
    )
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class raw_seq_model(nn.Module):
    def __init__(self,
                in_channels=4,                  
                n_classes=13, 
                num_filters = 64,
                filter_length=25,
                num_conv_layers=2,     
                num_dense_layers=2,
                conv_dropout_rate = 0.2,
                dense_dropout_rate = 0.5,
                bias = True, 
                return_logits = False):
        super(raw_seq_model, self).__init__() #why do i need to put model name again
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.return_logits = return_logits

        self.conv_layer1 = nn.Conv1d(in_channels, out_channels=num_filters, kernel_size=filter_length)
        torch.nn.init.xavier_uniform_(self.conv_layer1.weight)
        
        self.batch_norm = nn.BatchNorm1d(num_filters)        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, stride = 1)
        self.conv_block1 = conv_block1(num_filters, num_filters, kernel_size=3, padding=1, conv_dropout_rate=conv_dropout_rate)
        self.conv_block1.apply(init_weights)
        
        self.dense_block1 = dense_block1(num_filters, 256, dense_dropout_rate)
        self.dense_block1.apply(init_weights)

        self.linear_logit = nn.Linear(256, n_classes) 
        torch.nn.init.xavier_uniform_(self.linear_logit.weight)

        self.linear_no_logit = nn.Linear(256, n_classes) 
        torch.nn.init.xavier_uniform_(self.linear_no_logit.weight)

        self.predictions = nn.Sigmoid()
        #self.global_maxpool = F.max_pool2d(x, kernel_size=x.size()[2:])


    def forward(self, x):
        # print("Input tensor size:", x.size())
        x = self.conv_layer1(x)
        # print("tensor size after conv_layer1:", x.size())

        x = self.batch_norm(x)
        # print("tensor size after batch_norm:", x.size())

        x = self.relu(x)
        
        # x = self.maxpool(x)
        # print("tensor size after first max_pool:", x.size())

        for i in range(1, self.num_conv_layers + 1):
            x = self.conv_block1(x)
                        
        # print("tensor size after conv_block1:", x.size())

        x = F.max_pool1d(x, kernel_size=x.size()[2:]) #global_maxpool
        #x = self.maxpool(x)
        # x = x.squeeze(dim = -1)
        # x = torch.t(x)
        x = x.permute(0, 2, 1)

        # print("tensor size after global_maxpool:", x.size())

        # for i in range(1, self.num_dense_layers + 1):
            # x = self.dense_block1(x)
        
        # for i in range(1, self.num_dense_layers + 1):
        for layer in self.dense_block1:
            x = layer(x)
            # print(x.size())
        
        # print("tensor size after dense_block:", x.size())

        if self.return_logits:
            prediction = self.linear_logit(x)
            # print("tensor size after dense layer:", x.size())

        else:
            prediction = self.linear_no_logit(x)
            prediction = self.predictions(prediction)
            # print("tensor size after dense layer:", x.size())

        return prediction
        



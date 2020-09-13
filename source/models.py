import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 
from torchvision import datasets, transforms 
from torch.autograd import Variable 
from torch.nn import init
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
from sklearn.preprocessing import OneHotEncoder


class MaxoutCNN(nn.Module):
    def __init__(self, linear_pieces, in_channels, hidden_conv, output_dim, hidden_fcl, dataset, with_dropout):
        super(MaxoutCNN, self).__init__()

        # create list of conv layers for maxout
        self.conv1_list = ListModule(self, "conv1_")
        self.conv2_list = ListModule(self, "conv2_")
        self.conv3_list = ListModule(self, "conv3_")

        self.fc1_list = ListModule(self, "fc1_")
        self.fc2_list = ListModule(self, "fc2_")

        if dataset == 'MNIST':
            padding = 2

        if dataset == 'CIFAR-10':
            padding = 1

        # for convolutional layers
        for _ in range(linear_pieces[0]):
            self.conv1_list.append(nn.Conv2d(in_channels, hidden_conv[0], kernel_size=3, stride=1, padding=0))
            self.conv2_list.append(nn.Conv2d(hidden_conv[0], hidden_conv[1], kernel_size=3, stride=1, padding=padding))
            self.conv3_list.append(nn.Conv2d(hidden_conv[1], hidden_conv[2], kernel_size=3, stride=1, padding=0))

        # for fully connected layers
        if dataset == 'CIFAR-10':
            for _ in range(linear_pieces[1]):
                self.fc1_list.append(nn.Linear(output_dim, hidden_fcl))
                self.fc2_list.append(nn.Linear(hidden_fcl, 10))


        if dataset == 'MNIST':
            self.fcl = nn.Linear(output_dim, 10)

        self.output_dim = output_dim
        self.dataset = dataset
        self.with_dropout = with_dropout


    def forward(self, x): 
        # conv layer 1
        x = F.max_pool2d(self.maxout(x, self.conv1_list), kernel_size=4, stride=2)
        x = F.dropout(x, training=self.training, p=0.5)

        # conv layer 2
        x = F.max_pool2d(self.maxout(x, self.conv2_list), kernel_size=4, stride=2)
        x = F.dropout(x, training=self.training, p=0.5)

        # conv layer 3
        x = F.max_pool2d(self.maxout(x, self.conv3_list), kernel_size=2, stride=2)
        x = F.dropout(x, training=self.training, p=0.5)
        
        x = x.view(-1, self.output_dim)

        # FC layers
        if self.dataset == 'MNIST':
            x = self.fcl(x)

        if self.dataset == 'CIFAR-10': 
            x = self.maxout(x, self.fc1_list)
            x = F.dropout(x, training=self.training, p=0.5)
            x = self.maxout(x, self.fc2_list)

        return F.softmax(x , dim=1)


    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output




class CNN(nn.Module):

    def __init__(self, in_channels=1, n_filters=[32, 32, 64], n_conv_layers=3, output_dim=7*7, fc_hidden_size=230, output_shape=10):
        super(CNN, self).__init__()

        conv_layers = []

        # input layer  
        conv_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n_filters[0], kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            )
        )

        # hidden layers
        for i in range(1, n_conv_layers):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=n_filters[i-1], out_channels=n_filters[i], kernel_size=(3, 3), padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(n_filters[i]),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                    nn.Dropout(p=0.5)
                )
            )

        self.conv = nn.Sequential(*conv_layers)

        self.fcl = nn.Sequential(
            # Layer 1
            nn.Linear(n_filters[-1]*output_dim, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # Layer 2
            nn.Linear(fc_hidden_size, output_shape),
            nn.Softmax(dim=1)
        )

        self.n_filters = n_filters
        self.output_dim = output_dim

    def forward(self, x):
        return self.fcl(self.conv(x).view(-1, self.n_filters[-1]*self.output_dim))




class MLP(nn.Module):
    """ MLP 
    """

    def __init__(self, input_shape, hidden_size, 
        output_shape, n_hidden_layers, dropout_rate, output_dim):
        super(MLP, self).__init__()

        mlp_layers = []

        # input layer  
        mlp_layers.append(nn.Sequential(
            nn.Linear(input_shape, hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )
        )

        # hidden layers
        for i in range(1, n_hidden_layers):
            mlp_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size), 
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )

        # output layer
        mlp_layers.append(nn.Sequential(
            nn.Linear(hidden_size, output_shape), 
            nn.Softmax(dim=1)) 
        )

        # create model
        self.mlp = nn.Sequential(*mlp_layers)

        # keep variables
        self.output_dim = output_dim


    def forward(self, x):
        x = x.view(-1, self.output_dim)
        return self.mlp(x)



class MaxoutMLP(nn.Module):
    def __init__(self, linear_pieces=2, n_hidden_layers=2, input_shape=784, hidden_size=1024, output_shape=10):
        super(MaxoutMLP, self).__init__()

        # for maxout layer
        self.fc = []
        for i in range(n_hidden_layers):
            self.fc.append(ListModule(self, "fc"+str(i)+"_"))

        # first layer
        self.fc[0].append(nn.Linear(input_shape, hidden_size))

        # middle layers
        if n_hidden_layers > 2:
            for i in range(1, n_hidden_layers-2):
                for _ in range(linear_pieces):
                    self.fc[i].append(nn.Linear(hidden_size, hidden_size))

        # last layer
        self.fc[n_hidden_layers-1].append(nn.Linear(hidden_size, output_shape))

        # other parameters
        self.n_hidden_layers = n_hidden_layers
        self.input_shape = input_shape


    def forward(self, x): 
        x = x.view(-1, self.input_shape)

        for i in range(self.n_hidden_layers):
            x = self.maxout(x, self.fc[i])
            if i < 1:
                x = F.dropout(x, training=self.training) 

        return F.softmax(x, dim=1)


    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output




class AllCNN(nn.Module):

    def __init__(self, in_channels, n_filters, output_dim):
        super(DeepCNN, self).__init__()

        self.conv = nn.Sequential(
            # Block 1 -------------------------------------------------------
            # layer 1
            nn.Conv2d(in_channels, n_filters, 3, 1, 1),
            nn.ReLU(),

            # Layer 2
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),

            # Layer 3
            nn.Conv2d(n_filters, n_filters, 3, 2, 1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            # --------------------------------------------------------------


            # Block 2 -------------------------------------------------------
            # Layer 4
            nn.Conv2d(n_filters, 2*n_filters, 3, 1, 1),
            nn.BatchNorm2d(2*n_filters),
            nn.ReLU(),

            # Layer 5
            nn.Conv2d(2*n_filters, 2*n_filters, 3, 1, 1),
            nn.BatchNorm2d(2*n_filters),
            nn.ReLU(),

            # Layer 6
            nn.Conv2d(2*n_filters, 2*n_filters, 3, 2, 1),
            nn.BatchNorm2d(2*n_filters),
            nn.ReLU(),
            # --------------------------------------------------------------


            # Block 3 -------------------------------------------------------
            # Layer 7
            nn.Conv2d(2*n_filters, 4*n_filters, 3, 1, 1),
            nn.BatchNorm2d(4*n_filters),
            nn.ReLU(),

            # Layer 8
            nn.Conv2d(4*n_filters, 4*n_filters, 3, 1, 1),
            nn.BatchNorm2d(4*n_filters),
            nn.ReLU(),

            # Layer 9
            nn.Conv2d(4*n_filters, 4*n_filters, 3, 2, 1),
            nn.BatchNorm2d(4*n_filters),
            nn.ReLU(),
            # --------------------------------------------------------------

            # Block 4 -------------------------------------------------------
            # Layer 10
            nn.Conv2d(4*n_filters, 8*n_filters, 3, 1, 1),
            nn.BatchNorm2d(8*n_filters),
            nn.ReLU(),

            # Layer 11
            nn.Conv2d(8*n_filters, 8*n_filters, 3, 1, 1),
            nn.BatchNorm2d(8*n_filters),
            nn.ReLU(),

            # Layer 12
            nn.Conv2d(8*n_filters, 8*n_filters, 3, 2, 1),
            nn.BatchNorm2d(8*n_filters),
            nn.ReLU(),
            # --------------------------------------------------------------

            # Average Pooling 
            nn.AdaptiveAvgPool2d(1*1)
        )

        self.fcl = nn.Sequential(
            # Layer 13
            # output_dim = 8*self.n_filters*1*1
            nn.Linear(8*n_filters*1*1, 10),
            nn.Softmax(dim=1)
        )

        self.output_dim = output_dim
        self.n_filters = n_filters


    def forward(self, x):
        return self.fcl(self.conv(x).view(-1, 8*self.n_filters*1*1))





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





class ListModule(object):
    """
    Creates list of layers for maxout
    taken directly from https://github.com/Duncanswilson/maxout-pytorch/blob/master/maxout_pytorch.ipynb

    """
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

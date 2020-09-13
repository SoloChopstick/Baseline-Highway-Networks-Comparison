import os
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class MNISTDataLoader:
    def __init__(self, batch_size_train=50, batch_size_valid = 50, valid_size=10000, shuffle=False, download=False):
        
        # file path
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
            
        # transformation to be applied to dataset
        normalizer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

        # load the dataset
        train_dataset = datasets.MNIST(root=root, train=True, 
                    download=download, transform=normalizer)

        valid_dataset = datasets.MNIST(root=root, train=True, 
                    download=download, transform=normalizer)

        test_set = datasets.MNIST(root=root, train=True, transform=normalizer, download=download)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = num_train - 2*valid_size

        if shuffle == True:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx, test_idx = indices[:split], indices[split:(split+valid_size)], indices[(split+valid_size):]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        print(len(train_sampler))
        print(len(valid_sampler))
        print(len(test_sampler))

        self.train_loader = torch.utils.data.DataLoader(train_dataset, 
                        batch_size=batch_size_train, sampler=train_sampler)

        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                        batch_size=batch_size_valid, sampler=valid_sampler)

        self.test_loader = torch.utils.data.DataLoader(
                                    dataset=test_set,
                                    batch_size=batch_size_valid,
                                    shuffle=False,
                                    sampler=test_sampler)

        self.show_batch()


    def show_batch(self):
        """
        Plots batch of mnist images
        """

        data_iter = iter(self.train_loader)
        images, labels = data_iter.next()

        print('Labels: ', labels)
        print('Batch shape: ', images.size())

        im = utils.make_grid(images)
        plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
        plt.show()


class CIFAR10DataLoader:
    def __init__(self, batch_size_train=50, batch_size_valid = 50, valid_size=7500, shuffle=False, download=False):
        
        # file path
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)

        # transformation to be applied to dataset
        # normalizer = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        Z, mean, std= self.computeZCAMAtrix()

        normalizer = transforms.Compose(
        [     transforms.ToTensor(),
              transforms.Normalize(mean, std),
              transforms.LinearTransformation(Z)
              ])

        # load the dataset
        train_dataset = datasets.CIFAR10(root=root, train=True, 
                    download=download, transform=normalizer)

        valid_dataset = datasets.CIFAR10(root=root, train=True, 
                    download=download, transform=normalizer)

        test_set = datasets.CIFAR10(root=root, train=True, transform=normalizer, download=download)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = num_train - 2*valid_size

        if shuffle == True:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx, test_idx = indices[:split], indices[split:(split+valid_size)], indices[(split+valid_size):]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # print(len(train_sampler))
        # print(len(valid_sampler))
        # print(len(test_sampler))

        self.train_loader = torch.utils.data.DataLoader(train_dataset, 
                        batch_size=batch_size_train, sampler=train_sampler)

        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                        batch_size=batch_size_valid, sampler=valid_sampler)

        self.test_loader = torch.utils.data.DataLoader(
                                    dataset=test_set,
                                    batch_size=batch_size_valid,
                                    sampler=test_sampler,
                                    shuffle=False)


        # self.show_batch()


    def computeZCAMAtrix(self):

        #This function computes the ZCA matrix for a set of observables X where
        #rows are the observations and columns are the variables (M x C x W x H matrix)
        #C is number of color channels and W x H is width and height of each image
        
        root = './data' 
           
        
        temp = datasets.CIFAR10(root = root,
                                      train = True,
                                      download = False)
            
      
        #normalize the data to [0 1] range
        temp.data=temp.data/255
        # temp.train = temp.train
        print(temp.data.shape)
        # temp = np.array(temp)
        
        #compute mean and std and normalize the data to -1 1 range with 1 std
        mean=(temp.data.mean(axis=(0,1,2)))
        std=(temp.data.std(axis=(0,1,2)))   
        temp.data=np.multiply(1/std,np.add(temp.data,-mean)) 
        
        
        #reshape data from M x C x W x H to M x N where N=C x W x H 
        X = temp.data
        X = X.reshape(-1, 3072)
        
        # compute the covariance 
        cov = np.cov(X, rowvar=False)   # cov is (N, N)
        
        # singular value decomposition
        U,S,V = np.linalg.svd(cov)     # U is (N, N), S is (N,1) V is (N,N)
        # build the ZCA matrix which is (N,N)
        epsilon = 1e-5
        zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
      


        return (torch.from_numpy(zca_matrix).float(), mean, std) 


    def show_batch(self):
        """
        Plots batch of mnist images
        """

        data_iter = iter(self.train_loader)
        images, labels = data_iter.next()

        print('Labels: ', labels)
        print('Batch shape: ', images.size())

        img = utils.make_grid(images)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()



if __name__ == '__main__':
    MNISTDataLoader()
    CIFAR10DataLoader()
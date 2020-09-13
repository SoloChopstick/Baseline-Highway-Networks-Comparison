import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from torchvision import models
import csv
from Dataloaders2 import MNISTDataLoader, CIFAR10DataLoader
from models import ListModule, MaxoutCNN, MLP, CNN, MaxoutMLP, DeepCNN, count_parameters
import os
from config import config, get_name, maxout_cnn, maxout_mlp, cnn, mlp, all_cnn
import datetime
import time


# For GPU
cuda_available = torch.cuda.is_available()
print(cuda_available)


# OUTPUT STORAGE
# -----------------------------------------------------------------------------

# folder name based on hyperparameters
folder = get_name(config, cnn, mlp, maxout_cnn, maxout_mlp, all_cnn)
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
folder = folder+'_'+st

# create paths to performance and paramters results
parameters_path = 'results/'+folder+'/parameters'
performance_path = 'results/'+folder+'/performance'
performance_file = 'results/'+folder+'/performance/performance_log.txt'

if not os.path.exists(parameters_path):
    os.makedirs(parameters_path)

if not os.path.exists(performance_path):
    os.makedirs(performance_path)

with open(os.path.join(performance_path,'performance_log.txt'), "w+") as file1:
    file1.write('PERFORMANCE LOGS')


# LOAD DATA
# -----------------------------------------------------------------------------

if config['dataset'] == 'MNIST':
    data = MNISTDataLoader(
        batch_size_train=config['batch_size_train'], 
        batch_size_valid=config['batch_size_valid'],
        valid_size=config['valid_size']
        )

if config['dataset'] == 'CIFAR-10':
    data = CIFAR10DataLoader(
        batch_size_train=config['batch_size_train'], 
        batch_size_valid=config['batch_size_valid'],
        valid_size=config['valid_size']
        )


train_loader, valid_loader, test_loader = data.train_loader, data.valid_loader, data.test_loader



# INSTANTIATE CLASSIFIER
# -----------------------------------------------------------------------------

if config['model'] == 'MLP':
    classifier = MLP(
        n_hidden_layers=mlp[config['dataset']]['n_hidden_layers'],
        input_shape=mlp[config['dataset']]['input_shape'],
        hidden_size=mlp[config['dataset']]['hidden_size'],
        output_shape=mlp[config['dataset']]['output_shape'],
        output_dim=mlp[config['dataset']]['output_dim'],
        dropout_rate=mlp[config['dataset']]['dropout_rate']
        )


if config['model'] == 'CNN':
    classifier = CNN(
        in_channels=cnn[config['dataset']]['in_channels'], 
        n_filters=cnn[config['dataset']]['n_filters'], 
        n_conv_layers=cnn[config['dataset']]['n_conv_layers'], 
        output_dim=cnn[config['dataset']]['output_dim'], 
        fc_hidden_size=cnn[config['dataset']]['fc_hidden_size'], 
        output_shape=cnn[config['dataset']]['output_shape']
        )

if config['model'] == 'MaxoutMLP':
    classifier = MaxoutMLP(
        linear_pieces=maxout_mlp[config['dataset']]['linear_pieces'],
        n_hidden_layers=maxout_mlp[config['dataset']]['n_hidden_layers'],
        input_shape=maxout_mlp[config['dataset']]['input_shape'],
        hidden_size=maxout_mlp[config['dataset']]['hidden_size'],
        output_shape=maxout_mlp[config['dataset']]['output_shape']
        )

if config['model'] == 'MaxoutCNN':
    classifier = MaxoutCNN(
        linear_pieces=maxout_cnn[config['dataset']]['linear_pieces'], 
        in_channels=maxout_cnn[config['dataset']]['in_channels'], 
        hidden_conv=maxout_cnn[config['dataset']]['hidden_conv'], 
        output_dim=maxout_cnn[config['dataset']]['output_dim'], 
        hidden_fcl=maxout_cnn[config['dataset']]['hidden_fcl'], 
        dataset=config['dataset'], 
        with_dropout=maxout_cnn[config['dataset']]['with_dropout']
        )

if config['model'] == 'AllCNN':
    classifier = AllCNN(
        in_channels=all_cnn[config['dataset']]['in_channels'],
        n_filters=all_cnn[config['dataset']]['n_filters'],
        output_dim=all_cnn[config['dataset']]['output_dim']
        )


# number of parameters
print(count_parameters(classifier))

# For GPU
if cuda_available:
    classifier = classifier.cuda()


# OPTIMIZER AND CRITERION
# -----------------------------------------------------------------------------

optimizer = torch.optim.Adam(classifier.parameters(), lr=config['lr'], weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# instantiate lists for validation and training error
train_error = []
train_loss = []
val_error = []
val_loss = []
test_error = []
test_loss = []

train_best_accuracy = 0.
valid_best_accuracy = 0.

# Start training
for epoch in range(config['epochs']):

    # Per batch
    losses_train = []
    losses_val = []
    total_train = 0
    correct_train = 0
    i = 1

    # TRAINING ----------------------------------------------------------------
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # For GPU
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()

        # gradient
        optimizer.zero_grad()

        # forward pass
        outputs = classifier(inputs)

        # prediction
        _, predicted = torch.max(outputs.data, 1)

        # loss
        loss = criterion(outputs, targets)

        # backprop
        loss.backward()
        optimizer.step()

        # append batch results
        losses_train.append(loss.data.item())
        total_train += targets.size(0)
        correct_train += predicted.eq(targets.data).cpu().sum().item()

        # print status
        print('[Epoch] '+str(epoch)+'/'+str(config['epochs'])+' [Batch] '+str(i)+' [Loss] '+str(round(np.mean(losses_train),4)))
        i+=1


    # overall results
    train_error.append(100.*(1-correct_train/total_train))
    train_loss.append(np.mean(losses_train))

    # save parameters
    pt_file = 'model_parameters_'+str(epoch)+'.pt'
    torch.save(classifier.state_dict(), os.path.join(parameters_path, pt_file))


    # Evaluate on validation set
    classifier.eval()
    total_val = 0
    correct_val = 0

    # VALIDATION ------------------------------------------------------------
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        # For GPU
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()

        # forward pass
        outputs = classifier(inputs)

        # prediction
        _, predicted = torch.max(outputs.data, 1)

        # results per batch
        loss = criterion(outputs, targets)
        losses_val.append(loss.data.item())
        total_val += targets.size(0)
        correct_val += predicted.eq(targets.data).cpu().sum().item()

    # overall results
    val_error.append(100.*(1-correct_val/total_val))
    val_loss.append(np.mean(losses_val))

    # print performance
    train_accuracy = 100.*correct_train/total_train
    valid_accuracy = 100.*correct_val/total_val
    print('Epoch : %d Train Acc : %.3f' % (epoch, train_accuracy))
    print('Epoch : %d Valid. Acc : %.3f' % (epoch, valid_accuracy))
    print('--------------------------------------------------------------')

    classifier.train()

    # performance log text file
    train_best_accuracy = max(train_best_accuracy, train_accuracy)
    valid_best_accuracy = max(valid_best_accuracy, valid_accuracy)
    line = '[Epoch] '+str(epoch)+' [Accuracy Train] '+str(train_accuracy)+' [Accuracy Valid.] '+str(valid_accuracy)+' [Best Accuracy Train] '+str(train_best_accuracy)+' [Best Accuracy Valid.] '+str(valid_best_accuracy)

    with open(os.path.join(performance_path,'performance_log.txt'), 'a+') as file1:
        file1.write('\n'+line)



# FINAL ACCURACY
# -----------------------------------------------------------------------------
# Evaluate on test set
classifier.eval()
total_test = 0
correct_test = 0
losses_test = []

# TEST ------------------------------------------------------------
for batch_idx, (inputs, targets) in enumerate(test_loader):
    # For GPU
    if cuda_available:
        inputs, targets = inputs.cuda(), targets.cuda()

    # forward pass
    outputs = classifier(inputs)

    # prediction
    _, predicted = torch.max(outputs.data, 1)

    # results per batch
    loss = criterion(outputs, targets)
    losses_test.append(loss.data.item())
    total_test += targets.size(0)
    correct_test += predicted.eq(targets.data).cpu().sum().item()

# overall results
test_error.append(100.*(1-correct_test/total_test))
test_loss.append(np.mean(losses_test))

# print performance
test_accuracy = 100.*correct_test/total_test
print('Epoch : %d Train Acc : %.3f' % (epoch, train_accuracy))
print('Epoch : %d Valid. Acc : %.3f' % (epoch, valid_accuracy))
print('Epoch : %d Test Acc : %.3f' % (epoch, test_accuracy))
print('--------------------------------------------------------------')

# FINAL PERFORMANCE GRAPHS
# -----------------------------------------------------------------------------
plt.plot(train_error, label='Train')
plt.plot(val_error, label='Valid')
plt.ylabel('Error (%)')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(os.path.join(performance_path, 'error.png'))
plt.show()

plt.plot(train_loss, label='Train')
plt.plot(val_loss, label='Valid')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(os.path.join(performance_path, 'loss.png'))
plt.show()

# baseline-highway-networks
COMP551-w19 FinalProject Group 76

Marie-Eve Malette-Campeau, Han Wen Xie, David Kang

# Requirements
```
pytorch
torchvision
torch
matplotlib
numpy
keras
os
time
datetime
csv
```

# Folder structure
```
code
 |
 |--> source : Code folder
 |    |
 |    --> main.py : Main file to run
 |    --> models.py : contains all models except the DeepCNN
 |    --> config.py : Editable file controlling the hyperparameters
 |    --> Dataloaders2.py : loads MNIST and CIFAR-10 
 |--> results : Results folder
 |--> deep_cnn_cifar10.ipynb : Notebook for DeepCNN on CIFAR-10

```

# Running All Models, except DeepCNN
From  baseline-highway-networks folder call: ```python3 main.py```

# Running DeepCNN Model
From baseline-highway-networks folder, launch anaconda jupyter notebook: ```deep_cnn_cifar10.ipynb```

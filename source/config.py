
config = {

    'model': 'CNN', # MaxoutCNN, MaxoutMLP, MLP, CNN, AllCNN
    'dataset': 'CIFAR-10', # MNIST, CIFAR-10

    'lr': 3e-4,
    'epochs': 500, 
    'batch_size_train': 128,
    'batch_size_valid': 64, 
    'valid_size': 10000

    }


# ARCHITECTURE CONFIGURATIONS
# -------------------------------------------------------------------------

# MLP
#---------------------------------------------------------------------------

mlp = {
# CIFAR-10
    'CIFAR-10': {
        'n_hidden_layers': 3, 
        'input_shape': 3*32*32, 
        'hidden_size': 1500, 
        'output_shape': 10,
        'dropout_rate': 0.5,
        'output_dim' : 3*32*32
    },

# MNIST
    'MNIST': {
        'input_shape': 28*28, 
        'hidden_size': 500, 
        'output_shape': 10, 
        'n_hidden_layers': 2, 
        'dropout_rate': 0.5, 
        'output_dim': 28*28
    }  

}


# CNN
#---------------------------------------------------------------------------

cnn = {
# CIFAR-10
    'CIFAR-10': {
        'in_channels': 3, 
        'n_filters': [96, 192, 192], 
        'n_conv_layers': 3, 
        'output_dim': 8*8, 
        'fc_hidden_size': 128, 
        'output_shape': 10
    },

# MNIST
    'MNIST': {
        'in_channels': 1, 
        'n_filters': [32, 32, 64], 
        'n_conv_layers': 3, 
        'output_dim': 7*7, 
        'fc_hidden_size': 230, 
        'output_shape': 10
    }  

}


# MaxoutMLP
#---------------------------------------------------------------------------

maxout_mlp = {
# CIFAR-10
    'CIFAR-10': {
        'linear_pieces': 2, 
        'n_hidden_layers': 2, 
        'input_shape': 3*32*32, 
        'hidden_size': 3500, 
        'output_shape': 10
    },

# MNIST
    'MNIST': {
        'linear_pieces': 2, 
        'n_hidden_layers': 2, 
        'input_shape': 28*28, 
        'hidden_size': 500, 
        'output_shape': 10
    }  

}


# MaxoutCNN
#---------------------------------------------------------------------------


maxout_cnn = {
# CIFAR-10
    'CIFAR-10': {
        'linear_pieces': [2,5], 
        'in_channels': 3, 
        'hidden_conv': [96, 192, 192], 
        'output_dim': 192*2*2, 
        'hidden_fcl': 500, 
        'with_dropout': True
    },

# MNIST
    'MNIST': {
        'linear_pieces': [2,2], 
        'in_channels': 1, 
        'hidden_conv': [48, 48, 24], 
        'output_dim': 24*2*2, 
        'hidden_fcl': 200, 
        'with_dropout': True
    }  

}

# DeepCNN
#---------------------------------------------------------------------------

all_cnn = {
# CIFAR-10
    'CIFAR-10': {
    'in_channels': 3,
    'n_filters': 32,
    'output_dim': 8*32*1*1
    },

# MNIST
    'MNIST': {
    'in_channels': 1,
    'n_filters': 32,
    'output_dim': 8*32*1*1
    } 

}



def get_name(config, cnn, mlp, maxout_cnn, maxout_mlp, all_cnn):
    model = config['model']
    dataset = config['dataset']

    name = 'C'
    for k, v in config.items(): 
        name = name+'_'+k+'_'+str(v)

    if model == 'MaxoutCNN':
        for k, v in maxout_cnn[dataset].items(): 
            name = name+'_'+k+'_'+str(v)

    if model == 'CNN':
        for k, v in cnn[dataset].items(): 
            name = name+'_'+k+'_'+str(v)

    if model == 'MLP':
        print('yes')
        for k, v in mlp[dataset].items(): 
            name = name+'_'+k+'_'+str(v)

    if model == 'MaxoutMLP':
        for k, v in maxout_mlp[dataset].items(): 
            name = name+'_'+k+'_'+str(v)

    if model == 'AllCNN':
        for k, v in all_cnn[dataset].items(): 
            name = name+'_'+k+'_'+str(v)


    return name






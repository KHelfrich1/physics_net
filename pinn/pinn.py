"""
@author: Kyle Helfrich
"""

# Import modules
import argparse 
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from distutils.util import strtobool
from utils import training_data, train_network, visualize_model

# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using device: {}'.format(device))

# Set random seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Neural Network
class Net(nn.Module):
    
    def __init__(self, settings):
        super(Net, self).__init__()
        self.input_dim = 2
        self.output_dim = 1
        self.hidden_sizes = settings.hidden_sizes
        self.layers = nn.ModuleList()
        self.activation_func = settings.activation_func
        current_dim = self.input_dim
        for dim in self.hidden_sizes:
            self.layers.append(nn.Linear(current_dim, dim))
            current_dim = dim
        self.layers.append(nn.Linear(current_dim, self.output_dim))


    def forward(self, x):
        for layer in self.layers[:-1]:
            #x = F.relu(layer(x))
            x = self.activation_func(layer(x))
        out = self.layers[-1](x)
        
        return out
        
# Main code component
def main(settings):
    
    # Gather training data
    U0x, Ubp, Ubu, Ubux, Ui, periodic_func  = training_data(settings)

    # Initialize activation function
    activation_dict = {'relu' : F.relu,
                       'leaky_relu' : F.leaky_relu,
                       'tanh' : torch.tanh}
    settings.activation_func = activation_dict[settings.activation]

    # Operations if training a network
    if settings.train:
        # Initialize neural network
        model = Net(settings)

        # Initialize optimizer object    
        opt_dict = {'sgd' : optim.SGD(model.parameters(), lr=settings.lr),
                    'adagrad' : optim.Adagrad(model.parameters(), lr=settings.lr),
                    'adam' : optim.Adam(model.parameters(), lr=settings.lr),
                    'rmsprop' : optim.RMSprop(model.parameters(), lr=settings.lr)}
        opt = opt_dict[settings.optimizer]

        # Train Neural Net
        model = train_network(model, U0x, Ubp, Ubu, Ubux, Ui, opt, settings.epochs, settings, periodic_func)
        
        # Saving Neural Net
        if settings.save:
            torch.save(model, settings.save_name)
    else:
        # Load model
        try:
            print("Loading model...")
            model = torch.load(settings.save_name)    
            model.eval()
            print("Model loaded")
        except OSError as error:
            print(error)
            print("Unable to load model {}. Check that this model has been saved".format(settings.save_name))
    
    # Visualizing solution
    if settings.visualize:
        visualize_model(model, settings)        
    
if __name__ == "__main__": 

    desp = 'Running PINN...'
    print(desp)
    
    # Check to see if "experiments" folder exists, if not, it creates it.
    if not os.path.exists('./experiments'):
        print('"experiments" folder does not exist. Creating folder...')
        os.makedirs('./experiments/')
        
    parser = argparse.ArgumentParser(description=("""{}""".format(desp)), 
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
    # Loading model parameters
    parser.add_argument('--hidden_sizes', type=lambda s: [int(item) for item in s.split(',')], default='100,100,100', help="String of hidden sizes seperated by a comma, for example: '100,100,100,2'")
    parser.add_argument('--optimizer', type=str, default='adam', help='String of optimizer type consisting of [sgd, adam, rmsprop]')
    parser.add_argument('--lr', type=float, default=1e-3, help='Float of the learning rate.')
    parser.add_argument('--epochs', type=int, default=500000, help='Integer of the number of epochs to train the network.')
    parser.add_argument('--activation', type=str, default='tanh', help='String of which type of activation function to use consisting of [relu, leaky_relu, tanh]')
    
    # Loading initial conditions
    parser.add_argument('--xL', type=float, default=0, help='Float of the left endpoint of the boundary.')
    parser.add_argument('--xR', type=float, default=1.0, help='Float of the right endpoint of the boundary.')
    parser.add_argument('--T', type=float, default=2, help='Float of the total time.')
    parser.add_argument('--N0', type=int, default=100, help='Integer of the number of datapoints to use for the initial conditions.')
    parser.add_argument('--Nb', type=int, default=1000, help='Integer of the number of datapoints to use for the boundary conditions.')
    parser.add_argument('--Ni', type=int, default=20000, help='Integer of the number of datapoints to use for the interior points.')
    parser.add_argument('--initial_condition_func', type=str, default='sin', help='String of the initial periodic function consisting of [sin, cos]. Actual function is sin(2pi*x), cos(2pi*x), etc.')
    parser.add_argument('--boundary_scaling', type=float, default=0.10, help='Float of the scaling to apply to the periodic boundary condition function.')

    # Loading user defined operations
    parser.add_argument('--train', type=strtobool, default=0, help='String of whether to train the network or not.')
    parser.add_argument('--save', type=strtobool, default=0, help='String of whether to save the network or not.')
    parser.add_argument('--visualize', type=strtobool, default=1, help='String of whether to render 3D surface of solution.')

    args = parser.parse_args()
    
    # Change train variable from integer to boolean
    args.train = bool(args.train)
    
    # Change save variable from integer to boolean
    args.save = bool(args.save)
    
    # Change visualize variable from integer to boolean
    args.visualize = bool(args.visualize)
    
    
    # Check to see if "model" folder exists, if not, it creates it.
    if args.save:  
        if not os.path.exists('./models'):
            print('"models" folder does not exist. Creating folder...')
            os.makedirs('./models/')
    args.save_name = os.path.join('models', 'scaling_{:.2f}_T_{:1f}.pt'.format(args.boundary_scaling, args.T))
    
    main(args)
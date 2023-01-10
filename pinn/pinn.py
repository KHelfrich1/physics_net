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

from utils import training_data, train_network

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
        current_dim = self.input_dim
        for dim in self.hidden_sizes:
            self.layers.append(nn.Linear(current_dim, dim))
            current_dim = dim
        self.layers.append(nn.Linear(current_dim, self.output_dim))


    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        
        return out
        
# Main code component
def main(settings):
    
    # Gather training data
    U0x, Ubp, Ubu, Ubux, Ui, periodic_func  = training_data(settings)

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
    
    # Loading initial conditions
    parser.add_argument('--xL', type=float, default=0, help='Float of the left endpoint of the boundary.')
    parser.add_argument('--xR', type=float, default=1.0, help='Float of the right endpoint of the boundary.')
    parser.add_argument('--T', type=float, default=2, help='Integer of what to multiply pi by.')
    parser.add_argument('--N0', type=int, default=100, help='Integer of the number of datapoints to use for the initial conditions.')
    parser.add_argument('--Nb', type=int, default=1000, help='Integer of the number of datapoints to use for the boundary conditions.')
    parser.add_argument('--Ni', type=int, default=20000, help='Integer of the number of datapoints to use for the interior points.')
    parser.add_argument('--initial_condition_func', type=str, default='sin', help='String of the initial periodic function consisting of [sin, cos]')
    parser.add_argument('--boundary_scaling', type=float, default=0.10, help='Float of the scaling to apply to the periodic boundary condition function.')

    args = parser.parse_args()
    
    # Set time variable
    args.T = args.T*np.pi   
    main(args)
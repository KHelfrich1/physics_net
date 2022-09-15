# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:31:54 2022

@author: Kyle Helfrich
"""

import torch
import numpy as np

# Assigning device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Device: {}'.format(device))

def training_data(settings):
    '''
    Parameters
    ----------
    settings : Object containing the following attributes:
        xL  - Lower limit of x-values
        xR - Right limit of x-values
        T- End of time interval
        N0 - Number of initial condition data points to use for training
        Nb - Number of boundary data points to use for training
        Ni - Number of interior data points to use for training
        initial_condition_func - String of which periodic funciton to use
        initial_condition_scaling - Float of the scaling to apply to the periodic initial condition
        
    Returns
    -------
    U0 - Random initial conditions. Numpy array of size (N0, 3) of the form (x,t,u) = (xi, 0, 0)
    Ub - Random boundary conditions. Numpy array of size (Nb, 3) of the form (x,t,u) = (xi, ti, ui)
    Ui - Random interior conditions. Numpy array of size (Ni, 3) of the form (x,t,u) = (xi, ti, 0)    
    '''
    
    # Load initial condition function
    init_func = {'sin' : np.sin,
                 'cos' : np.cos}
    base_func = init_func[settings.initial_condition_func]
    init_func = lambda x : settings.initial_condition_scaling*base_func(x)
    
    # Create random initial conditions 
    dx = (settings.xR - settings.xL)/(10*settings.N0)
    x_values = np.arange(settings.xL, settings.xR+dx, dx)
    x_rand = np.random.randint(0, len(x_values), size=(settings.N0))
    x0 = np.reshape(x_values[x_rand], (-1,1))
    U0 = np.concatenate([x0, np.zeros_like(x0), np.zeros_like(x0)], axis=-1)
    
    # Create random boundary conditions
    dt = settings.T/(10*settings.Nb)
    t_values = np.arange(0, settings.T+dt, dt)
    
    dTL = int(np.floor(settings.Nb/2.0))
    dTR = settings.Nb-dTL
    
    t_randL = np.random.randint(0, len(t_values), size=(dTL))
    t_randR = np.random.randint(0, len(t_values), size=(dTR))
    
    tL = np.reshape(t_values[t_randL], (-1,1))
    tR = np.reshape(t_values[t_randR], (-1,1))
    
    ub = init_func(tL)
    UbL = np.concatenate([settings.xL*np.ones_like(tL), tL, ub], axis=-1)
    UbR = np.concatenate([settings.xR*np.ones_like(tR), tR, np.zeros_like(tR)], axis=-1)
    Ub =np.concatenate([UbL, UbR], axis=0)
    
    
    # Gather interior points
    dx = (settings.xR - settings.xL)/(10*settings.Ni)
    x_values = np.arange(settings.xL, settings.xR+dx, dx)
    x_rand = np.random.randint(0, len(x_values), size=(settings.Ni))
    xi = np.reshape(x_values[x_rand], (-1,1))
    
    dt = settings.T/(10*settings.Ni)
    t_values = np.arange(0, settings.T+dt, dt)
    t_rand = np.random.randint(0, len(t_values), size=(settings.Ni))
    ti = np.reshape(t_values[t_rand], (-1,1))
    
    Ui = np.concatenate([xi, ti, np.zeros_like(xi)]) 
       
    return U0, Ub, Ui

###############################################################################

#def losses()


###############################################################################

def train_network(model, U0, Ub, Ui, opt, epochs):
    '''
    Parameters
    ----------
        model : Pytorch neural network object
        U0 : Initial conditions 
        Ub : Boundary conditions
        Ui : Interior point conditions
        opt : Optimizer
        epochs : Integer of the number of epochs the model can be run 
        
    Returns
    -------
    losses : List of loss values
    model : trained model
    '''    

    # Convert data types
    x0 = torch.from_numpy(U0[:,:2]).float().to(device)
    u0 = torch.from_numpy(U0[:,-1]).float().to(device)
    
    temp = []
    
    #xi = torch.from_numpy
    
    
    
    

    # Gather initial results 




      
    model.to(device)
    u0_hat = model(torch.from_numpy(U0[:,:2]).float().to(device))
    print(u0_hat.shape)
    
    #u0_hat = model(torch.from_numpy(U0[:,:2]))
    
    
    
    
    
    #for i in range(1, settings.epochs+1):
        
    

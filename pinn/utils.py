# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:31:54 2022

@author: Kyle Helfrich
"""

import numpy as np

def initial_data(settings):
    '''
    Parameters
    ----------
    settings : Object containing the following attributes:
        xL  - Lower limit of x-values
        xR - Right limit of x-values
        N0 - Number of data points to use for training
        initial_condition_func - String of which periodic funciton to use
        initial_condition_scaling - Float of the scaling to apply to the periodic initial condition
        
    Returns
    -------
    X0 - Numpy array of size (N0, 2) of the form (x,t) = (xi, 0)
    U0 - Numpy array of size (N0, 2) of the form (x, u) = (xi, ui)
    '''
    
    # Load initial condition function
    init_func = {'sin' : np.sin,
                 'cos' : np.cos}
    base_func = init_func[settings.initial_condition_func]
    init_func = lambda x : settings.initial_condition_scaling*base_func(x)
    
    # Create random samples
    dx = (settings.xR - settings.xL)/(10*settings.N0)
    x_values = np.arange(settings.xL, settings.xR+dx, dx)
    x_rand = np.random.randint(0, len(x_values), size=(settings.N0))
    x0 = x_values[x_rand]
    
    # Create initial conditions
    u0 = init_func(x0)
 
    # Create matrics
    X0 = np.concatenate(((x0,1), (0*x0,1)), axis=1)
    print(X0.shape, X0[:5,:])
    
    
    
    return 1, 2
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
        boundary_scaling - Float of the scaling to apply to the periodic initial condition
        
    Returns
    -------
    U0x - Random initial conditions. Numpy array of size (N0,1). Entries consist
          of random x values for IC u(x,0)=0.
    Ubp - Random boundary conditions of the periodic function. Numpy array of 
          size (Nb/3, 2). Entries consist of random t and u values for 
          BC u(0,t) = alpha*h(t).
    Ubu - Random boundary conditions. Numpy array of size (Nb/3,1). Entries 
          consist of random t values for BC u(1,t) = 0.
    Ubux - Random boundary conditions. Numpy array of size (Nb/3,1). Entries 
          consist of random t values for BC ux(1,t) = 0.
    Ui - Random interior conditions. Numpy array of size (Ni, 2). Entries 
        consist of random (x,t) values for the PDE ut + ux + uux + uxxx=0.    
    '''
    
    # Load periodic function
    init_func = {'sin' : np.sin,
                 'cos' : np.cos}
    base_func = init_func[settings.initial_condition_func]
    periodic_func = lambda x : settings.boundary_scaling*base_func(x)
    
    # Create random initial conditions 
    dx = (settings.xR - settings.xL)/(10*settings.N0)
    x_values = np.arange(settings.xL, settings.xR+dx, dx)
    x_rand = np.random.randint(0, len(x_values), size=(settings.N0))   
    U0x = np.reshape(x_values[x_rand], (-1,1))
        
    # Create random boundary conditions
    dt = settings.T/(10*settings.Nb)
    t_values = np.arange(0, settings.T+dt, dt)
    
    dTL = int(np.ceil(settings.Nb/3.0))
    dTR = settings.Nb-dTL
    
    t_randL = np.random.randint(0, len(t_values), size=(dTL))
    t_randR = np.random.randint(0, len(t_values), size=(dTR))
    
    tL = np.reshape(t_values[t_randL], (-1,1))
    tR = np.array_split(t_values[t_randR], 2)   
    
    Ubu = np.reshape(tR[0], (-1,1))
    Ubux = np.reshape(tR[-1], (-1,1))
    
    ub = periodic_func(tL)
    Ubp = np.concatenate([tL, ub], axis=-1)
        
    # Gather interior points
    dx = (settings.xR - settings.xL)/(10*settings.Ni)
    x_values = np.arange(settings.xL, settings.xR+dx, dx)
    x_rand = np.random.randint(0, len(x_values), size=(settings.Ni))
    xi = np.reshape(x_values[x_rand], (-1,1))
    
    dt = settings.T/(10*settings.Ni)
    t_values = np.arange(0, settings.T+dt, dt)
    t_rand = np.random.randint(0, len(t_values), size=(settings.Ni))
    ti = np.reshape(t_values[t_rand], (-1,1))
    
    Ui = np.concatenate([xi, ti], axis=-1) 
       
    return U0x, Ubp, Ubu, Ubux, Ui

###############################################################################

def losses(model, u0_hat, ubp, ubp_hat, ubu_hat, BCux_pts, ubux_hat, PDE_pts, pdeu_hat):
    
    '''
    Parameters
    ----------
    model : Pytorch neural network object
    u0_hat  - Predicted initial condition results of model
    ubp - Actual boundary condition of periodic function of model
    ubp_hat - Predicted boundary condition of periodic function of model
    ubu_hat - Predited boundary condition of u(1,t) = 0 of model
    BCux_pts - Input values used to test boundary condition ux(1,t) = 0
    ubux_hat - Predicted boundary condition of ux(1,t) = 0
    PDE_pts - Input values used to test pde interior points
    pdeu_hat - Predicted interior points of the model.
        
    Returns
    -------
    mse0 - Mean squared error of the initial conditions.
    mseb - Mean squared error of the boundary conditions.
    msep - Mean square error of the pde condition. 
    mse - Total mean square error.    
    '''

    # Initial condition loss    
    mse0 = torch.mean(torch.square(u0_hat))
    
    # Boundary condition loss
    periodic_loss = torch.nn.MSELoss()
    msebp = periodic_loss(ubp_hat, ubp)
    
    msebu = torch.mean(torch.square(ubu_hat))
    
    
    dubx_hat = torch.autograd.grad(ubux_hat, BCux_pts, grad_outputs=torch.ones_like(ubux_hat))
    dubx_hat = dubx_hat[0][:,0]
    msebux = torch.mean(torch.square(dubx_hat))  
    
    mseb = msebp + msebu + msebux
    
    # PDE condition loss    
    duxt_hat = torch.autograd.grad(pdeu_hat, PDE_pts, grad_outputs=torch.ones_like(pdeu_hat), create_graph=True)
    ut_hat = duxt_hat[0][:,-1]
    ux_hat = duxt_hat[0][:,0]
    uxx_hat = torch.autograd.grad(ux_hat, PDE_pts, grad_outputs=torch.ones_like(ux_hat), create_graph=True)
    uxx_hat = uxx_hat[0][:,0]
    uxxx_hat = torch.autograd.grad(uxx_hat, PDE_pts, grad_outputs=torch.ones_like(uxx_hat), create_graph=True)
    uxxx_hat = uxxx_hat[0][:,0]    
    msep = torch.mean(torch.square(ut_hat + ux_hat + pdeu_hat[:,0]*ux_hat + uxxx_hat))

    mse = mse0 + mseb + msep    
    
    return mse0, mseb, msep, mse
    
###############################################################################

def train_network(model, U0x, Ubp, Ubu, Ubux, Ui, opt, epochs):
    '''
    Parameters
    ----------
    model : Pytorch neural network object
    U0x : Initial conditions 
    Ubp : Boundary conditions of the periodic function
    Ubu : Boundary points for BC u(1,t) = 0
    Ubux : Boundary poitns for BC ux(1,t) = 0
    Ui : Interior point conditions
    opt : Optimizer
    epochs : Integer of the number of epochs the model can be run 
        
    Returns
    -------
    losses : List of loss values
    model : trained model
    '''    

    # Load model 
    model.to(device)
    
    # Process initial conditions
    x0 = torch.from_numpy(U0x).float().to(device)    
    IC_pts = torch.cat([x0, torch.zeros_like(x0)], axis=-1)
    
    # Process boundary conditions
    tbp = torch.from_numpy(np.reshape(Ubp[:,0], (-1,1))).float().to(device)
    ubp = torch.from_numpy(np.reshape(Ubp[:,-1], (-1,1))).float().to(device)
    BCp_pts = torch.cat([torch.zeros_like(tbp),tbp], axis=-1)
    
    tbu = torch.from_numpy(Ubu).float().to(device)
    BCu_pts = torch.cat([torch.ones_like(tbu), tbu], axis=-1)
    
    tbux = torch.from_numpy(Ubux).float().to(device)
    BCux_pts = torch.cat([torch.ones_like(tbux), tbux], axis=-1)
    BCux_pts.requires_grad = True

    # Process interior points
    PDE_pts = torch.from_numpy(Ui).float().to(device)
    PDE_pts.requires_grad = True
   
    # Lists to store losses
    mse0_list, mseb_list, msep_list, mse_list = [], [], [], []

   
    # Start of training loop
    epoch = 0
    while epoch <= epochs:    
        # IC Predictions
        u0_hat = model(IC_pts)

        # BC Predictions
        ubp_hat = model(BCp_pts) 
        ubu_hat = model(BCu_pts)
        ubux_hat = model(BCux_pts) 

        # PDE Predictions    
        pdeu_hat = model(PDE_pts)
   
        # Gather losses
        mse0, mseb, msep, mse = losses(model, u0_hat, ubp, ubp_hat, ubu_hat, BCux_pts, ubux_hat, PDE_pts, pdeu_hat)
        
        mse0_list.append(mse0.cpu().detach().item())
        mseb_list.append(mseb.cpu().detach().item())
        
        msep_list.append(msep.cpu().detach().item())
        mse_list.append(mse.cpu().detach().item())

        # Update model parameters
        if epoch < epochs:
            opt.zero_grad()
            mse.backward()
            opt.step()
        
        print(mse)    
        
        # Update counter
        epoch += 1    



    return model
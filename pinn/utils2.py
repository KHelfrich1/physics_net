# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:31:54 2022

@author: Kyle Helfrich
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# Assigning device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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
        initial_condition_func - String of which periodic function to use
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
                 'cos' : np.cos,
                 'sech' : np.cosh}
    base_func = init_func[settings.initial_condition_func]

    # Handle lack of sech function in numpy
    if settings.initial_condition_func != 'sech':
        periodic_func = lambda x : settings.boundary_scaling*base_func(2*np.pi*x)
    else:       
        periodic_func = lambda x : (12*settings.initial_scaling**2)*(1.0/base_func(settings.initial_scaling*x)**2)

    # Create random initial conditions 
    dx = (settings.xR - settings.xL)/(10*settings.N0)
    x_values = np.arange(settings.xL, settings.xR+dx, dx)
    x_rand = np.random.randint(0, len(x_values), size=(settings.N0))   
    U0x = periodic_func(settings.initial_scaling*np.reshape(x_values[x_rand], (-1,1)))**2

        
    # Create random boundary conditions
    dt = settings.T/(10*settings.Nb)
    t_values = np.arange(0, settings.T+dt, dt)
    
    dTL = int(np.ceil(settings.Nb/3.0))
    dTR = settings.Nb-dTL
    
    t_randL = np.random.randint(0, len(t_values), size=(dTL))
    t_randR = np.random.randint(0, len(t_values), size=(dTR))
    
    UbL = np.reshape(t_values[t_randL], (-1,1))
    tR = np.array_split(t_values[t_randR], 2)   
    
    UbRu = np.reshape(tR[0], (-1,1))
    UbRx = np.reshape(tR[-1], (-1,1))
        
          
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
         
    return U0x, UbL, UbRu, UbRx, Ui, periodic_func

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

def train_network(model, U0x, UbL, UbRu, UbRx, Ui, opt, epochs, settings=None, periodic_func=None):
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
    settings : Optional class object containing various flags
        
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
    ubL = torch.from_numpy(UbL).float().to(device)    
    BCL_pts = torch.cat([torch.zeros_like(ubL),ubL], axis=-1)
    
    ubRu = torch.from_numpy(UbRu).float().to(device)
    BCRu_pts = torch.cat([settings.xR*torch.ones_like(ubRu), ubRu], axis=-1)
        
    ubRx = torch.from_numpy(UbRx).float().to(device)
    BCRx_pts = torch.cat([settings.xR*torch.ones_like(ubRx), ubRx], axis=-1)
    BCRx_pts.requires_grad = True

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
        ubL_hat = model(BCL_pts) 
        ubRu_hat = model(BCRu_pts)
        ubRx_hat = model(BCRx_pts) 

        # PDE Predictions    
        pdeu_hat = model(PDE_pts)
   
        # Gather losses
        mse0, mseb, msep, mse = losses(model, u0_hat, ubL_hat, ubRu_hat, ubRx_hat, pdeu_hat)
        
        mse0_list.append(mse0.cpu().detach().item())
        mseb_list.append(mseb.cpu().detach().item())        
        msep_list.append(msep.cpu().detach().item())
        mse_list.append(mse.cpu().detach().item())

        
        if epoch % int(epochs*0.1) == 0:
            print('Epoch: {:d} MSE: {:.2e} MSE0: {:.2e} MSEb: {:.2e} MSEp: {:.2e}'.format(epoch, mse_list[-1], mse0_list[-1], mseb_list[-1], msep_list[-1]))
            print_losses(mse_list, mse0_list, mseb_list, msep_list, epoch, settings.T)
            test_p(model, settings, periodic_func, epoch)

        # Update model parameters
        if epoch < epochs:
            opt.zero_grad()
            mse.backward()
            opt.step()
                
        # Update counter
        epoch += 1    

    return model


###############################################################################

def print_losses(mse, mse0, mseb, msep, epoch, T):
    '''
    Parameters
    ----------
    mse : List of MSE loss values
    mse0 : List of Initial Condition MSE values or MSE0
    mseb : List of Boundary Conditions MSE values or MSEb
    msep : List of Periodic Boundary Conditions MSE values or MSEp
    epoch : Integer of current epoch
    T : Float of the final time T

    Returns
    -------
    Plot of losses
    '''    
    fig, ax = plt.subplots()
    ax.plot(mse, label="MSE")
    ax.plot(mse0, label="MSE0")
    ax.plot(mseb, label="MSEb")
    ax.plot(msep, label="MSEp")
    ax.legend(loc='upper right')
    #ax.set_yscale('log')
    fig.savefig(os.path.join('experiments', 'losses_T' + '{:.3f}'.format(T) + '_epoch' + str(epoch)+ '.png'))
    plt.close()
    
    
###############################################################################

def test_p(model, settings, periodic_func, epoch):
    '''
    Parameters
    ----------
    model : Pytorch neural network object
    settings : Class object containing various settings
    periodic_func : Periodic function used for boundary conditions.
    epoch : Current training epoch

    Returns
    -------
    Plot of periodic results    
    
    '''  
    
    # Create time array
    dt = settings.T/(settings.Nb)
    t_values = np.arange(0, settings.T+dt, dt)
    
    # Compute periodic conditions
    ub = periodic_func(t_values)
        
    # Compute predicted results
    t_test = torch.from_numpy(np.reshape(t_values, (-1,1))).float().to(device)
    BC_test = torch.cat([torch.zeros_like(t_test), t_test], axis=-1)
    BC_test2 = torch.cat([0.5*settings.xR*torch.ones_like(t_test), t_test], axis=-1)
    BC_test3 = torch.cat([0.8*settings.xR*torch.ones_like(t_test), t_test], axis=-1)
    BC_test4 = torch.cat([settings.xR*torch.ones_like(t_test), t_test], axis=-1)
    
    with torch.inference_mode():
        utest = model(BC_test)
        utest2 = model(BC_test2)
        utest3 = model(BC_test3)
        utest4 = model(BC_test4)

        utest = utest.cpu().detach().numpy().reshape((-1,))
        utest2 = utest2.cpu().detach().numpy().reshape((-1,))
        utest3 = utest3.cpu().detach().numpy().reshape((-1,))
        utest4 = utest4.cpu().detach().numpy().reshape((-1,))
        
    fig, ax = plt.subplots()
    ax.plot(t_values, ub, label='Boundary Condition')
    ax.plot(t_values, utest, label='Predicted Condition at X = 0')
    ax.plot(t_values, utest2, label='Predicted Condition at X = 0.50')
    ax.plot(t_values, utest3, label='Predicted Condition at X = 0.80')
    ax.plot(t_values, utest4, label='Predicted Condition at X = 1.0')
    ax.legend(loc='upper right')
    fig.savefig(os.path.join('experiments', 'periodic_T' + '{:.3f}'.format(settings.T) + '_epoch' + str(epoch) + '.png'))
    plt.close()
   
    return



###############################################################################

def visualize_model(model, settings):
    '''
    Parameters
    ----------
    model : Pytorch neural network object
    settings : Class object containing various settings
    
    Returns
    -------
    3D Plot of PDE solution    
    '''  
    
    # Subsample x-axis and t-axis into 100 by 100 data points
    dx = (settings.xR - settings.xL)/99
    x_values = np.reshape(np.arange(settings.xL, settings.xR+dx, dx), (-1,1))    
    x_matrix = np.repeat(x_values, repeats=100, axis=1)
    
    dt = settings.T/99
    t_values = np.reshape(np.arange(0, settings.T+dt, dt), (1,-1))
    t_matrix = np.repeat(t_values, repeats=100, axis=0)

    # Evaluate data using network
    net_data = torch.from_numpy(np.concatenate([x_matrix.reshape((-1,1)), t_matrix.reshape((-1,1))], axis=-1)).float().to(device)
    with torch.inference_mode():
        predicted = model(net_data)
        predicted = predicted.cpu().detach().numpy().reshape([-1,100])
                
    # Plot results
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_matrix, t_matrix, predicted, 
                    edgecolor='royalblue', 
                    lw=0.5, rstride=8, 
                    cstride=8,
                    alpha=0.3)
    ax.set_xlabel('$X$', fontsize=10, rotation=0)
    ax.set_ylabel('$t$', fontsize=10, rotation=0)
    ax.set_zlabel('$u(x,t)$', fontsize=10, rotation=0)
    plt.title('3D Plot with $h(t)=0.1\sin(2\pi t)$')
    
    
    # #ax.legend(loc='upper right')
    plt.show()
    # fig.savefig(os.path.join('experiments', 'periodic_T' + '{:.3f}'.format(settings.T) + '_3D.png'))
    # plt.close()
   
    return

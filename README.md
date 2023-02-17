# KdV Net

KdV Net is a nueral network that has been designed to solve the KdV equation with and without boundary conditions. 

# Requirements
  * Python3
  * NumPy
  * PyTorch
  * CUDA enabled device (GPU) is optional

# Repository (repo) layout
Currently, this repo has three different branches. You can acess each branch in the upper left drop down.

* **main** - this is where the final project will be located. Right now it is up to date but most likely will not be.
* **pinn** - this is the code from the paper XXXX. It is designed to use TensorFlow instead of PyTorch
* **physics_net** - this is the current working branch and will be the most up to date.

**Note:** If you are wanting to make changes, please create another branch and work on that branch to avoid overwritting work by others.

# Scripts

The repo contains the following files:

* `batch_runs.bt` - This is a Windows batch file that can be used to run multiple experiments in a row.
* `utils.py` - This is the script that contains extra helper functions
* `pinn.py` - This is the original network file.
* `pinn2.py` - This is a different BVP network.

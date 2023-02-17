@ECHO OFF 

:: This batch file is used to run multiple configurations for the pinn project.

cd ./pinn

ECHO Starting runs from batch file... 

ECHO ==========================
ECHO ========================== >> outputlog.txt

ECHO Run 1
ECHO Run 1 >> outputlog.txt

ECHO ==========================
ECHO ========================== >> outputlog.txt

set Tf=2
ECHO Final time set at T=%Tf%
ECHO %Tf% >> outputlog.txt

python pinn.py --hidden_sizes=100,100,100 --optimizer=adam --lr=1e-3 --epochs=500000 --activation=tanh --xL=0 --xR=1 --T=%Tf% --N0=100 --Nb=1000 --Ni=20000 --initial_condition_func=sin --boundary_scaling=0.1 --train=False --save=True --visualize=True >> outputlog.txt


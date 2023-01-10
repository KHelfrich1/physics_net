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
ECHO %Tf%
ECHO %Tf% >> outputlog.txt

python pinn.py --hidden_sizes=100,100,100 --optimizer=adam --lr=1e-3 --epochs=500000 --xL=0 --xR=1.0 --T=%Tf% --N0=100 --Nb=1000 --Ni=20000 --initial_condition_func=sin --boundary_scaling=0.01 >> outputlog.txt

ECHO ==========================
ECHO ========================== >> outputlog.txt

ECHO Run 2
ECHO Run 2 >> outputlog.txt

ECHO ==========================
ECHO ========================== >> outputlog.txt

set Tf=4
ECHO %Tf%
ECHO %Tf% >> outputlog.txt

python pinn.py --hidden_sizes=100,100,100 --optimizer=adam --lr=1e-3 --epochs=500000 --xL=0 --xR=1.0 --T=%Tf% --N0=100 --Nb=1000 --Ni=20000 --initial_condition_func=sin --boundary_scaling=0.01 >> outputlog.txt

ECHO ==========================
ECHO ========================== >> outputlog.txt

ECHO Run 3
ECHO Run 3 >> outputlog.txt

ECHO ==========================
ECHO ========================== >> outputlog.txt

set Tf=8
ECHO %Tf%
ECHO %Tf% >> outputlog.txt

python pinn.py --hidden_sizes=100,100,100 --optimizer=adam --lr=1e-3 --epochs=500000 --xL=0 --xR=1.0 --T=%Tf% --N0=100 --Nb=1000 --Ni=20000 --initial_condition_func=sin --boundary_scaling=0.01 >> outputlog.txt

ECHO ==========================
ECHO ========================== >> outputlog.txt

ECHO Run 4
ECHO Run 4 >> outputlog.txt

ECHO ==========================
ECHO ========================== >> outputlog.txt

set Tf=10
ECHO %Tf%
ECHO %Tf% >> outputlog.txt

python pinn.py --hidden_sizes=100,100,100 --optimizer=adam --lr=1e-3 --epochs=500000 --xL=0 --xR=1.0 --T=%Tf% --N0=100 --Nb=1000 --Ni=20000 --initial_condition_func=sin --boundary_scaling=0.01 >> outputlog.txt

ECHO ==========================
ECHO ========================== >> outputlog.txt

ECHO Run 5
ECHO Run 5 >> outputlog.txt

ECHO ==========================
ECHO ========================== >> outputlog.txt

set Tf=100
ECHO %Tf%
ECHO %Tf% >> outputlog.txt

python pinn.py --hidden_sizes=100,100,100 --optimizer=adam --lr=1e-3 --epochs=500000 --xL=0 --xR=1.0 --T=%Tf% --N0=100 --Nb=1000 --Ni=20000 --initial_condition_func=sin --boundary_scaling=0.01 >> outputlog.txt
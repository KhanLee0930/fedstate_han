import numpy as np
import math
import sys
import os
current_dir = os.path.dirname(__file__)
functions_dir = os.path.join(current_dir, 'Functions')
sys.path.append(functions_dir)
import pickle
import random
import itertools
import fl.fedsate.src.StarlinkDataForFL.StarlinkData as Generator
import Distributor as Scheduler
import matplotlib.pyplot as plt

N = 10 # Number of Users
M = 5
delta = 1 # 1Gb
K = 10
np.random.seed(42)
D = np.random.randint(0, 100, size=(N, M))
data = Generator.Model_Generator(N, M, D, delta, K)

Psi = data['Psi']
Phi = data['Phi']
C_access = data['C_access']
C_e = data['C_e']
N = data['N']
S = data['S']
S_set = data['S_set']
K = data['K']
Client_Set = data['Client_Set']
Para = [N, M, K, S] 
E_involved = data['E_involved']

C_access = np.ones((N,S))*C_access
C_e = [C_e for i in range(len(Phi[0]))]
print('Scheduler Starts')
f_trajectory, D_optimal, D_sm = Scheduler.DataBalance(D, Psi, Phi, delta, C_access, 
                                                      C_e, Para, S_set, E_involved, Client_Set)

for i in range(5):
    plt.plot(f_trajectory[i]) 
plt.legend(['Total Time', 'Collect', 'Move', 'Train','Gradient Length'])
plt.show() 
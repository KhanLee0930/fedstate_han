import numpy as np
import math
import sys
import os
import torch
current_dir = os.path.dirname(__file__)
functions_dir = os.path.join(current_dir, 'Functions')
sys.path.append(functions_dir)
import pickle
import random
import itertools
import StarlinkDataForFL.StarlinkData as Generator
# from StarlinkDataForFL.Distributor import DataBalance
from actor import DataBalance
import matplotlib.pyplot as plt
from critic import Critic
from utils.utils import parse_args, run_pre_experiments, Net, load_datasets, set_parameters, get_parameters, train, test

if __name__ == '__main__':
    args = parse_args()
    save_dir = os.path.join("data", f"{args.p}-{args.E}-{args.B}-{args.alpha}")
    print(f"Save Directory: {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initial_model = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    params = get_parameters(initial_model)

    # Lightweight Experiments
    configs, zeta, num_classes = run_pre_experiments(args.p, args.B, args.alpha, args.num_samples, initial_model, criterion, save_dir, device)

    # Critic
    critic = Critic(configs, zeta, args.E, args.epsilon, args.B, args.p, device)

    N = 10 # Number of Users
    M = num_classes # Number of Label Types
    delta = 1 # 1Gb
    K = 8 # Number of Clients
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

    f_trajectory, D_optimal, D_sm = DataBalance(D, Psi, Phi, delta, C_access, 
                                                        C_e, Para, S_set, E_involved, Client_Set, critic)

    print(D, D.sum())
    print(D_sm, D_sm.sum())
    print(D_optimal, D_optimal.sum())
    for i in range(4):
        plt.plot(f_trajectory[i]) 
    plt.legend(['Total Time', 'Collect', 'Move', 'Train'])
    plt.show() 
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:59:43 2024

@author: Hao WU
"""

import numpy as np
import math
import random
import networkx as nx
# import gurobipy as gp
import itertools
import copy
import time

''' =========== Problem Input =============
D: A N*M matrix whose element D[n][m] represents the sample number of label m in user n
Psi: A N*S matrix whose element Psi[n][s] = 1 is satellite s is visiable by user n
delta: The size of a data sample
C_access: A N*S matrix whose element C_access[n][s] is the link capacity between satellite s and user n
F: A list of computing power of K clients
Phi: Phi[s*K+k][e] = 1 if the path from s to k passes e
C_e: ISL capacity
=========== Variables =====================
y
x
t_move
t_collect
t_train
D_sm: A two-layer list with D[s][m] being the number of samples 
D_optimal: optimal data distribution 

=========== Intermediate Parameters =======
N: Number of Users
M: Number of Label Types
K: Number of Clients
S: Number of Satellites Involved in Data Move
'''


def DataBalance(D, Psi, Phi, delta, C_access, C_e, Para, S_set, E_involved, Client_Set):
    # ============ Initialization ============== 
    [N, M, K, S] = Para
    
    # Pre Processing : Phi --> Phi_List
    Phi_List = [[] for e in E_involved]
    Phi_List2 = [[] for k1 in Phi]
    for e in range(len(E_involved)):
        for k1 in range(len(Phi)): 
            if Phi[k1][e] == 1:
                Phi_List[e].append(k1)
                Phi_List2[k1].append(e)
    # ------------- x ---------------
    x = []
    for n, s in itertools.product(range(N), range(S)):
        Visible_Sat = np.sum(Psi[n,:])
        x_m = []
        for m in range(M):
            if Psi[n][s] == 1:
                x_ns_m = D[n][m]/Visible_Sat
            else:
                x_ns_m = 0
            x_m.append(x_ns_m)
        x.append(x_m)
    
    # ----------- D_s,m -----------------
    D_sm = np.zeros((S,M))
    for s, m in itertools.product(range(S), range(M)):
        for n in range(N):
            D_sm[s][m] += x[n*S+s][m]
    
    # ------------D_optimal--------------
    D_optimal = np.zeros((K, M))
    # copy.deepcopy(D_sm[:K,:M])
    # print(D_optimal) 
    
    # ------------- y -------------------
    y = []
    for k1, k2 in itertools.product(range(S), repeat=2):
        y_m = []
        for m in range(M):
            y_m.append(0)
        y.append(y_m)      
    
    # ----------- t_move -----------------
    t_move = 0
    
    # ----------- t_collect --------------
    t_collect = 0
    for n in range(N):
        t_collect_n = 0
        for s in range(S):
            t_collect_n += sum(x[n*S+s][m] for m in range(M)) * delta / C_access[n][s]
        # print(t_collect_n)
        t_collect = max(t_collect, t_collect_n)
    
    # ------------ t_train ---------------
    t_train, grad_t_train = critic(D, D_optimal)
    
    # --------Lagrangian Multiplier-------
    Lambda_collect = [1 for n in range(N)]
    Lambda_move = [1 for e in E_involved]
    mu_D_sm = [1 for s, m in itertools.product(range(S), range(M))]
    mu_D_nm = [1 for n, m in itertools.product(range(N), range(M))]
    mu_km = [1 for k, m in itertools.product(range(S), range(M))]
    
    Increment_f = 100
    f_previous = 0
    # ================ Iteration ===================
    Square_Sum = 1000
    f_trajectory = [[] for i in range(5)]
    iteration_step = 0
    # while Square_Sum > 100 and Increment_f > 0.1:
    for i in range(100):
        # 0-term
        alpha = 100/(1 + iteration_step * 0.01)
        rho = 1 + iteration_step * 0.01
        iteration_step += 1
        t_train, grad_t_train = critic(D, D_optimal)
        f = t_collect + t_move + t_train
        Increment_f = abs(f - f_previous)
        f_previous = f
        print(f'Grad_Len: {round(Square_Sum,2)}, Rate: {alpha}')
        print(f'Time Array: {[round(f,2), round(t_collect,2), round(t_move,2), round(t_train,2)]}')
        print()
        f_trajectory[0].append(round(f,2))
        f_trajectory[1].append(round(t_collect,2))
        f_trajectory[2].append(round(t_move,2))
        f_trajectory[3].append(round(t_train,2))
        f_trajectory[4].append(round(Square_Sum,2))
        
        g_collect = []
        for n in range(N):
            g_collect_n = - t_collect
            for s in range(S):
                g_collect_n += sum(x[n*S+s][m] for m in range(M)) * delta / C_access[n][s]
            g_collect.append(g_collect_n)
        
        g_move = []
        for e in range(len(E_involved)):
            g_move_e = - t_move * C_e[E_involved[e]]
            for k3 in Phi_List[e]:
                g_move_e += sum(y[k3][m] for m in range(M)) * delta
            g_move.append(g_move_e)
        
        h_D_sm = []
        for s, m in itertools.product(range(S), range(M)):
            h_D_sm_item = - D_sm[s][m]
            for n in range(N):
                 h_D_sm_item += Psi[n][s] * x[n*S+s][m]
            h_D_sm.append(h_D_sm_item)
        
        h_D_nm = []
        for n, m in itertools.product(range(N), range(M)):
            h_D_nm_item = - D[n][m]
            for s in range(S):
                 h_D_nm_item += x[n*S+s][m]
            h_D_nm.append(h_D_sm_item)
        
        h_km = []
        for k, m in itertools.product(range(S), range(M)):
            if k in Client_Set:
                h_km_item = D_sm[k][m] - D_optimal[Client_Set.index(k)][m]
            else:
                h_km_item = D_sm[k][m]
            h_km_item += sum(-y[k*S+k1][m] + y[k1*S+k][m] for k1 in range(S))
            h_km.append(h_km_item)   
        
        
        # print(g_move)
        # print(y)
        # time.sleep(10)
        '''
        Update x: grad_x = f'(x) + sum Lambda g'(x) + sum mu h'(x) 
        + rho (sum max(0,g(x))*g'(x) + sum h(x)*h'(x))
        '''
        # Update x
        Square_Sum = 0
        x_update = []
        for n, s in itertools.product(range(N), range(S)):
            for m in range(M):
                f_grad = 0 
                g_grad = Lambda_collect[n] * delta / C_access[n][s] 
                h_grad = mu_D_sm[s*M+m] * Psi[n][s] + mu_D_nm[n*M + m]
                Pg_grad = rho*max(0, g_collect[n])* delta / C_access[n][s] 
                Ph_grad = rho* (h_D_sm[s*M + m] * Psi[n][s] + h_D_nm[n*M + m])
                grad_sum = f_grad + g_grad + h_grad + Pg_grad + Ph_grad
                x_update.append(grad_sum)
                Square_Sum += grad_sum**2
        # Update y
        y_update = []
        for k1, k2 in itertools.product(range(S), repeat=2):
            for m in range(M):
                f_grad = 0 
                g_grad = sum(Lambda_move[e] * delta for e in Phi_List2[k1*S+k2]) 
                h_grad = mu_km[k1*M+m] * -1 + mu_km[k2*M+m] * 1
                Pg_grad = rho*sum( max(0, g_move[e]) * delta for e in Phi_List2[k1*S+k2]) 
                Ph_grad = rho* (h_km[k1*M + m] * -1 + h_km[k2*M + m] * 1)
                grad_sum = f_grad + g_grad + h_grad + Pg_grad + Ph_grad
                y_update.append(grad_sum)
                Square_Sum += grad_sum**2
        
        # Update D_sm
        D_sm_update = []
        for s, m in itertools.product(range(S), range(M)):
            f_grad = 0 
            g_grad = 0
            h_grad = mu_D_sm[s*M + m] * -1 + mu_km[s*M + m]
            Pg_grad = 0
            Ph_grad = rho* (h_D_sm[s*M + m] * -1 + h_km[s*M + m])
            grad_sum = f_grad + g_grad + h_grad + Pg_grad + Ph_grad
            D_sm_update.append(grad_sum)
            Square_Sum += grad_sum**2
        
        # Update D_optimal
        D_optimal_update = []
        for k, m in itertools.product(range(K), range(M)):
            f_grad = grad_t_train[k][m]
            g_grad = 0
            h_grad = mu_km[Client_Set[k]*M+m] * -1 
            Pg_grad = 0
            Ph_grad = rho* h_km[Client_Set[k]*M+m]* -1
            grad_sum = f_grad + g_grad + h_grad + Pg_grad + Ph_grad
            D_optimal_update.append(grad_sum)
            Square_Sum += grad_sum**2
            
        # Update t_collect
        f_grad = 1 
        g_grad = sum(Lambda_collect)*-1
        Pg_grad = rho*sum(max(0, g_collect[n])*-1 for n in range(N))
        t_collect_update = f_grad + g_grad + Pg_grad
        Square_Sum += t_collect_update**2
        
        # Update t_move
        f_grad = 1
        g_grad = sum(Lambda_move[e] * -C_e[E_involved[e]] for e in range(len(E_involved)))
        Pg_grad = rho*sum(max(0, g_move[e])*-C_e[E_involved[e]] for e in range(len(E_involved)))
        t_move_update = f_grad + g_grad + Pg_grad
        Square_Sum += t_move_update**2
        
        
        Square_Sum = Square_Sum**0.5
        # print(Square_Sum)
        
        for n, s in itertools.product(range(N), range(S)):
            for m in range(M):
                x[n*S+s][m] = max(0, x[n*S+s][m] - alpha * x_update.pop(0)/Square_Sum)
        
        for k1, k2 in itertools.product(range(S), repeat=2):
            for m in range(M):
                y[k1*S+k2][m] = max(0, y[k1*S+k2][m] - alpha * y_update.pop(0)/Square_Sum)   
                
        for s, m in itertools.product(range(S), range(M)):
            D_sm[s][m] = max(0, D_sm[s][m] - alpha * D_sm_update.pop(0)/Square_Sum)
            
        for k, m in itertools.product(range(K), range(M)):
            D_optimal[k][m] = max(0, D_optimal[k][m] - alpha * D_optimal_update.pop(0)/Square_Sum)
        
        t_collect = max(t_collect - alpha * t_collect_update/Square_Sum, 0)
        t_move = max(t_move - alpha * t_move_update/Square_Sum, 0)
        # print(t_move_update/Square_Sum, t_collect_update/Square_Sum)
        
        # Update lambda
        for n in range(N):
            Lambda_collect[n] = max(0, Lambda_collect[n] + rho * g_collect[n])
            # print(['Gradient!!!!!!!!', Lambda_collect[n], g_collect[n], t_collect])
        for e in range(len(E_involved)):
            Lambda_move[e] = max(0, Lambda_move[e] + rho * g_move[e])
            
        # Update mu
        for s, m in itertools.product(range(S), range(M)):
            mu_D_sm[s*M+m] += rho * h_D_sm[s*M+m]
        for n, m in itertools.product(range(N), range(M)):
            mu_D_nm[n*M+m] += rho * h_D_nm[n*M+m]
        for k, m in itertools.product(range(K), range(M)):
            mu_km[k*M+m] += rho * h_km[k*M+m]
            
    return f_trajectory, D_optimal, D_sm



def critic(D, D_optimal):
    """
    Calculate D_even, the square sum of differences, and the difference matrix.
    
    Parameters:
        D_optimal (numpy.ndarray): Input matrix.
        
    Returns:
        tuple: (square_sum, difference_matrix)
            - square_sum (float): Square sum of the differences between D_even and D_optimal.
            - difference_matrix (numpy.ndarray): Matrix of D_optimal - D_even.
    
    Return the estimated training time and the gradient
    """
    average_value = np.mean(D)
    D_even = np.full_like(D_optimal, average_value)
    
    # Calculate the difference matrix
    difference_matrix = (D_optimal - D_even)
    
    # Calculate the square sum of the differences
    square_sum = 0.5*np.sum(np.square(difference_matrix))
    
    #  difference_matrix = difference_matrix / square_sum
    #  print(difference_matrix)
    return square_sum, difference_matrix



# # Main Test
# '''
# 4 Users, 4 Lables, 5 Collector, 5 Clients

# '''

# N = 4
# M = 4
# S = 5
# K = 5
# Para = [N, M, K, S]

# D = np.array([[1,2,3,4],
#               [5,6,7,8],
#               [9,10,11,12],
#               [13,14,15,16]],dtype=float)

# Psi = np.array([[1,1,0,0,0],[0,1,1,0,0],[0,0,1,1,0],[0,0,0,1,1]])

# # 1. Establish the network with directed links between nodes
# G = nx.DiGraph()  # Create a directed graph

# # Add nodes (optional, as edges will implicitly add nodes)
# G.add_nodes_from([0, 1, 2, 3, 4])

# # Add directed links between the nodes
# edges = [
#     (0, 4, 0), (4, 0, 1),  # Links between nodes 0 and 4
#     (1, 4, 2), (4, 1, 3),  # Links between nodes 1 and 4
#     (2, 4, 4), (4, 2, 5),  # Links between nodes 2 and 4
#     (3, 4, 6), (4, 3, 7)   # Links between nodes 3 and 4
# ]

# # Add edges with their corresponding index as an attribute
# for u, v, idx in edges:
#     G.add_edge(u, v, index=idx)

# Phi = np.zeros((S*K,8))
# for s in range(S):
#     for k in range(K):
#         path = nx.shortest_path(G, source=s, target=k)
#         for i in range(len(path) - 1):
#             Phi[s*K+k][G[path[i]][path[i+1]]['index']] = 1

# delta = 1

# C_access = np.ones((N,S))*1

# C_e = [0.5 for i in range(8)]

# F = [1 for i in range(K)] 

# outcome1, outcome2 = DataBalance(D, Psi, Phi, delta, C_access, C_e, Para, )

# for i in range(4):
#     plt.plot(outcome1[i]) 
# plt.legend(['Total Time', 'Collect', 'Move', 'Train'])
# plt.show() 
    
import numpy as np
import math
import random
import networkx as nx
# import gurobipy as gp
import itertools
import copy
import time
import torch
from critic import TrainingCritic

# torch.autograd.set_detect_anomaly(True)

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

class FedSateDataBalancer:

    def __init__(self, D, Psi, Phi, delta, C_access, C_e, Para, S_set, E_involved, Client_Set, num_epochs, device='cuda'):
        # ============ Initialization ==============
        [N, M, K, S] = Para

        # Ensure inputs are PyTorch tensors
        D = torch.tensor(D, dtype=torch.float32).to(device)          # N x M
        Psi = torch.tensor(Psi, dtype=torch.float32).to(device)      # N x S
        Phi = torch.tensor(Phi, dtype=torch.float32).to(device)      # (S*K) x E
        C_access = torch.tensor(C_access, dtype=torch.float32).to(device)  # N x S
        C_e = torch.tensor(C_e, dtype=torch.float32).to(device)
        S_set = torch.tensor(S_set).to(device)
        E_involved = torch.tensor(E_involved).to(device)
        Client_Set = torch.tensor(Client_Set).to(device)

        # # Pre Processing: Phi -> e2k and k2e
        # e2k = []
        # for e in range(Phi.shape[1]):  # Number of edges E
        #     k1_list = torch.where(Phi[:, e] == 1)[0].tolist()
        #     e2k.append(k1_list)

        # k2e = []
        # for k1 in range(Phi.shape[0]):  # Number of paths S*K
        #     e_list = torch.where(Phi[k1, :] == 1)[0].tolist()
        #     k2e.append(e_list)

        Phi = Phi.view(S, S, -1)  # S x S x E
        Phi = Phi[:, :K, :] # S x K x E


        self.N = N
        self.M = M
        self.K = K
        self.S = S
        self.D = D
        self.Psi = Psi
        self.Phi = Phi
        self.delta = delta
        self.C_access = C_access
        self.C_e = C_e
        self.S_set = S_set
        self.E_involved = E_involved
        self.Client_Set = Client_Set
        self.num_epochs = num_epochs
        self.device = device

    # ----------- D_s,m -----------------
    # Sum over all users to get D_sm
    def uploaded_data(self, x):
        softmax_x = torch.softmax(x, dim=1)
        scaled_x = softmax_x * self.D.unsqueeze(1).expand(-1, self.S, -1)  # N x S x M
        D_sm = torch.sum(scaled_x, dim=0)
        data_upload_violation = torch.sum(scaled_x, dim=1) - self.D

        return D_sm, data_upload_violation # (S, M)
    
    def moved_data(self, D_sm, y_k, y_s):
        D_km = torch.zeros(self.K, self.M, dtype=torch.float32)
        y = torch.cat((y_k, y_s), dim=0)
        softmax_y = torch.softmax(y, dim=1)
        y = softmax_y * D_sm.unsqueeze(1).expand(-1, self.K, -1) # S x K x M

        # Sum over sources to get data moved to each destination
        moved_to_data = y.sum(dim=0)  # K x M

        # Sum over destinations to get data moved from each source
        moved_from_data = y.sum(dim=1)  # S x M

        # **Replace in-place operation with out-of-place**
        D_sm_new = D_sm - moved_from_data  # S x M

        # Extract previous data for the first K destinations
        previous_data = D_sm_new[:self.K, :]  # K x M

        # Combine moved data with previous data
        D_km = moved_to_data + previous_data  # K x M

        # Calculate data moving violation for sources >= K
        data_moving_violation = D_sm_new[self.K:, :]  # (S-K) x M

        return D_km, data_moving_violation
    
    # ----------- t_collect --------------
    def collecting_time(self, x):
        softmax_x = torch.softmax(x, dim=1)
        scaled_x = softmax_x * self.D.unsqueeze(1).expand(-1, self.S, -1)  # N x S x M

        # Sum over labels to get total data per user and satellite
        x_sum_over_m = torch.sum(scaled_x, dim=2)

        # Compute t_collect_n for each user
        t_collect_n = torch.sum(x_sum_over_m * self.delta / self.C_access, dim=1)

        # Get the maximum t_collect across all users
        t_collect = torch.max(t_collect_n)

        return t_collect

    # ------------ t_move ---------------
    def moving_time(self, D_sm, y_k, y_s):
        y = torch.cat((y_k, y_s), dim=0)  # S x K x M
        softmax_y = torch.softmax(y, dim=1)
        y = softmax_y * D_sm.unsqueeze(1).expand(-1, self.K, -1) 

        # Calculate the link load with y (S, K, M) and Phi (S, K, E), there should only be movement from S to K. The link load should be (E,)
        total_flow = y.sum(dim=2)

        # Compute link load
        # For each link e, link_load[e] = sum over s, k of total_flow[s][k] * Phi[s][k][e]
        # print(total_flow.shape, Phi.shape)

        link_load = torch.einsum('sk, ske -> e', total_flow, self.Phi)
        # Alternative implementation, same output
        #link_load = torch.sum(total_flow.unsqueeze(-1) * Phi, dim=(0, 1))
        # Compute the max moving time
        t_move = torch.max(link_load * self.delta / self.C_e)

        return t_move

    def balance_critic(self, critic):

        N = self.N
        M = self.M
        K = self.K
        S = self.S
        D = self.D
        Psi = self.Psi
        num_epochs = self.num_epochs
        device = self.device


        # ------------- x (to optimize) ---------------
        # Compute the number of visible satellites per user
        Visible_Sat = torch.sum(Psi, dim=1, keepdim=True)  # N x 1
        # Avoid division by zero
        Visible_Sat_nonzero = Visible_Sat.clone()
        Visible_Sat_nonzero[Visible_Sat_nonzero == 0] = 1

        # Compute D divided by the number of visible satellites
        D_div_Visible = D / Visible_Sat_nonzero  # N x M

        # Expand D_div_Visible to match the dimensions
        D_div_Visible_expanded = D_div_Visible.unsqueeze(1).expand(-1, S, -1)  # N x S x M

        # Compute x using Psi as a mask
        x = D_div_Visible_expanded * Psi.unsqueeze(2)  # N x S x M

        x = torch.nn.Parameter(x.to(device))

        
        # ------------- y (to optimize) -------------------
        # Initialize y as zeros
        y_k = torch.nn.Parameter(torch.ones(S-K, K, M, dtype=torch.float32, device=device))
        y_s = torch.nn.Parameter(torch.ones(K, K, M, dtype=torch.float32, device=device))    # Wrap it as a Parameter

        # ------------ D_optimal --------------
        # # Compute the total number of samples for each label m
        # total_D_m = torch.sum(D_sm, dim=0)  # M

        # # Distribute samples equally among K satellites
        # # For integer counts, use floor division and handle remainders
        # samples_per_satellite = total_D_m // K  # M
        # remainder = total_D_m % K  # M

        # # Initialize D_optimal with samples_per_satellite
        # D_optimal = samples_per_satellite.unsqueeze(0).expand(K, -1).clone()  # K x M

        # # Distribute the remainder among the first 'remainder' satellites
        # for i in range(K):
        #     D_optimal[i] += (remainder > i).int()

        # ------------ t_train ---------------
        # t_train, grad_t_train = critic(D, D_optimal)

        optimizer = torch.optim.Adam([x, y_k, y_s], lr=1e-3)

        rho = 10

        # ['Total Time', 'Collect', 'Move', 'Train']
        f_trajectory = []

        for i in range(num_epochs):
            optimizer.zero_grad()

            D_sm, data_upload_violation = self.uploaded_data(x)
            D_km, data_moving_violation = self.moved_data(D_sm, y_k, y_s)
            # upload_violation = torch.abs(data_upload_violation.sum())
            # moving_violation = torch.abs(data_moving_violation.sum())

            t_collect = self.collecting_time(x)
            t_move = self.moving_time(D_sm, y_k, y_s)

            # training time proxy from theory
            _, t_train = critic.time_estimation(D_km)

            # Compute the loss
            loss = t_train + t_collect + t_move
            total_time = loss

            # Save the trajectory
            f_trajectory.append([total_time.item(), t_collect.item(), t_move.item()])

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.mul_(Psi.unsqueeze(2))  # N x S x M
            if i % 10000 == 0:
                print(f'Epoch {i}, Total Time: {total_time.item()}, Collect: {t_collect.item()}, Move: {t_move.item()}, Train: {t_train.item()}')
        D_sm, _ = self.uploaded_data(x)
        D_optimal, _ = self.moved_data(D_sm, y_k, y_s)
        return f_trajectory, D_optimal, D_sm
    
    def balance_iid(self):

        N = self.N
        M = self.M
        K = self.K
        S = self.S
        D = self.D
        Psi = self.Psi
        num_epochs = self.num_epochs
        device = self.device


        # ------------- x (to optimize) ---------------
        # Compute the number of visible satellites per user
        Visible_Sat = torch.sum(Psi, dim=1, keepdim=True)  # N x 1
        # Avoid division by zero
        Visible_Sat_nonzero = Visible_Sat.clone()
        Visible_Sat_nonzero[Visible_Sat_nonzero == 0] = 1

        # Compute D divided by the number of visible satellites
        D_div_Visible = D / Visible_Sat_nonzero  # N x M

        # Expand D_div_Visible to match the dimensions
        D_div_Visible_expanded = D_div_Visible.unsqueeze(1).expand(-1, S, -1)  # N x S x M

        # Compute x using Psi as a mask
        x = D_div_Visible_expanded * Psi.unsqueeze(2)  # N x S x M

        x = torch.nn.Parameter(x.to(device))

        
        # ------------- y (to optimize) -------------------
        # Initialize y as zeros
        y_k = torch.nn.Parameter(torch.ones(S-K, K, M, dtype=torch.float32, device=device))
        y_s = torch.nn.Parameter(torch.ones(K, K, M, dtype=torch.float32, device=device))    # Wrap it as a Parameter

        # ------------ D_optimal --------------
        # # Compute the total number of samples for each label m
        total_D_m = torch.sum(D, dim=0)  # M

        # Distribute samples equally among K satellites
        # For integer counts, use floor division and handle remainders
        samples_per_satellite = total_D_m // K  # M
        remainder = total_D_m % K  # M

        # Initialize D_optimal with samples_per_satellite
        D_optimal = samples_per_satellite.unsqueeze(0).expand(K, -1).clone()  # K x M

        # Distribute the remainder among the first 'remainder' satellites
        for i in range(K):
            D_optimal[i] += (remainder > i).int()

        # ------------ t_train ---------------
        # t_train, grad_t_train = critic(D, D_optimal)

        optimizer = torch.optim.Adam([x, y_k, y_s], lr=1e-3)

        rho = 10

        # ['Total Time', 'Collect', 'Move']
        f_trajectory = []

        for i in range(num_epochs):
            optimizer.zero_grad()

            D_sm, data_upload_violation = self.uploaded_data(x)
            D_km, data_moving_violation = self.moved_data(D_sm, y_k, y_s)
            # upload_violation = torch.abs(data_upload_violation.sum())
            # moving_violation = torch.abs(data_moving_violation.sum())

            t_collect = self.collecting_time(x)
            t_move = self.moving_time(D_sm, y_k, y_s)
            iid_panelty = torch.sum(torch.abs(D_km - D_optimal))

            # Compute the loss
            loss = t_collect + t_move + rho * iid_panelty
            total_time = t_collect + t_move

            # Save the trajectory
            f_trajectory.append([total_time.item(), t_collect.item(), t_move.item()])

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.mul_(Psi.unsqueeze(2))  # N x S x M
            if i % 10000 == 0:
                print(f'Epoch {i}, Total Uploading & Transmission Time: {total_time.item()}, Collect: {t_collect.item()}, Move: {t_move.item()}')
        D_sm, _ = self.uploaded_data(x)
        D_optimal, _ = self.moved_data(D_sm, y_k, y_s)
        return f_trajectory, D_optimal, D_sm

    def balance_non_iid(self):
        
        N = self.N
        M = self.M
        K = self.K
        S = self.S
        D = self.D
        Psi = self.Psi
        num_epochs = self.num_epochs
        device = self.device


        # ------------- x (to optimize) ---------------
        # Compute the number of visible satellites per user
        Visible_Sat = torch.sum(Psi, dim=1, keepdim=True)  # N x 1
        # Avoid division by zero
        Visible_Sat_nonzero = Visible_Sat.clone()
        Visible_Sat_nonzero[Visible_Sat_nonzero == 0] = 1

        # Compute D divided by the number of visible satellites
        D_div_Visible = D / Visible_Sat_nonzero  # N x M

        # Expand D_div_Visible to match the dimensions
        D_div_Visible_expanded = D_div_Visible.unsqueeze(1).expand(-1, S, -1)  # N x S x M

        # Compute x using Psi as a mask
        x = D_div_Visible_expanded * Psi.unsqueeze(2)  # N x S x M

        x = torch.nn.Parameter(x.to(device))

        
        # ------------- y (to optimize) -------------------
        # Initialize y as zeros
        y_k = torch.nn.Parameter(torch.ones(S-K, K, M, dtype=torch.float32, device=device))
        y_s = torch.nn.Parameter(torch.ones(K, K, M, dtype=torch.float32, device=device))    # Wrap it as a Parameter

        # ------------ D_optimal --------------
        # # Compute the total number of samples for each label m
        # total_D_m = torch.sum(D_sm, dim=0)  # M

        # # Distribute samples equally among K satellites
        # # For integer counts, use floor division and handle remainders
        # samples_per_satellite = total_D_m // K  # M
        # remainder = total_D_m % K  # M

        # # Initialize D_optimal with samples_per_satellite
        # D_optimal = samples_per_satellite.unsqueeze(0).expand(K, -1).clone()  # K x M

        # # Distribute the remainder among the first 'remainder' satellites
        # for i in range(K):
        #     D_optimal[i] += (remainder > i).int()

        # ------------ t_train ---------------
        # t_train, grad_t_train = critic(D, D_optimal)

        optimizer = torch.optim.Adam([x, y_k, y_s], lr=1e-3)

        rho = 10

        # ['Total Time', 'Collect', 'Move']
        f_trajectory = []

        for i in range(num_epochs):
            optimizer.zero_grad()

            D_sm, data_upload_violation = self.uploaded_data(x)
            D_km, data_moving_violation = self.moved_data(D_sm, y_k, y_s)
            # upload_violation = torch.abs(data_upload_violation.sum())
            # moving_violation = torch.abs(data_moving_violation.sum())

            t_collect = self.collecting_time(x)
            t_move = self.moving_time(D_sm, y_k, y_s)

            # Compute the loss
            loss = t_collect + t_move
            total_time = t_collect + t_move

            # Save the trajectory
            f_trajectory.append([total_time.item(), t_collect.item(), t_move.item()])

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.mul_(Psi.unsqueeze(2))  # N x S x M
            if i % 10000 == 0:
                print(f'Epoch {i}, Total Uploading & Transmission Time: {total_time.item()}, Collect: {t_collect.item()}, Move: {t_move.item()}')
        D_sm, _ = self.uploaded_data(x)
        D_optimal, _ = self.moved_data(D_sm, y_k, y_s)
        return f_trajectory, D_optimal, D_sm

    def balance_vanilla_uploading(self):

        N = self.N
        M = self.M
        K = self.K
        S = self.S
        D = self.D
        Psi = self.Psi
        num_epochs = self.num_epochs
        device = self.device


        # ------------- x ---------------
        # Compute the number of visible satellites per user
        Visible_Sat = torch.sum(Psi, dim=1, keepdim=True)  # N x 1
        # Avoid division by zero
        Visible_Sat_nonzero = Visible_Sat.clone()
        Visible_Sat_nonzero[Visible_Sat_nonzero == 0] = 1

        # Compute D divided by the number of visible satellites
        D_div_Visible = D / Visible_Sat_nonzero  # N x M

        # Expand D_div_Visible to match the dimensions
        D_div_Visible_expanded = D_div_Visible.unsqueeze(1).expand(-1, S, -1)  # N x S x M

        # Compute x using Psi as a mask
        x = D_div_Visible_expanded * Psi.unsqueeze(2)  # N x S x M

        x = x.to(device)
        
        # ------------- y (to optimize) -------------------
        # Initialize y as zeros
        y_k = torch.nn.Parameter(torch.ones(S-K, K, M, dtype=torch.float32, device=device))
        y_s = torch.nn.Parameter(torch.ones(K, K, M, dtype=torch.float32, device=device))    # Wrap it as a Parameter

        # ------------ D_optimal --------------
        # # Compute the total number of samples for each label m
        # total_D_m = torch.sum(D_sm, dim=0)  # M

        # # Distribute samples equally among K satellites
        # # For integer counts, use floor division and handle remainders
        # samples_per_satellite = total_D_m // K  # M
        # remainder = total_D_m % K  # M

        # # Initialize D_optimal with samples_per_satellite
        # D_optimal = samples_per_satellite.unsqueeze(0).expand(K, -1).clone()  # K x M

        # # Distribute the remainder among the first 'remainder' satellites
        # for i in range(K):
        #     D_optimal[i] += (remainder > i).int()

        # ------------ t_train ---------------
        # t_train, grad_t_train = critic(D, D_optimal)

        optimizer = torch.optim.Adam([y_k, y_s], lr=1e-3)

        rho = 10

        # ['Total Time', 'Collect', 'Move']
        f_trajectory = []

        for i in range(num_epochs):
            optimizer.zero_grad()

            D_sm, data_upload_violation = self.uploaded_data(x)
            D_km, data_moving_violation = self.moved_data(D_sm, y_k, y_s)
            # upload_violation = torch.abs(data_upload_violation.sum())
            # moving_violation = torch.abs(data_moving_violation.sum())

            t_collect = self.collecting_time(x)
            t_move = self.moving_time(D_sm, y_k, y_s)

            # Compute the loss
            loss = t_move
            total_time = loss

            # Save the trajectory
            f_trajectory.append([total_time.item(), t_collect.item(), t_move.item()])

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.mul_(Psi.unsqueeze(2))  # N x S x M
            if i % 10000 == 0:
                print(f'Epoch {i}, Collect: {t_collect.item()}, Move: {t_move.item()}')
        D_sm, _ = self.uploaded_data(x)
        D_optimal, _ = self.moved_data(D_sm, y_k, y_s)
        return f_trajectory, D_optimal, D_sm

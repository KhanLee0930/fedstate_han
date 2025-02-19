import numpy as np
import math
import random
import networkx as nx
# import gurobipy as gp
import itertools
import copy
import time
import torch

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

    def __init__(self, D, Psi, Phi, delta, C_access, C_e, N, M, K, S, S_set, E_involved, num_epochs, model_size, device='cuda'):
        # ============ Initialization ==============
        # Ensure inputs are PyTorch tensors
        D = torch.tensor(D, dtype=torch.float32).to(device)          # N x M
        Psi = torch.tensor(Psi, dtype=torch.float32).to(device)      # N x S
        Phi = torch.tensor(Phi, dtype=torch.float32).to(device)      # (S*S) x E
        C_access = torch.tensor(C_access, dtype=torch.float32).to(device)  # N x S
        E = Phi.size()[-1]
        C_e_values = torch.arange(0.1, C_e + 1e-6, 0.1, device=device)
        C_e = C_e_values[torch.randint(0, len(C_e_values), (E,))]
        S_set = torch.tensor(S_set).to(device)
        E_involved = torch.tensor(E_involved).to(device)

        Phi = Phi.view(S, S, -1)  # S x S x E
        random_indices = torch.randperm(S)[:K].to(device)  # Randomly select K indices to be client
        communication_load = torch.sum(Phi[random_indices][:, random_indices, :], dim=(0, 1))
        self.communication_time = torch.max(communication_load * model_size / C_e)
        Phi = Phi[:, random_indices, :] # S x K x E


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

    def uploading_strategy(self, method, critic, balance_params, B, E):
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
        # Uploading strategy for each of the N users to each of the S satellites, for each class M
        # The initialization is such that each user uniformly uploads their dataset to all visible satellites
        x = torch.nn.Parameter(x.to(device))

        # ------------- y (to optimize) -------------------
        # Initialize y as zeros
        # Satellities are split into two, the S-K group that are not involved in training, and the K group that are involved in training
        # y_k defines the movement of data from the S-K satellites to the K satellites
        y_k = torch.nn.Parameter(torch.ones(S-K, K, M, dtype=torch.float32, device=device))
        # y_s defines the movement of data within the K satellites
        y_s = torch.nn.Parameter(torch.ones(K, K, M, dtype=torch.float32, device=device))    # Wrap it as a Parameter

        # ------------- batch_size, E (to optimize) -------------------
        batch_size = torch.nn.Parameter(torch.tensor([B], dtype=torch.float32, device=device))
        E = torch.nn.Parameter(torch.tensor([E], dtype=torch.float32, device=device))
        # If we are additionally balancing the batch size and number of local epochs, simply pass it to the optimizer
        if balance_params:
            optimizer = torch.optim.Adam([x, y_k, y_s, batch_size, E], lr=1e-2)
        else:
            optimizer = torch.optim.Adam([x, y_k, y_s], lr=1e-2)

        # ------------ D_optimal --------------
        # # Compute the total number of samples for each label m
        total_D_m = torch.sum(D, dim=0)  # M
        # Distribute samples equally among K satellites
        # For integer counts, use floor division and handle remainders
        samples_per_satellite = total_D_m // K  # M
        # Initialize D_optimal with samples_per_satellite
        D_optimal = samples_per_satellite.unsqueeze(0).expand(K, -1).clone()  # K x M

        # ------------ t_train ---------------
        # t_train, grad_t_train = critic(D, D_optimal)
        # ['Total Time', 'Collect', 'Move', 'Train']
        f_trajectory = []

        for i in range(num_epochs):
            optimizer.zero_grad()

            D_sm, data_upload_violation = self.uploaded_data(x)
            D_km, data_moving_violation = self.moved_data(D_sm, y_k, y_s)
            #upload_violation = torch.abs(data_upload_violation.sum())
            #moving_violation = torch.abs(data_moving_violation.sum())
            t_collect = self.collecting_time(x)
            t_move = self.moving_time(D_sm, y_k, y_s)
            _, t_train, Lambda, lr = critic.time_estimation(D_km, batch_size, E)
            total_time = t_collect + t_move + t_train
            iid_penalty = torch.sum(torch.abs(D_km - D_optimal))

            # Balance method minimizes the total time take
            if method == "Balance":
                loss = total_time
            # IID method minimizes the the time taken to upload and move the data, with the constraint of IID
            elif method == "IID":
                loss = t_collect + t_move + iid_penalty * 10
            # Non-IID just seeks to get the data to the clients as fast as possible
            elif method == "Non-IID":
                loss = t_collect + t_move

            # Save the trajectory
            f_trajectory.append([total_time.detach().item(), t_collect.detach().item(), t_move.detach().item(), t_train.detach().item()])

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.mul_(Psi.unsqueeze(2))  # N x S x M
                E.clamp_(1, 1000000)
                batch_size.clamp(1, 1000000)
            if i % 10000 == 0:
                print(f"Batch Size: {batch_size.item()}, Local Epochs: {E.item()}")
                print(f'Epoch {i}, Total Time: {total_time.item()}, Collect: {t_collect.item()}, Move: {t_move.item()}, Train: {t_train.item()}, Gradient Diversity: {Lambda}, Learning Rate: {lr.detach().item()}')
                print(f"Distribution: {D_km}")

        D_sm, _ = self.uploaded_data(x)
        D_optimal, _ = self.moved_data(D_sm, y_k, y_s)
        return f_trajectory, D_optimal, D_sm, int(E.cpu().item()), 2 ** int(math.log2(batch_size.cpu().item()))
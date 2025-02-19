# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:29:21 2024

@author: Hao Wu
"""

import numpy as np
import math
import sys
import os
current_dir = os.path.dirname(__file__)
functions_dir = os.path.join(current_dir, 'Functions')
sys.path.append(functions_dir)
import pickle
import itertools
import SPOnGrid as SPG
from utils.utils import obtain_initial_distribution
import torch
import random

def split_number(total, num_parts):
    """Splits `total` into `num_parts` non-uniformly while summing to `total`, allowing replacement."""
    # For example, split_number(10, 3) might return something like [3, 4, 3], where 10 data points is split into 3 parts
    if total == 0:
        return [0] * num_parts  # If total is 0, return zeros

    # Pick num_parts - 1 cut points, allowing duplicates
    cuts = np.random.choice(range(total + 1), num_parts - 1, replace=True)
    cuts.sort()  # Ensure order

    # Compute partitions
    return [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, num_parts - 1)] + [total - cuts[-1]]


def LinkDelay(Lat1, Lon1, Lat2, Lon2, Altittude):
    Lat1 = math.radians(Lat1)
    Lon1 = math.radians(Lon1)
    Lat2 = math.radians(Lat2)
    Lon2 = math.radians(Lon2)
    cos_angle = math.cos(Lat1)*math.cos(Lat2)*math.cos(Lon1-Lon2)+math.sin(Lat1)*math.sin(Lat2)
    Distance1 = (6371+Altittude)*(2*(1-cos_angle))**0.5
    '''
    Distance1 and Distance2 returns the same value
    '''
    # Distance2 = (6371+Latittude) * math.acos(math.cos(Lat1) * math.cos(Lat2) * math.cos(Lon1 - Lon2)
    #                                          + math.sin(Lat1)* math.sin(Lat2))/3 * 10**-5
    return Distance1


def Inter_Shell_Graph(OrbitNum1, SatNum1, LatMat1, LatLimit):
    '''
    Input:
    OrbitNum1: Number of Orbit
    SatNum1: Number of Satellites per orbit
    Ignore LatMat1 and LatLimit for Stralink


    Output:
    G: Adjacent Matrix  | 1 (connected)  999 (disconnected)
    EMap: Index Matrix  | EMap[a][b] represents the link index from a to b
    E: Link Index List  | The n-th element represents the end nodes of the n-th link
    '''
    G = np.zeros((OrbitNum1 * SatNum1,
                  OrbitNum1 * SatNum1),dtype = int) + 999 # Adjecent Matrix
    EMap = np.zeros((OrbitNum1 * SatNum1,
                     OrbitNum1 * SatNum1),dtype = int) + 999
    E = []
    edge_index = 0
    for orb in range(0,OrbitNum1):
        for sat in range(0,SatNum1):
            sat_index = orb * SatNum1 + sat
            #----------------Inter-Orbit------------------------
            if abs(LatMat1[sat_index]) < LatLimit:
                neibor1 = ((orb-1)%OrbitNum1)*SatNum1 + sat
                if abs(LatMat1[neibor1]) < LatLimit:
                    G[sat_index,neibor1] = 1
                    EMap[sat_index,neibor1] = edge_index
                    E.append([int(sat_index),int(neibor1)])
                    edge_index += 1
                neibor2 = ((orb+1)%OrbitNum1)*SatNum1 + sat
                if abs(LatMat1[neibor2]) < LatLimit:
                    G[sat_index,neibor2] = 1
                    EMap[sat_index,neibor2] = edge_index
                    E.append([int(sat_index),int(neibor2)])
                    edge_index += 1
            #-------------Intra-Orbit-------------------------
            if sat == 0:
                sat_neibor3 = SatNum1 - 1
                sat_neibor4 = sat + 1
            elif sat == SatNum1 - 1:
                sat_neibor3 = sat - 1
                sat_neibor4 = 0
            else:
                sat_neibor3 = sat - 1
                sat_neibor4 = sat + 1
            neibor3 = (orb*SatNum1 + sat_neibor3)
            neibor4 = (orb*SatNum1 + sat_neibor4)
            G[sat_index,neibor3] = 1
            G[sat_index,neibor4] = 1
            EMap[sat_index,neibor3] = edge_index
            EMap[sat_index,neibor4] = edge_index + 1
            E.append([int(sat_index),int(neibor3)])
            E.append([int(sat_index),int(neibor4)])
            edge_index += 2
    return G, EMap, E


def Model_Generator(N, alpha, save_dir):
    # Required Data : Psi, Phi, delta, C_access, C_e, N, S, K
    OrbitNum1 = 72
    SatNum1   = 22
    LatMat1 = [0 for i in range(OrbitNum1 * SatNum1)]
    LatLimit = 90
    C_access = 0.5 # Gbps
    C_e = 1 # Gbps
    # ========= Load Positions of Satellites  ========================
    fileName = open("/home/svu/e1143336/fedstate/src/StarlinkDataForFL/StarLink_OneShell_1200Slots_Step5s.pkl", "rb")
    altitude = 540 # 540 km

    # Shuffle before any processing to prevent ordering of the satellites, required since we simply sample the last p satellites as the clients
    data = pickle.load(fileName)
    Lat_list = data['Lat']
    Lon_list = data['Lon']

    print(['Data Readout Finished!'])

    # 50 users, 10 regions * 5 countries/region
    PositionUser = [
        # North America
        [40.7128, -74.0060],   # New York, USA
        [34.0522, -118.2437],  # Los Angeles, USA
        [45.5017, -73.5673],   # Montreal, Canada
        [49.2827, -123.1207],  # Vancouver, Canada
        [19.4326, -99.1332],    # Mexico City, Mexico
        # South America
        [-23.5505, -46.6333],  # São Paulo, Brazil
        [-34.6037, -58.3816],  # Buenos Aires, Argentina
        [-12.0464, -77.0428],  # Lima, Peru
        [-3.1190, -60.0217],  # Manaus, Brazil
        [-33.4489, -70.6693],  # Santiago, Chile
        # Western & Northern Europe
        [51.5074, -0.1278],  # London, UK
        [52.5200, 13.4050],  # Berlin, Germany
        [59.3293, 18.0686],  # Stockholm, Sweden
        [64.1355, -21.8954],  # Reykjavik, Iceland
        [40.4168, -3.7038],  # Madrid, Spain
        # Eastern & Southern Europe
        [55.7558, 37.6176],  # Moscow, Russia
        [41.9028, 12.4964],  # Rome, Italy
        [45.8150, 15.9819],  # Zagreb, Croatia
        [50.0755, 14.4378],  # Prague, Czech Republic
        [37.9838, 23.7275],  # Athens, Greece
        # Northern & Western Africa
        [30.0444, 31.2357],  # Cairo, Egypt
        [6.5244, 3.3792],  # Lagos, Nigeria
        [14.6928, -17.4467],  # Dakar, Senegal
        [36.8065, 10.1815],  # Tunis, Tunisia
        [5.6037, -0.1870], # Accra
        # Eastern & Southern Africa
        [-1.2921, 36.8219],  # Nairobi, Kenya
        [-26.2041, 28.0473],  # Johannesburg, South Africa
        [-17.8249, 31.0530],  # Harare, Zimbabwe
        [-15.3875, 28.3228], # Lusaka, Zambia
        [-25.9692, 32.5732], # Maputo
        # Middle East
        [25.276987, 55.296249],  # Dubai, UAE
        [35.6892, 51.3890],  # Tehran, Iran
        [31.7683, 35.2137],  # Jerusalem, Israel
        [24.7136, 46.6753],  # Riyadh, Saudi Arabia
        [33.5138, 36.2765],  # Damascus, Syria
        # East & Southeast Asia
        [35.6895, 139.6917],  # Tokyo, Japan
        [31.2304, 121.4737],  # Shanghai, China
        [37.5665, 126.9780],  # Seoul, South Korea
        [1.3521, 103.8198],  # Singapore
        [13.7563, 100.5018],  # Bangkok, Thailand
        # South & Central Asia
        [19.0760, 72.8777],  # Mumbai, India
        [27.7172, 85.3240],  # Kathmandu, Nepal
        [6.9271, 79.8612],  # Colombo, Sri Lanka
        [41.3275, 69.2672],  # Tashkent, Uzbekistan
        [39.9334, 32.8597],  # Ankara, Turkey
        # Oceania
        [-33.8688, 151.2093],  # Sydney, Australia
        [-37.8136, 144.9631],  # Melbourne, Australia
        [-17.7333, 168.3273],  # Port Vila, Vanuatu
        [-41.2865, 174.7762],  # Wellington, New Zealand
        [-20.2008, 57.5074]  # Mauritius
    ]
    num_regions = 10
    user_per_region = 5
    N_per_region = N // num_regions
    # Select N_per_region for each region
    PositionUser = [PositionUser[i] for i in range(len(PositionUser)) if i % user_per_region < N_per_region]
    print("Position of Users selected, should be evenly chosen from each region")
    print(PositionUser)

    # Initial distribution is non-IID across the 10 regions
    initial_distribution = obtain_initial_distribution(num_regions, alpha, save_dir) # (num_regions, num_classes)
    # User distribution is obtained by splitting the dataset in each region
    user_distribution = [[split_number(x, N_per_region) for x in region] for region in initial_distribution] # (num_regions, num_classes, N_per_region)
    user_distribution = torch.tensor(user_distribution)
    user_distribution = torch.flatten(torch.transpose(user_distribution, 2, 1), end_dim=1) # (num_regions * N_per_region, num_classes)
    print("Initial distribution that is generated non-IID across the 10 regions")
    print(initial_distribution)
    print("Dividing the data from each region to each user in that region")
    print(user_distribution)

    S_n = [[] for n in range(N)]
    S_set = set()
    for n, s in itertools.product(range(N), range(OrbitNum1*SatNum1)):
        [Lat1, Lon1] = PositionUser[n]
        Lat2 = Lat_list[s]
        Lon2 = Lon_list[s]
        if LinkDelay(Lat1, Lon1, Lat2, Lon2, altitude) < 928:
            S_n[n].append(s)
            S_set.add(s)
    S_set = list(S_set)

    S = len(S_set)
    Psi = np.zeros((N,S))
    for n, s_set in enumerate(S_n):
        for s in s_set:
            Psi[n][S_set.index(s)] = 1

    # ================== Generate Connectivity ==========================
    G, EMap, E = Inter_Shell_Graph(OrbitNum1, SatNum1, LatMat1, LatLimit)

    Phi = np.zeros((S*S, len(E)), dtype = bool)
    for k1, k2 in itertools.product(range(S), repeat=2):
        n1 = S_set[k1]
        n2 = S_set[k2]
        Path = SPG.SPOnGrid(n1, n2, [], [],  'ISL', 1)
        for n1 in range(len(Path[0])-1):
            edge_index = EMap[Path[0][n1]][Path[0][n1+1]]
            Phi[k1*S+k2][edge_index] = 1
            # if edge_index not in E_involved:
            #     E_involved.append(edge_index)
    E_involved = [x for x in range(len(E))]
    data = {'Psi':Psi,
            'Phi':Phi,
            'C_access':C_access,
            'C_e':C_e,
            'N':N,
            'S':S,
            'S_set':S_set,
            'E_involved':E_involved,
            'G': (G != 999).astype(int)}
    return data, user_distribution
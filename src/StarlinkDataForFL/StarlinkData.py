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


def Model_Generator(N, M, D, delta, K):
    # Required Data : Psi, Phi, delta, C_access, C_e, N, S, K
    OrbitNum1 = 72
    SatNum1   = 22
    LatMat1 = [0 for i in range(OrbitNum1 * SatNum1)]
    LatLimit = 90
    C_access = 0.5 # Gbps
    C_e = 1 # Gbps
    # ========= Load Positions of Satellites  ========================
    fileName = open("/home/svu/e1143336/fedstate/src/StarlinkDataForFL/StarLink_OneShell_1200Slots_Step5s.pkl", "rb")
    orb_period = 1200
    altitude = 540 # 540 km
    Lat_list = []
    Lon_list = []
    for epoch in range(orb_period):
        data = pickle.load(fileName)
        Lat_list.append(data['Lat'])
        Lon_list.append(data['Lon'])
    print(['Data Readout Finished!'])

    PositionUser = [
        [60.3913, 5.3221],     # Bergen, Norway
        [-25.2744, 133.7751],  # Australia
        [-34.6037, -58.3816],  # Buenos Aires, Argentina
        [35.6895, 139.6917],   # Tokyo, Japan
        [55.7558, 37.6176],    # Moscow, Russia
        [51.5074, -0.1278],    # London, UK
        [48.8566, 2.3522],     # Paris, France
        [40.7128, -74.0060],   # New York, USA
        [34.0522, -118.2437],  # Los Angeles, USA
        [-23.5505, -46.6333],  # São Paulo, Brazil
        [-33.8688, 151.2093],  # Sydney, Australia
        [-1.2921, 36.8219],    # Nairobi, Kenya
        [30.0444, 31.2357],    # Cairo, Egypt
        [20.5937, 78.9629],    # India (Central)
        [56.1304, -106.3468],  # Canada (Central)
        [35.6762, 139.6503],   # Tokyo (alternate point), Japan
        [39.9042, 116.4074],   # Beijing, China
        [-22.9068, -43.1729],  # Rio de Janeiro, Brazil
        [37.7749, -122.4194],  # San Francisco, USA
        [52.5200, 13.4050],    # Berlin, Germany
        [59.3293, 18.0686],    # Stockholm, Sweden
        [-26.2041, 28.0473],   # Johannesburg, South Africa
        [31.2304, 121.4737],   # Shanghai, China
        [35.6895, 139.6917],   # Tokyo (another alternate point), Japan
        [41.3851, 2.1734],     # Barcelona, Spain
        [38.7223, -9.1393],    # Lisbon, Portugal
        [-3.3896, 36.6822],    # Arusha, Tanzania
        [-15.7942, -47.8822],  # Brasília, Brazil
        [-20.2008, 57.5074],   # Mauritius
        [40.4168, -3.7038],    # Madrid, Spain
        [1.3521, 103.8198],    # Singapore
        [51.1657, 10.4515],    # Germany (Central)
        [45.8150, 15.9819],    # Zagreb, Croatia
        [-13.1631, -72.5450],  # Machu Picchu, Peru
        [19.8968, -155.5828],  # Hawaii, USA
        [-33.4489, -70.6693],  # Santiago, Chile
        [45.5017, -73.5673],   # Montreal, Canada
        [49.2827, -123.1207],  # Vancouver, Canada
        [35.0116, 135.7681],   # Kyoto, Japan
        [-17.8249, 31.0530],   # Harare, Zimbabwe
        [43.6532, -79.3832],   # Toronto, Canada
        [51.0447, -114.0719],  # Calgary, Canada
        [-12.0464, -77.0428],  # Lima, Peru
        [48.4284, -123.3656],  # Victoria, Canada
        [49.8951, -97.1384],   # Winnipeg, Canada
        [52.1332, -106.6700],   # Saskatoon, Canada
        [34.3416, 108.9398],   # XI'AN
        [-37.8136, 144.9631],   # Melbourne, Australia
        [1.3521, 103.8198],     # Singapore
        [37.5665, 126.9780]     # Seoul, South Korea (capital city)
    ]

    epoch = 0

    S_n = [[] for n in range(N)]
    S_set = set()
    for n, s in itertools.product(range(N), range(OrbitNum1*SatNum1)):
        [Lat1, Lon1] = PositionUser[n]
        Lat2 = Lat_list[epoch][s]
        Lon2 = Lon_list[epoch][s]
        if LinkDelay(Lat1, Lon1, Lat2, Lon2, altitude) < 928:
            S_n[n].append(s)
            S_set.add(s)
    S_set = list(S_set)

    S = len(S_set)
    Psi = np.zeros((N,S))
    for n, s_set in enumerate(S_n):
        for s in s_set:
            Psi[n][S_set.index(s)] = 1

    if K > S or K > N:
        print("Invalid Input!!!!!")
        return False
    else:
        # Select K client's from near the K largest dataset
        D_rows = np.sum(D,1)
        indices = np.argpartition(D_rows, -K)[-K:]  # Get the indices of the top K elements
        # Sort these indices based on the actual values in descending order
        sorted_indices = indices[np.argsort(-D_rows[indices])]

    Client_Set = []
    for i1 in sorted_indices:
        Client_Set.append(S_set.index(S_n[i1][0]))
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



    file_write = open("/home/svu/e1143336/fedstate/src/StarlinkDataForFL/SatelliteRelatedData.pkl", "wb")
    data = {'Psi':Psi,
            'Phi':Phi,
            'C_access':C_access,
            'C_e':C_e,
            'N':N,
            'S':S,
            'S_set':S_set,
            'K':K,
            'E_involved':E_involved,
            'Client_Set':Client_Set}
    pickle.dump(data, file_write)
    # print(data)
    return data

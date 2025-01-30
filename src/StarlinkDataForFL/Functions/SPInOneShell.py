# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:12:00 2024

@author: Hao Wu
"""
import random
import math

def _idx2coord(idx, x_sz, y_sz, offset):
    idx_ = idx - offset
    return (idx_ // y_sz, idx_ % y_sz)

def _coord2idx(coord, x_sz, y_sz, offset):
    return coord[0] * y_sz + coord[1] + offset

def _direction(x1, x2, NumOribt):
    if (x2 > x1 and NumOribt/2 > (x2 - x1)) or (x2 < x1 and NumOribt/2 < (x1 - x2)):
        return 1
    else:
        return -1
        

def SPInOneShell(src_idx, dst_idx, idx_offset, NumOribt, NumSat, path_num):
    x1, y1 = _idx2coord(src_idx, NumOribt, NumSat, idx_offset)
    x2, y2 = _idx2coord(dst_idx, NumOribt, NumSat, idx_offset)
    
    # Determine the forward direction
    x_direction = _direction(x1, x2, NumOribt)
    y_direction = _direction(y1, y2, NumSat)
    
    x_dis = min(abs(x2-x1), NumOribt - abs(x2-x1))
    y_dis = min(abs(y2-y1), NumSat - abs(y2-y1))
    
    if path_num > math.comb(x_dis + y_dis, x_dis):
        return False
    
    KeepGenerating = True
    path_list = []
    while KeepGenerating:     
        z = random.sample(range(x_dis + y_dis), x_dis)
        path = [src_idx]
        [x,y] = [x1, y1]
        for k in range(x_dis + y_dis):
            if k in z:
                [x, y] = [x+x_direction, y]
            else:
                [x, y] = [x, y+y_direction] 
            path.append(_coord2idx([x, y], NumOribt, NumSat, idx_offset))
        path_list.append(path)
        if len(path_list) >= path_num:
            KeepGenerating = False
        print(len(path_list))
    return path_list


#======== Test ========
NumOribt = 72
NumSat = 22
idx_offset = 0
[x1,y1] = [71,9]
[x2,y2] = [22,20]
src_idx = _coord2idx([x1, y1], NumOribt, NumSat, idx_offset)
dst_idx = _coord2idx([x2, y2], NumOribt, NumSat, idx_offset)
path_list =SPInOneShell(src_idx, dst_idx, idx_offset, NumOribt, NumSat, 5)
print(len(path_list[0]))
print(path_list)
            
        
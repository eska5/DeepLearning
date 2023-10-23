import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

gamma = 0.9
delta_max = 4

file_name = 'map_small.txt'
# file_name = 'map_easy.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape
actions_count = 4

num_of_steps_max = int(5*(num_of_rows + num_of_columns))

def transition_probability(x ,y, x_prim, y_prim ,a):
    if a == 1:
        if y_prim == y + 1:
            return 0.4
        if x_prim == x - 1:
            return 0.2
        if y_prim == y - 1:
            return 0.04
        if x_prim == x + 1:
            return 0.36
    if a == 2:
        if y_prim == y + 1:
            return 0.2
        if x_prim == x - 1:
            return 0.4
        if y_prim == y - 1:
            return 0.36
        if x_prim == x + 1:
            return 0.04
    if a == 3:
        if y_prim == y + 1:
            return 0.04
        if x_prim == x - 1:
            return 0.2
        if y_prim == y - 1:
            return 0.4
        if x_prim == x + 1:
            return 0.36
    if a==4:
        if y_prim == y + 1:
            return 0.2
        if x_prim == x - 1:
            return 0.04
        if y_prim == y - 1:
            return 0.36
        if x_prim == x + 1:
            return 0.4

def get_all_moves(x, y):
    moves = []
    moves.append((x , y + 1))
    moves.append((x - 1, y ))
    moves.append((x, y - 1))
    moves.append((x + 1, y))

    return moves

def strategia_wartosci():
    delta = delta_max
    V = np.zeros([num_of_rows, num_of_columns], dtype=float)

    while delta >= delta_max:
        V_temp = np.copy(V)
        delta = 0
        
        for x in range(num_of_rows):
             for y in range(num_of_columns-1):
                max_rw = -np.inf
                for a in range(1,5):
                    values_sum = 0 
                    _, reward = sf.environment([x, y], a, reward_map)

                    for x_prim, y_prim in get_all_moves(x, y):
                        if y_prim == y +1  and y_prim >= num_of_columns:
                            isPossible=-0.4
                        elif y_prim == y -1 and y_prim < 0:
                            isPossible=-0.4
                        elif x_prim == x +1 and x_prim >= num_of_rows:
                            isPossible=-0.4
                        elif x_prim == x -1 and x_prim < 0:
                            isPossible=-0.4
                        else:
                            isPossible=V[x_prim, y_prim]
                        
                        values_sum += (transition_probability(x,y, x_prim, y_prim, a) * isPossible)

                    reward += (gamma * values_sum)
                    Q[x,y,a-1] = reward

                    if reward > max_rw:
                        max_rw = reward

                V[x,y] = max_rw
                delta = max(delta, abs(V[x,y] - V_temp[x, y]))
    return Q

Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float) 
Q = strategia_wartosci()
result = sf.sailor_test(reward_map, Q, 1000)
sf.draw(reward_map,Q)

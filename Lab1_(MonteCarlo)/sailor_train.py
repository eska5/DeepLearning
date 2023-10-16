import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf
 
number_of_episodes = 200
number_of_iterations = 100
gamma = 0.87

file_name = 'map_small.txt'
# file_name = 'map_easy.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

maps = ['map_small.txt', 'map_easy.txt', 'map_big.txt', 'map_spiral.txt']



### MONTE CARLO ITERACJA STRATEGII

def monte_carlo_iteracja_strategii(reward_map, Q):
    for _ in range(number_of_iterations):
        for state_init_X in range(num_of_rows):
            for state_init_Y in range(num_of_columns):
                for init_action in range(1, 5):
                    state = np.zeros([2], dtype=int)
                    sum_of_rewards = np.zeros([number_of_episodes], dtype=float)    
                    for episode in range(number_of_episodes):
                        state[0] = state_init_X
                        state[1] = state_init_Y
                        the_end = False
                        nr_pos = 0
                        gamma_temp = gamma
                        while not the_end:
                            nr_pos = nr_pos + 1
                            if nr_pos == 1:
                                action = init_action 
                            else:
                                action = 1 + np.argmax(Q[state[0], state[1], :])

                            state, reward = sf.environment(state, action, reward_map)

                            if (nr_pos == num_of_steps_max) or (state[1] >= num_of_columns-1):
                                the_end = True

                            sum_of_rewards[episode] =  sum_of_rewards[episode] + (gamma_temp * reward)
                            gamma_temp = gamma_temp * gamma
                    Q[state_init_X, state_init_Y, init_action - 1] += (np.sum(sum_of_rewards)/number_of_episodes)    

    return Q


### MONTE CARLO ITERACJA WARTOŚCI
    
def monte_carlo_iteracja_wartości(reward_map, Q):
    for _ in range(number_of_iterations):
        alpha_counter = 0
        for state_init_X in range(num_of_rows):
            for state_init_Y in range(num_of_columns):
                for init_action in range(1, 5):

                    alpha_counter += 1
                    state = np.zeros([2],dtype=int)      
                    state[0] = state_init_X
                    state[1] = state_init_Y
                    the_end = False
                    nr_pos = 0
                    alpha_n = 1/(alpha_counter)
                    sum_of_rewards = 0
                    while the_end == False:
                        nr_pos = nr_pos + 1         
                        if nr_pos == 1:
                            action = init_action
                        else: 
                            action = 1 + np.argmax(Q[state[0],state[1], :])
                        state_next, reward  = sf.environment(state, action, reward_map)

                        sum_of_rewards += np.power(gamma,nr_pos-1)*reward                
                        state = state_next      
                    
                        if (nr_pos == num_of_steps_max) or (state[1] >= num_of_columns-1):
                            the_end = True

                    Q[state_init_X, state_init_Y, init_action-1] = (1 - alpha_n) * Q[state_init_X, state_init_Y, init_action-1] + (alpha_n * sum_of_rewards)
                                
    return Q



### MAIN
for method in range(2):
    for map in maps:
        reward_map = sf.load_data(map)
        num_of_rows, num_of_columns = reward_map.shape
        num_of_steps_max = int(5*(num_of_rows + num_of_columns))
        Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float) 

        if method == 0:
            monte_carlo_iteracja_strategii(reward_map, Q)
        else:
            monte_carlo_iteracja_wartości(reward_map, Q)
            

        sf.sailor_test(reward_map,Q,1000)
        sf.draw(reward_map,Q)
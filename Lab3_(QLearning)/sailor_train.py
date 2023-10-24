import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

# file_name = 'map_small.txt'
#file_name = 'map_easy.txt'
file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(5*(num_of_rows + num_of_columns))  
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  

def pick_action(epsilon):
    if epsilon > np.random.uniform(0,1):
        return np.random.randint(0, 4)
    else:
        return -1


### PARAMETERS ###

epsilon_values = np.arange(0, 1.1, 0.1)
epoch_values = [2000, 5000, 10000]
file_names = ['map_small.txt', 'map_easy.txt', 'map_big.txt', 'map_spiral.txt']


num_of_episodes = 10000
lr = 0.01
base_epsilon = 0.65
checkpoint = 50
gamma = 0.9
epsilon = base_epsilon
sum = 0

for file_name in file_names:
    reward_map = sf.load_data(file_name)
    num_of_rows, num_of_columns = reward_map.shape
    num_of_steps_max = int(5*(num_of_rows + num_of_columns))  
    for num_of_episodes in epoch_values:
        results = []
        epsilonList = []
        for epsilon in epsilon_values:
            Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float) 
            for episode in range(num_of_episodes):
                if episode% checkpoint == 0:
                    print(f"episode :{episode+1}, avg for {checkpoint} last values: {sum / checkpoint}")
                    sum = 0
                # Początkowy state
                state = (np.random.randint(0, num_of_rows), 0)
                running = True
                step = 0
                while running:
                    step += 1
                    act_pick = pick_action(epsilon)
                    action = act_pick if act_pick != -1 else np.argmax(Q[state])
                    new_st, r = sf.environment(state, action + 1, reward_map)
                    new_st = tuple(new_st)
                    sum += r
                    if r > 0:
                        running = False
                    else:
                        difference = np.max(Q[new_st + (action,)] - Q[state + (action,)])
                        Q[state +(action,)] += lr * (r + gamma * difference)
                    state = new_st

            results.append(sf.sailor_test(reward_map, Q, 1000))
            epsilonList.append(epsilon)
        plt.scatter(epsilonList, results, alpha=0.5)
        plt.xlabel('Epsilon')
        plt.ylabel('result')
        plt.title(f'Epsilon vs. result, Epochs: {num_of_episodes}')
        plt.show()

            # sf.sailor_test(reward_map, Q, 1000)
            # sf.draw(reward_map,Q)

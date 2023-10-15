import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 700000                   # number of training epizodes (multi-stage processes) 
number_of_iterations = 100
gamma = 0.90                                 # discount factor


# file_name = 'map_small.txt'
# file_name = 'map_easy.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained usability table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

# miejsce na algorytm uczenia - modelem jest tablica Q
# (symulację epizodu można wziąć z funkcji sailor_test())
# ............................

### BASE TEST FUNC ###
def sailor_test(reward_map, Q, num_of_episodes):
    num_of_rows, num_of_columns = reward_map.shape
    num_of_steps_max = int(5*(num_of_rows + num_of_columns)) 
       # maximum number of steps in an episode
    sum_of_rewards = np.zeros([num_of_episodes], dtype=float)

    for episode in range(num_of_episodes):
        state = np.zeros([2],dtype=int)                            # initial state here [1 1] but rather random due to exploration
        state[0] = np.random.randint(0,num_of_rows)
        the_end = False
        nr_pos = 0
        while the_end == False:
            nr_pos = nr_pos + 1                            # move number
        
            # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom): 
            action = 1 + np.argmax(Q[state[0],state[1], :])
            state_next, reward  = sf.environment(state, action, reward_map)
            state = state_next       # going to the next state
        
            # end of episode if maximum number of steps is reached or last column
            # is reached
            if (nr_pos == num_of_steps_max) | (state[1] >= num_of_columns-1):
                the_end = True
        
            sum_of_rewards[episode] += reward
    print('test-'+str(num_of_episodes)+' mean sum of rewards = ' + str(np.mean(sum_of_rewards)))

###










### MONTE CARLO ITERACJA STRATEGII

def monte_carlo_iteracja_strategii(reward_map, Q, num_of_episodes, num_of_iterations, gammma):
    num_of_rows, num_of_columns = reward_map.shape
    num_of_steps_max = int(5 * (num_of_rows + num_of_columns))

    # LOSOWE POLICY
    policy = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)

    for _ in range(num_of_iterations):
        for episode in range(num_of_episodes):
            Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)
            # TWORZENIE LISTY z 3 parametrami
            returns = [[[ [] for _ in range(4)] for _ in range(num_of_columns)] for _ in range(num_of_rows)]

            state = np.zeros([2], dtype=int)  # initial state here [1 1] but rather random due to exploration
            state = 0, np.random.randint(0, num_of_rows)
            the_end = False
            nr_pos = 0
            episode = []

            while not the_end:
                nr_pos += 1  # move number

                # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom):
                action = 1 + np.argmax(policy[state[0],state[1], :])
                state_next, reward = sf.environment(state, action, reward_map)
                episode.append((state, action, state_next, reward))
                state = state_next  # going to the next state

                # end of episode if maximum number of steps is reached or last column is reached
                if nr_pos == num_of_steps_max or state[1] >= num_of_columns - 1:
                    the_end = True

            G = 0
            for t in range(0, len(episode)-1, 1):
                s, a, ns, r = episode[t]
                G = gamma * G + r
                print(a-1)
                print(returns[ns[0]][ns[1]][a-1])
                returns[ns[0]][ns[1]][a-1].append(G)
                Q[ns[0], ns[1], a-1] = np.mean(returns[ns[0]][ns[1]][a-1])


            # Update policy
            # print(Q)
            # print(policy)
            new_policy = Q
            # print(new_policy)
            for i in range(num_of_rows):
                for j in range(num_of_columns):
                    for z in range(4):
                        if new_policy[i][j][z] > policy[i][j][z] or new_policy[i][j][z] < policy[i][j][z] and new_policy[i][j][z] != 0:
                            policy[i][j][z] = new_policy[i][j][z]
            print(policy)
            print("---")

    return policy



### MONTE CARLO ITERACJA WARTOŚCI

def monte_carlo_iteracja_wartości(reward_map, Q, iterations,epochs, gammma):
    num_of_rows, num_of_columns = reward_map.shape
    num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
    array_1 = np.arange(0, num_of_rows, 1) 
    array_2 = np.arange(0, num_of_columns, 1) 
    array_3 = np.arange(1, 5, 1) 
    state_action_pairs = np.array(np.meshgrid(array_1, array_2, array_3)).T.reshape(-1, 3)

    for epoch in range(epochs):
        alpha_counter = 0
        for x,y,a in state_action_pairs:
            alpha_counter += 1
            state = np.zeros([2],dtype=int)                            # initial state here [1 1] but rather random due to exploration
            state[0] = x
            state[1] = y
            the_end = False
            nr_pos = 0
            alpha_n = 1/(alpha_counter)  # Adjusting alpha based on episode number
            sum_of_rewards = 0
            while the_end == False:
                nr_pos = nr_pos + 1                            # move number
                # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom):
                if nr_pos == 1:
                    action = a
                else: 
                    action = 1 + np.argmax(Q[state[0],state[1], :])
                state_next, reward  = sf.environment(state, action, reward_map)

                # Update Q-values
                sum_of_rewards += np.power(gamma,nr_pos-1)*reward                
                state = state_next       # going to the next state
            
                # end of episode if maximum number of steps is reached or last column is reached
                if (nr_pos == num_of_steps_max) or (state[1] >= num_of_columns-1):
                    the_end = True

            Q[x,y,a-1] = (1 - alpha_n) * Q[x,y,a-1] + (alpha_n * sum_of_rewards)
                                # print(Q)

        # if alpha_counter%5 == 0:
        #     print(Q)
        #     sf.draw(reward_map,Q)


            # TESTOWANIE WYNIKÓW     
            
            # if (episode % 100 == 0):
            #     print(f"episode {episode}")
            #     sf.draw(reward_map,Q)
            #     print(Q)
            #     print("---")
    return Q

# QQ = monte_carlo_iteracja_strategii(reward_map=reward_map,Q=Q,num_of_episodes=1000, num_of_iterations=100)
Q = monte_carlo_iteracja_wartości(reward_map, Q, number_of_episodes,number_of_iterations, gamma)
# Q = monte_carlo_iteracja_strategii(reward_map, Q, number_of_episodes, 5)
print("----")
print(Q)
sf.sailor_test(reward_map,Q,1000)
sf.draw(reward_map,Q)
# sf.draw(reward_map,Q)
# # sf.draw(reward_map,Q)
# sf.sailor_test(reward_map, Q, 1000)
# sf.draw(reward_map,Q)

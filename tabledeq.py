import gym
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import math
from config import config

def render_graph(total_episodes, total_steps):
    if config.graph == 1:
        plt.plot(total_episodes, total_steps, color='r')
        plt.pause(0.001)
    
def init_graph():
    if config.graph == 1:
        plt.ylabel('steps')
        plt.xlabel('episode')
        plt.plot([200] * 150, label = "goal")
        plt.pause(0.001)
        plt.legend(loc='best')

def policy(state):
    if (random.random() < config.epsilon):
        action = random.randint(0, 1)
    else:
        action = np.argmax(Q_table[discrete(state)])
    return (action)

def update_q_table(state, action, new_state, reward):
    try:
        Q_table[discrete(state)][action] = ((1 - config.learning_rate) * Q_table[discrete(state)][action]) + (config.learning_rate * (reward + (config.discount_factor * (max(Q_table[discrete(new_state)])))))
    except:
        print(f'state = {state}, discrete = {discrete(state)}')
        print(f'new_state = {new_state}, discrete = {discrete(new_state)}')

def init_q_table():
    Q_table = np.zeros(config.qt_size_array + [env.action_space.n])
    return (Q_table)

def discrete_one(value, lower_limit, upper_limit, steps):
    discrete = int((value - lower_limit) / ((upper_limit - lower_limit) / steps))
    return(discrete)

def discrete(state):
    cart_pos_float, cart_velo_float, pole_pos_float, pole_velo_float = state
    cart_pos_discrete = discrete_one(cart_pos_float, config.state_space_limits[0][0], config.state_space_limits[0][1], config.qt_size_array[0])
    cart_velo_discrete = discrete_one(cart_velo_float, config.state_space_limits[1][0], config.state_space_limits[1][1], config.qt_size_array[1])
    pole_pos_discrete = discrete_one(pole_pos_float, config.state_space_limits[2][0], config.state_space_limits[2][1], config.qt_size_array[2])
    pole_velo_discrete = discrete_one(pole_velo_float, config.state_space_limits[3][0], config.state_space_limits[3][1], config.qt_size_array[3])
    return (cart_pos_discrete, cart_velo_discrete, pole_pos_discrete, pole_velo_discrete)
    

env = gym.envs.make("CartPole-v1")
init_graph()
Q_table = init_q_table()
total_steps, total_episodes= [0], [0]
for episode in range(config.episodes):
    state = env.reset()
    steps = 0
    done = False
    if (episode % 10) == 0:
        print(f'Q_table[5, 5, 5, 5][0] = {Q_table[5, 5, 5, 5][0]}')
    while not done:
        action = policy(state)
        new_state, reward, done, _  = env.step(action)
        if config.render == 1:
            env.render()
        steps += reward
        if done == True:
            update_q_table(state, action, new_state, config.reward_values[0])
            break
        else:
            update_q_table(state, action, new_state, config.reward_values[1])
            state = new_state
    config.epsilon = config.epsilon * config.epsilon_decay
    total_episodes.append(episode)
    total_steps.append(steps)
    render_graph(total_episodes, total_steps)
if config.graph == 1:
    plt.show()
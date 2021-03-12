import gym
import random
from matplotlib import pyplot as plt
import numpy as np
# import math
from config import config
import pickle

def render_graph(avg_steps, avg_episodes, goal):
    if config.graph == 1:
        plt.plot(avg_episodes, avg_steps, label='avg_steps', color='olive')
        plt.plot(range(goal[0], goal[1]), [200] * (goal[1] - goal[0]), color = 'blue')
        goal = [goal[1], goal[1] + config.graph_frequency]
        plt.pause(0.001)
        return (goal)
    return None


def init_graph():
    if config.graph == 1:
        plt.title('cartpole_dd')
        plt.ylabel('steps')
        plt.xlabel('episode')
        

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
Q_table = init_q_table()

def learn():
    init_graph()
    total_steps, total_episodes, avg_steps, avg_episodes, goal = [0], [0], [0], [0], [0, config.graph_frequency]
    for episode in range(config.episodes):
        state = env.reset()
        steps = 0
        done = False
        while not done:
            action = policy(state)
            new_state, reward, done, _  = env.step(action)
            if (config.render == 1):
                env.render()
            steps += 1
            if done == True:
                reward = config.reward_values[0]
            update_q_table(state, action, new_state, reward)
            state = new_state

        if ((episode % config.average == 0) & (episode != 0)):
            avg_steps.append(np.mean(total_steps[-config.average :]))
            avg_episodes.append(episode)
        config.epsilon = config.epsilon * config.epsilon_decay
        if (config.epsilon < 0.1):
            config.epsilon = 0.9
        total_episodes.append(episode)
        total_steps.append(steps)

        if config.graph and (episode % config.graph_frequency == 0):
            goal = render_graph(avg_steps, avg_episodes, goal)

        if avg_steps[-1] > 400:
            with open("q_table_bis.pkl", "wb+") as f:
                pickle.dump(Q_table, f)
            return
        
    if config.graph == 1:
        plt.title('cartpole_dd')
        plt.ylabel('steps')
        plt.xlabel('episode')
        plt.legend()
        plt.show()

def play():
    config.epsilon = 0.0
    while True:
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            new_state, reward, done, _  = env.step(action)
            env.render()
            state = new_state

def read_cutie():
    with open("q_table.pkl", "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    learn()
    # Q_table = read_cutie()
    # play()

    

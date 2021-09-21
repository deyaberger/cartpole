import gym
import random
from matplotlib import pyplot as plt
import numpy as np
from config import infos
import pickle
import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', help='Choose if you want to load an existing qtable', action='store_true')
    parser.add_argument('-e', '--epochs', help='Choose the number of epochs you need for your training', type=int, default=200)
    args = parser.parse_args()
    return (args)


def render_graph(avg_steps, avg_episodes, goal):
    if infos.graph == 1:
        plt.plot(avg_episodes, avg_steps, label='avg_steps', color='olive')
        plt.plot(range(goal[0], goal[1]), [200] * (goal[1] - goal[0]), color = 'blue')
        goal = [goal[1], goal[1] + infos.graph_frequency]
        plt.pause(0.001)
        return (goal)
    return None


def init_graph():
    if infos.graph == 1:
        plt.title('cartpole_dd')
        plt.ylabel('steps')
        plt.xlabel('episode')
        

def policy(state):
    if (random.random() < infos.epsilon):
        action = random.randint(0, 1)
    else:
        action = np.argmax(Q_table[discrete(state)])
    return (action)

def update_q_table(state, action, new_state, reward):
    try:
        Q_table[discrete(state)][action] = ((1 - infos.learning_rate) * Q_table[discrete(state)][action]) + (infos.learning_rate * (reward + (infos.discount_factor * (max(Q_table[discrete(new_state)])))))
    except:
        print(f'state = {state}, discrete = {discrete(state)}')
        print(f'new_state = {new_state}, discrete = {discrete(new_state)}')

def init_q_table():
    Q_table = np.zeros(infos.qt_size_array + [env.action_space.n])
    return (Q_table)

def discrete_one(value, lower_limit, upper_limit, steps):
    discrete = int((value - lower_limit) / ((upper_limit - lower_limit) / steps))
    return(discrete)

def discrete(state):
    cart_pos_float, cart_velo_float, pole_pos_float, pole_velo_float = state
    cart_pos_discrete = discrete_one(cart_pos_float, infos.state_space_limits[0][0], infos.state_space_limits[0][1], infos.qt_size_array[0])
    cart_velo_discrete = discrete_one(cart_velo_float, infos.state_space_limits[1][0], infos.state_space_limits[1][1], infos.qt_size_array[1])
    pole_pos_discrete = discrete_one(pole_pos_float, infos.state_space_limits[2][0], infos.state_space_limits[2][1], infos.qt_size_array[2])
    pole_velo_discrete = discrete_one(pole_velo_float, infos.state_space_limits[3][0], infos.state_space_limits[3][1], infos.qt_size_array[3])
    return (cart_pos_discrete, cart_velo_discrete, pole_pos_discrete, pole_velo_discrete)
    
env = gym.envs.make("CartPole-v1")
Q_table = init_q_table()


def learn():
    init_graph()
    total_steps, total_episodes, avg_steps, avg_episodes, goal = [0], [0], [0], [0], [0, infos.graph_frequency]
    for episode in range(infos.episodes):
        state = env.reset()
        steps = 0
        done = False
        while not done:
            action = policy(state)
            new_state, reward, done, _  = env.step(action)
            if (infos.render == 1):
                env.render()
            steps += 1
            if done == True:
                reward = infos.reward_values[0]
            update_q_table(state, action, new_state, reward)
            state = new_state

        if ((episode % infos.average == 0) and (episode != 0)):
            avg_steps.append(np.mean(total_steps[-infos.average :]))
            avg_episodes.append(episode)
        infos.epsilon = infos.epsilon * infos.epsilon_decay
        if (infos.epsilon < infos.epislon_min):
            infos.epsilon = 1
        total_episodes.append(episode)
        total_steps.append(steps)

        if infos.graph and (episode % infos.graph_frequency == 0):
            goal = render_graph(avg_steps, avg_episodes, goal)
        if (episode % infos.graph_frequency == 0):
            print(f'episode = {episode}, avg_steps = {round(avg_steps[-1], 3)} and epsilon == {round(infos.epsilon, 3)}')

        if avg_steps[-1] > 400:
            with open(f"q_table_bis.pkl", "wb+") as f:
                pickle.dump(Q_table, f)
            return
        
    if infos.graph == 1:
        plt.title('cartpole_dd')
        plt.ylabel('steps')
        plt.xlabel('episode')
        plt.legend()
        plt.show()
        
    if infos.render == 1:
        env.close()

def play():
    infos.epsilon = 0.0
    while True:
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            new_state, reward, done, _  = env.step(action)
            env.render()
            state = new_state
    env.close()

def read_cutie(file):
    with open(file, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    args = parse_arguments()
    if args.load == False:
        learn()
    elif args.load == True:
        Q_table = read_cutie("q_table.pkl")
    play()
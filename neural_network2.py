# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

import gym
from matplotlib import pyplot as plt
from config import infos
import pickle
from collections import deque
import numpy as np
import random
import copy
import sys


env = gym.envs.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(f"state_size = {state_size}, action size = {action_size}")
output_dir = "./cartpole/outs"
memory = deque(maxlen=2000)

def init_model(state_size, action_size):
    learning_rate = infos.learning_rate
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_shape=[state_size], activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    #model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
    return model

def policy(state, predicted_qvalues, epsilon):
    if (random.random() < epsilon):
        action = random.randint(0, 1)
    else:
        action = np.argmax(predicted_qvalues)
    return (action)

def load(name, model):
    model.load_weights(name)
    
def save(name, model):
    model.save_weights(name)

def fit_model(state, action, reward, new_state, m1, m2, target_qvalues):
    target = reward + (infos.discount_factor * max(m2.predict(new_state)[0]))
    target_qvalues[0][action] = target 
    #is .fit() good ?
    # print(f"state = {state}, action = {action}, reward = {reward}, target_qvalues = {target_qvalues}")
    m1.fit(state, target_qvalues, verbose = 0)
    return m1, m2

def copy_model(model):
  model_copy = keras.models.clone_model(model)
  model_copy.build((None, action_size)) # replace 10 with number of variables in input layer
  model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=infos.learning_rate))
  model_copy.set_weights(model.get_weights())
  return (model_copy)

def eval(m1):
    results = []
    for episode in range(3):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        steps = 0
        done = False
        
        while not done and steps < 200:
            predicted_qvalues = m1.predict(state)
            action = np.argmax(predicted_qvalues[0])
            state, _, done, _  = env.step(action)
            state = np.reshape(state, [1, state_size])
            steps += 1
        results.append(steps)
    return np.mean(results)

def play(m1):
    while True:
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            action = random.randint(0, 1)#np.argmax(m1.predict(state)[0])
            new_state, reward, done, _  = env.step(action)
            env.render()
            state = new_state
    env.close()


def learn():
    epsilon = infos.epsilon
    m1 = init_model(state_size, action_size)
    m2 = init_model(state_size, action_size)
    m2.set_weights(m1.get_weights())
    for episode in range(infos.episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        steps = 0
        done = False
        
        while not done:
            predicted_qvalues = m2.predict(state)
            action = policy(state, predicted_qvalues[0], epsilon)
            new_state, reward, done, _  = env.step(action)
            new_state = np.reshape(new_state, [1, state_size])
            steps += 1
            if done == True:
                reward = infos.reward_values[0]
            memory.append((state, action, reward, new_state, done))
            m1, m2 = fit_model(state, action, reward, new_state, m1, m2, predicted_qvalues)
            state = new_state
        
        if len(memory) > 200 and (random.random() < 0.5):
          print(f"*** memory replay for episode:{episode}")
          minibatch = random.sample(memory, infos.batch_size)
          ### check minibatch
          for state, action, reward, new_state, done in minibatch:
              predicted_qvalues = m1.predict(state)
              action = policy(state, predicted_qvalues[0], epsilon)
              m1, m2 = fit_model(state, action, reward, new_state, m1, m2, predicted_qvalues)

        epsilon = epsilon * infos.epsilon_decay
        if (epsilon < infos.epislon_min):
            epsilon = infos.epislon_min
          
        print(f'\nepisode = {episode}, total_steps = {steps} and epsilon == {round(epsilon, 3)}')
        if episode % 10 == 0 and episode != 0:
          print(f"evaluation m1 = {eval(m1)}")
          m2.set_weights(m1.get_weights())

        if episode % 50 == 0 and episode != 0:
            save(f'outs/with_dqn_{episode}.hdf5', m1)
    return m1

if __name__ == "__main__":
    #m1 = learn()
    episode = sys.argv[1]
    m1 = None
    # m1 = init_model(state_size, action_size)
    # m1.load_weights(f'outs/with_dqn_{episode}.hdf5')
    play(m1)
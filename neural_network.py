import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import gym
from matplotlib import pyplot as plt
from config import infos
import pickle
from collections import deque
import numpy as np
import random
from tqdm import tqdm

env = gym.envs.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class DQNAgent:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=infos.len_memory)
        self.epsilon = infos.epsilon
        self.m1 = self.kreate_model()
        #self.m2 = self.kreate_model()
        #self.m2.set_weights(self.m1.get_weights())
        
        
    def kreate_model(self):
        learning_rate = infos.learning_rate
        model = keras.Sequential()
        model.add(keras.layers.Dense(8, input_shape=[self.state_size], activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
        return model

    def acting(self, state):
        if (random.random() < self.epsilon):
            return random.randrange(self.action_size)
        action = np.argmax(self.m1.predict(state)[0]) ### change to m2
        return (action)
    
    def fitting(self, state, action, reward, new_state, target_qvalues):
        target = reward + (infos.discount_factor * max(self.m1.predict(new_state)[0])) ### change to m2
        target_qvalues[0][action] = target 
        self.m1.fit(state, target_qvalues, verbose = 0)
        
    def evaluate(self):
        results = []
        for episode in range(infos.eval_size):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            steps = 0
            done = False
            
            while not done and steps < infos.replay_memory:
                predicted_qvalues = self.m1.predict(state)
                action = np.argmax(predicted_qvalues[0])
                state, _, done, _  = env.step(action)
                state = np.reshape(state, [1, state_size])
                steps += 1
            results.append(steps)
        return np.mean(results)
    
    def update_epsilon(self):
        if self.epsilon > infos.epislon_min:
            self.epsilon = self.epsilon * infos.epsilon_decay
        if (self.epsilon <= infos.epislon_min):
            self.epsilon = infos.epislon_min

def load(name, model):
    model.load_weights(name)
    
def save(name, model):
    model.save_weights(name)


def learn():
    agent = DQNAgent(state_size, action_size)
    for episode in tqdm(range(infos.episodes)):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        steps = 0
        done = False
        
        while not done:
            predicted_qvalues = agent.m1.predict(state) ### change to m2
            action = agent.acting(state)
            new_state, reward, done, _  = env.step(action)
            new_state = np.reshape(new_state, [1, state_size])
            steps += 1
            if done == True:
                reward = infos.reward_values[0]
            agent.memory.append((state, action, reward, new_state, done))
            agent.fitting(state, action, reward, new_state, predicted_qvalues)
            state = new_state
        
        if len(agent.memory) > infos.replay_memory and (random.random() < 0.5):
          minibatch = random.sample(agent.memory, infos.replay_memory)
          for state, action, reward, new_state, done in minibatch:
              predicted_qvalues = agent.m1.predict(state) ### change to m2
              action = agent.acting(state)
              agent.fitting(state, action, reward, new_state, predicted_qvalues)

        agent.update_epsilon()
        # print(f'\nepisode = {episode}, total_steps = {steps} and epsilon == {round(epsilon, 3)}')
        # if episode % 10 == 0 and episode != 0:
        #   print(f"evaluation m1 = {agent.evaluate()}")
        #   agent.m1.set_weights(agent.m1.get_weights()) ### change to m2

        if episode % 50 == 0 and episode != 0:
            agent.m1.save_weights(f'weigths/with_dqn_{episode}.hdf5', agent.m1)
            
    return agent.m1

if __name__ == "__main__":
    m1 = learn()
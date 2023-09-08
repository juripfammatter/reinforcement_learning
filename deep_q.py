#%%
import gym
import numpy as np
from collections import deque
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from memory_profiler import profile

#%%
def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform(42) # seed for identical sequence of random numbers
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_shape=state_shape, activation='swish', kernel_initializer=init))
    model.add(keras.layers.Dense(16, activation='swish', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

#%%
def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.95 # Learning rate
    discount_factor = 0.6

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64*2
    mini_batch = random.sample(replay_memory, batch_size)                       # 128
    current_states = np.array([transition[0] for transition in mini_batch])     # 128x4
    current_qs_list = model.predict(current_states)                             # 128x2
    new_current_states = np.array([transition[3] for transition in mini_batch]) # 128x4
    future_qs_list = target_model.predict(new_current_states)                   # 128x2

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)                                                   # 128x4 float32
        Y.append(current_qs)                                                    # 128x2 float32
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
    
#%%
def test(env, model, episodes):
    steps = 0

    for episode in range(0,episodes):
        steps += 1 
      
        score = 0
        terminated = False
        observation = env.reset()[0]

        while not terminated:
            # choose best action
            observation = observation.reshape([1, observation.shape[0]])
            action = np.argmax(model.predict(observation))
            
            # Take step
            new_observation, reward, terminated, truncated, info  = env.step(action)
            score+=reward
          
            observation = new_observation
            print(observation)
        
        print("score: ", score)

#%%
# @profile
def deep_q(env, model, target_model, params, results):
    # parameters
    train_episodes, decay = params
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.2 # At a minimum, we'll always explore 1% of the time

    # target_update_counter = 0
    steps_to_update_target_model = 0
    # X = states, y = actions
    X = []
    y = []
    replay_memory = deque(maxlen=50_000)

    # main algorithm
    for episode in range(int(train_episodes)):
        total_training_rewards = 0
        observation = env.reset()[0]
        done = False
        while not done:
            steps_to_update_target_model += 1

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                encoded = observation
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
            new_observation, reward, done,trunc, info = env.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(env, replay_memory, model, target_model, done)

            observation = new_observation
            total_training_rewards += reward

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        results.append(total_training_rewards)
    env.close()

    return model

#%% 
RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

env = gym.make('CartPole-v1')
np.random.seed(RANDOM_SEED)

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# parameters
train_episodes = 1000
# test_episodes = 200
decay = 0.005

params = [train_episodes, decay]

# 1. Initialize the Target and Main models
# Main Model (updated every 4 steps)
model = agent(env.observation_space.shape, env.action_space.n)
# Target Model (updated every 100 steps)
target_model = agent(env.observation_space.shape, env.action_space.n)
target_model.set_weights(model.get_weights())

results = []

start_time = time.time()
deep_q(env, model, target_model, params, results)
print("needed ", time.time()-start_time, " seconds")
# %%
sns.set_theme()
fig, axs = plt.subplots(1)
sns.scatterplot(results, alpha = 0.5, s=5)
# axs.set(ylim = [0,680])
# %%
test_env = gym.make("CartPole-v1", render_mode="human")
test_env.reset()

episodes = 10
test(test_env, model, episodes)
test_env.close()
# %%

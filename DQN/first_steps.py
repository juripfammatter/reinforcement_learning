# https://github.com/maciejbalawejder/Reinforcement-Learning-Collection/blob/main/Q-Table/Qtable.ipynb
# https://www.gymlibrary.dev/environments/classic_control/cart_pole/
# https://medium.com/analytics-vidhya/q-learning-is-the-most-basic-form-of-reinforcement-learning-which-doesnt-take-advantage-of-any-8944e02570c5
import gymnasium as gym
import time
import numpy as np

# returns Q-table
def Qtable(state_space,action_space,bin_size = 30):
    
    bins = [np.linspace(-4.8,4.8,bin_size),
            np.linspace(-4,4,bin_size),
            np.linspace(-0.418,0.418,bin_size),
            np.linspace(-4,4,bin_size)]
    
    q_table = np.random.uniform(low=-1,high=1,size=([bin_size] * state_space + [action_space]))
    return q_table, bins


def Discrete(state, bins):
    index = []
    for i in range(len(state)): index.append(np.digitize(state[i],bins[i]) - 1)
    return tuple(index)


def Q_learning(env,q_table, bins, episodes = 5000, gamma = 0.95, lr = 0.1, timestep = 100, epsilon = 0.2):
    
    rewards = 0
    steps = 0
    for episode in range(1,episodes+1):
        steps += 1 
        # env.reset() => initial observation
        current_state = Discrete(env.reset()[0],bins)
      
        score = 0
        terminated = False
        while not terminated: 
            #if episode%timestep==0: env.render()

            # Exploration
            if np.random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[current_state])
            
            # Take step
            observation, reward, terminated, truncated, info  = env.step(action)
            next_state = Discrete(observation,bins)
            score+=reward
          
            #Update Q-table
            if not terminated:
                    max_future_q = np.max(q_table[next_state])
                    current_q = q_table[current_state+(action,)]
                    new_q = (1-lr)*current_q + lr*(reward + gamma*max_future_q)
                    q_table[current_state+(action,)] = new_q
            current_state = next_state
            
        # End of the loop update
        else:
            rewards += score
            if score > 195 and steps >= 100: print('Solved')
        if episode % timestep == 0: print(reward / timestep)

def test(env, q_table):
    # env.reset() => initial observation
    current_state = Discrete(env.reset()[0],bins)
    score = 0

    for _ in range(10000):
        # take action based on q-table
        action = np.argmax(q_table[current_state])
        # take next step
        observation, reward, terminated, truncated, info  = env.step(action)
        next_state = Discrete(observation,bins)

        score+=reward

        current_state = next_state

        if terminated: return


train_env = gym.make("CartPole-v1")
train_env.reset()

# stats
print(train_env.action_space) # 2 actions
print(train_env.observation_space) # 4 states
# Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
# [x_min, vx_min, phi_min, vphi_min][x_max, vx_max, phi_max, vphi_max]

### Q-Table
q_table, bins = Qtable(len(train_env.observation_space.low), train_env.action_space.n)
print(q_table.shape)

### Training
Q_learning(train_env,q_table, bins, lr = 0.25, gamma = 0.995, episodes = 5*(10**3), timestep = 1000)
train_env.close()


### Testing
test_env = gym.make("CartPole-v1", render_mode="human")
test_env.reset()
test(test_env,q_table)

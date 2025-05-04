from collections import defaultdict
import time
import gymnasium as gym
import highway_env
import highway_env.envs
import highway_env.envs.common
import highway_env.envs.common.observation
from matplotlib import pyplot as plt
import numpy as np


class HighwayAgent:
    def __init__(self, method, env, alpha = 0.1, gamma = 0.9, epsilon_start = 0.8, epsilon_decay = 0.9999, epsilon_min = 0.1):
        self.method = method
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_values = np.zeros((64, 5))
    
    def getAction(self, state, isOptimal = False):  # isOptimal = True, in case you want completely greedy (usually used in testing)
        if np.random.rand() > self.epsilon or isOptimal:
            return np.argmax(self.q_values[state])
        else:
            return self.env.action_space.sample()
            
        
    def update(self, state, action, reward, next_state, next_action, done):
        self.q_values[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_values[next_state]) * (not done) - self.q_values[state, action])
        

    def decayEpsilon(self, episode):
        self.epsilon = max(self.epsilon_min, self.epsilon_start * (self.epsilon_decay ** episode))

class CustomHighwayObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.ego = env.unwrapped.vehicle


    def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            modified_obs = self.cont2discrete(obs)
            return modified_obs, info

    def step(self, action):
        # Original environment step function, obs is TTC
        obs, reward, terminated, truncated, info = self.env.step(action)
        modified_obs = self.cont2discrete(obs)

        return modified_obs, reward, terminated, truncated, info
    
    def cont2discrete(self, obs):
        ego_speed = self.ego.speed
        speed_index = 0 if ego_speed <= 15 else 1 if ego_speed <= 20 else 2

        # Flatten the relevant speed layer (3 lanes x bins)
        lanes = obs[speed_index]
        ttc_bins = []

        for lane in lanes:
            # Find the first non-zero index (i.e., closest vehicle)
            non_zero_indices = np.nonzero(lane)[0]
            if len(non_zero_indices) == 0:
                ttc = 6  # Treat as max distance bin
            else:
                ttc = non_zero_indices[0]

            # Bin conversion using vectorized logic
            if ttc < 2:
                ttc_bin = 0
            elif ttc < 4:
                ttc_bin = 1
            elif ttc < 6:
                ttc_bin = 2
            else:
                ttc_bin = 3

            ttc_bins.append(ttc_bin)

        return ttc_bins[0] * 16 + ttc_bins[1] * 4 + ttc_bins[2]

            
def train(method, num_episodes = 40_000, alpha = 0.1, gamma = 0.9, epsilon_start = 0.8, epsilon_decay = 0.9999, epsilon_min = 0.1, showProgress = False):             
    env = gym.make(
        'highway-fast-v0',  
        config = {
        "observation": {
            "type": "TimeToCollision",
        },
        "vehicles_count": 10,

    }
    )

    env = CustomHighwayObs(env)

    agent = HighwayAgent(method, env, alpha, gamma, epsilon_start, epsilon_decay, epsilon_min)
    max_steps = 10
    episode_rewards = []
    smoothed_rewards = []
    window = 1000
    for episode in range(num_episodes):
        state = env.reset()[0]
        action = agent.getAction(state)
        total_reward = steps = 0
        done = False
        # Episode Start
        while not done and steps <= max_steps:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.getAction(next_state)
            done = terminated or truncated
            total_reward += reward

            agent.update(state, action, reward, next_state, next_action, done)

            action = next_action
            state =  next_state
            steps += 1
        # Episode end
        
        episode_rewards.append(total_reward)
        agent.decayEpsilon(episode)
        smoothed_rewards.append(np.mean(episode_rewards[-window:]))

        if showProgress and episode % (num_episodes/10) == 0:
            print(f'\r{method}: {int(episode/num_episodes*100)}%', end='', flush=True)
    env.close()
    return smoothed_rewards, env, agent

smoothed,  env, agent = train("Q", num_episodes=10_000, showProgress=True)
env.close()
print('\nDone!')
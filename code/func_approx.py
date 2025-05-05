from collections import defaultdict
import time
import gymnasium as gym
import highway_env
import highway_env.envs
import highway_env.envs.common
import highway_env.envs.common.observation
from matplotlib import pyplot as plt
import numpy as np
import math

class TileCoder:
    def __init__(self, env, num_tilings, tiles_per_tiling, low, high):
        self.num_tilings = num_tilings
        self.tiles_per_tiling = tiles_per_tiling
        self.low = low
        self.high = high
        self.tile_width = (high - low) / tiles_per_tiling
        self.offsets = np.linspace(0, self.tile_width, num_tilings, endpoint=False)
        self.actions = env.actions_space.n

    def get_features(self, s, a):
        features = np.zeros(self.num_tilings * self.tiles_per_tiling * self.actions)
        for i in range(self.num_tilings):
            shifted_s = s + self.offsets[i]
            pos = math.floor((shifted_s - self.low) / self.tile_width)
            pos = min(max(pos, 0), self.tiles_per_tiling - 1)
            index = i * self.tiles_per_tiling * self.actions + pos * self.actions + a
            features[index] = 1
        return features
    
class SemiGradientSarsaAgent:
    def __init__(self, env, tile_coder, alpha, gamma, epsilon):
        self.tc = tile_coder
        self.alpha = alpha / tile_coder.num_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = range(env.action_space.n)
        self.w = np.zeros(tile_coder.num_tilings * tile_coder.tiles_per_tiling * len(self.actions))

    def select_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax([self.get_q(s, a) for a in self.actions])

    def get_q(self, s, a):
        return np.dot(self.w, self.tc.get_features(s, a))

    def update(self, s, a, r, s_next, a_next):
        features = self.tc.get_features(s, a)
        target = r + self.gamma * self.get_q(s_next, a_next)
        prediction = np.dot(self.w, features)
        self.w += self.alpha * (target - prediction) * features

def train(agent, num_episodes=500):
    env = gym.make(
        'highway-fast-v0', 
        config = {
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [0, 1],
                "y": [0, 1],
                "vx": [0, 1],
                "vy": [0, 1],
            },
        },
        "vehicles_count": 5,

    }
    )
    max_steps = 10
    episode_rewards = []
    smoothed_rewards = []
    moving_avg_window = num_episodes//20

    for episode in range(num_episodes):
        state = env.reset()[0]
        action = agent.select_action(state)
        total_reward = steps = 0
        done = False

        # Episode Start
        while not done and steps <= max_steps:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.select_action(next_state)
            done = terminated or truncated
            total_reward += reward

            agent.update(state, action, reward, next_state, next_action, done)

            action = next_action
            state =  next_state
            steps += 1
        # Episode end
        
        episode_rewards.append(total_reward)
        agent.decayEpsilon(episode)

        if showProgress and episode % (num_episodes/10) == 0:
            print(f'\r{method}: {int(episode/num_episodes*100)}%', end='', flush=True)
        smoothed_rewards.append(np.mean(rewards[-moving_avg_window:]))

    env.close()
    return smoothed_rewards, env, agent

    return agent



tile_coder = TileCoder(num_tilings=100, tiles_per_tiling=8, low=0.0, high=1.0)
agent = SemiGradientSarsaAgent(tile_coder, alpha=0.1, gamma=0.99, epsilon=0.1)

lengths, returns, visits, trained_agent = train(env, agent, num_episodes=500)



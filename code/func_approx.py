import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import time
import gymnasium as gym
import highway_env
import highway_env.envs
import highway_env.envs.common
import highway_env.envs.common.observation



class TileCoder:
    def __init__(self, num_tilings, tiles_per_tiling, low, high):
        self.num_tilings = num_tilings
        self.tiles_per_tiling = tiles_per_tiling
        self.low = low
        self.high = high
        self.tile_width = (high - low) / tiles_per_tiling
        self.offsets = np.linspace(0, self.tile_width, num_tilings, endpoint=False)

    def get_features(self, s, a):
        features = np.zeros(self.num_tilings * self.tiles_per_tiling * 2)
        for i in range(self.num_tilings):
            shifted_s = s + self.offsets[i]
            pos = math.floor((shifted_s - self.low) / self.tile_width)
            pos = min(max(pos, 0), self.tiles_per_tiling - 1)
            index = i * self.tiles_per_tiling * 2 + pos * 2 + a
            features[index] = 1
        return features




class SemiGradientSarsaAgent:
    def __init__(self, tile_coder, alpha, gamma, epsilon):
        self.tc = tile_coder
        self.alpha = alpha / tile_coder.num_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.w = np.zeros(tile_coder.num_tilings * tile_coder.tiles_per_tiling * 2)

    def select_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        return np.argmax([self.get_q(s, 0), self.get_q(s, 1)])

    def get_q(self, s, a):
        return np.dot(self.w, self.tc.get_features(s, a))

    def update(self, s, a, r, s_next, a_next):
        features = self.tc.get_features(s, a)
        target = r + self.gamma * self.get_q(s_next, a_next)
        prediction = np.dot(self.w, features)
        self.w += self.alpha * (target - prediction) * features


class DifferentialSarsaAgent:
    def __init__(self, tile_coder, alpha, beta, epsilon):
        self.tc = tile_coder
        self.alpha = alpha / tile_coder.num_tilings
        self.beta = beta
        self.epsilon = epsilon
        self.avg_reward = 0.0
        self.w = np.zeros(tile_coder.num_tilings * tile_coder.tiles_per_tiling * 2)

    def select_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        return np.argmax([self.get_q(s, 0), self.get_q(s, 1)])

    def get_q(self, s, a):
        return np.dot(self.w, self.tc.get_features(s, a))

    def update(self, s, a, r, s_next, a_next):
        x = self.tc.get_features(s, a)
        x_next = self.tc.get_features(s_next, a_next)
        delta = r - self.avg_reward + np.dot(self.w, x_next) - np.dot(self.w, x)
        self.avg_reward += self.beta * delta
        self.w += self.alpha * delta * x



np.random.seed(42)
env = gym.make(
        'highway-fast-v0',  
        config = {
        "observation": {
            "type": "Kinematic",
            "features": ["presence","x", "y", "vx", "vy"],
            "features_range": {
                "x": [0, 1.0],
                "y": [0, 1.0],
                "vx": [0, 1.0],
                "vy": [0, 1.0],
            }
        },
        "vehicles_count": 5,
    }
    )
tile_coder = TileCoder(num_tilings=40, tiles_per_tiling=35, low=0.0, high=1.0)
agent = DifferentialSarsaAgent(tile_coder, alpha=0.1, beta=0.01, epsilon=0.05)
# agent = SemiGradientSarsaAgent(tile_coder, alpha=0.1, gamma=0.99, epsilon=0.1)

# rewards, visits, episode_lengths, trained_agent = train(env, agent, steps=10000)
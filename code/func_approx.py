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
    def __init__(self, num_tilings, tiles_per_dim, low, high, feature_vector_size):
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.feature_vector_size = feature_vector_size
        self.low = low
        self.high = high
        self.offsets = np.linspace(0, 1, num_tilings, endpoint=False)  # Offsets for tilings

    def get_features(self, s, a):
        active_tiles = []
        # Normalize state variables to [0, 1] using low and high for each dimension
        scaled_s = (s - self.low) / (self.high - self.low)

        # Iterate through each tiling
        for i in range(self.num_tilings):
            # Apply offset to each dimension after scaling
            shifted = scaled_s + self.offsets[i]
            # Compute the positions of the tiles for each dimension
            positions = np.floor(shifted * self.tiles_per_dim).astype(int)
            tile = (i, *positions, a)  # Include tiling index and action
            hashed_index = hash(tile) % self.feature_vector_size  # Hash the feature into the vector space
            active_tiles.append(hashed_index)

        features = np.zeros(self.feature_vector_size)
        # Set active tiles in the feature vector
        for idx in active_tiles:
            features[idx] = 1
        return features


class SemiGradientSarsaAgent:
    def __init__(self, env, tile_coder, alpha, gamma, epsilon_start, epsilon_decay, epsilon_min):
        self.tc = tile_coder
        self.alpha = alpha / tile_coder.num_tilings
        self.gamma = gamma
        self.epsilon = self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.actions = range(env.action_space.n)
        self.w = np.zeros(tile_coder.feature_vector_size)

    def select_action(self, s, isOptimal=False):
        if np.random.rand() < self.epsilon and not isOptimal:
            return np.random.choice(self.actions)
        return np.argmax([self.get_q(s, a) for a in self.actions])

    def get_q(self, s, a):
        return np.dot(self.w, self.tc.get_features(s, a))

    def update(self, s, a, r, s_next, a_next):
        features = self.tc.get_features(s, a)
        target = r + self.gamma * self.get_q(s_next, a_next)
        prediction = np.dot(self.w, features)
        self.w += self.alpha * (target - prediction) * features

    def decay_epsilon(self, episode):
        self.epsilon = max(self.epsilon_min, self.epsilon_start * (self.epsilon_decay ** episode))

def train(num_episodes=500):
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
    tile_coder = TileCoder(num_tilings=16, tiles_per_dim=8, feature_vector_size=200_000,  low=np.zeros(25), high=np.ones(25))
    agent = SemiGradientSarsaAgent(env, tile_coder, alpha=0.2, gamma=0.99, epsilon_start=0.8, epsilon_min=0.01, epsilon_decay=0.995)


    max_steps = 10
    episode_rewards = []

    for episode in range(num_episodes):
        start = time.perf_counter()
        state = env.reset()[0].flatten()
        action = agent.select_action(state)
        total_reward = steps = 0
        done = False

        # Episode Start
        while not done and steps <= max_steps:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()
            next_action = agent.select_action(next_state)
            done = terminated or truncated
            total_reward += reward

            agent.update(state, action, reward, next_state, next_action)

            action = next_action
            state =  next_state
            steps += 1
        # Episode end
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon(episode)
        if episode % (num_episodes/100) == 0:
            print(f'\r{int(episode/num_episodes*100)}%', end='', flush=True)

        # smoothed_rewards.append(np.mean(total_reward[-moving_avg_window:]))

    env.close()
    return episode_rewards, env, agent


rewards, env, trained_agent = train(num_episodes=10_000)

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
        
        self.q_values = {}
    
    def getAction(self, state, isOptimal = False):  # isOptimal = True, in case you want completely greedy (usually used in testing)
        q_values =  self.q_values + self.q_values2 if self.method == 'Double Q' else self.q_values
        if np.random.rand() > self.epsilon or isOptimal:
            return np.argmax(q_values[state])
        else:
            return self.env.action_space.sample()
            
        
    def update(self, state, action, reward, next_state, next_action, done):
        if self.method == 'SARSA':
            self.q_values[state, action] += self.alpha * (reward + self.gamma * self.q_values[next_state, next_action] * (not done) - self.q_values[state, action]) 

        elif self.method == 'Expected SARSA':
            action_prob = np.ones(self.env.action_space.n) * (self.epsilon / self.env.action_space.n)
            best_action = self.getAction(next_state, True)
            action_prob[best_action] += (1 - self.epsilon)
            expected =  np.dot(action_prob, self.q_values[next_state])
            self.q_values[state, action] += self.alpha * (reward + self.gamma * expected * (not done) - self.q_values[state, action])
        
        elif self.method == 'Q':
            self.q_values[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_values[next_state]) * (not done) - self.q_values[state, action])

        elif self.method == 'Double Q':
            if np.random.rand() > 0.5:
                next_action = np.argmax(self.q_values[next_state])
                self.q_values[state, action] += self.alpha * (reward + self.gamma * self.q_values2[next_state, next_action] * (not done) - self.q_values[state, action])
            else:
                next_action = np.argmax(self.q_values2[next_state]) 
    
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
        # Original environment step function, obs is kinematics and absoulte
        obs, reward, terminated, truncated, info = self.env.step(action)
        # print("raw = ",obs)
        modified_obs = self.cont2discrete(obs)

        return modified_obs, reward, terminated, truncated, info
    
    def cont2discrete(self, obs):
        '''
        This function gets the old kinematic observation and returns our new discrete observation [ttc_bin, lane, heading]
        where:
        ttc_bin: a bin that represents ttc disrubuated as such {0: 0-2, 1: 2-4, 2:4-6, 3: >6}
        lane: represents which lane said vehicle is
        heading: represents where that vehicle is going (1 is down, -1 is up, 0 is straight)
                                                0        1  2   3   4
        NOTE: Old obs is as such for each car [Presance, x, y, vx, vy]
        For y, each lane occupies a 0.25 space, from 0 to 1
        For vy, positve is going down, negative is going up
        '''
        temp = obs[1:]
        new_obs = np.zeros((temp.shape[0], 3))
        ego_obs = obs[0]
        for index, car_obs in enumerate(temp):
            if car_obs[0] > 0:
                distance = car_obs[1]  # Distance between the car and us, (-) means we are infront
                speed = car_obs[3] # Relative speed between us and car, (-) means we are slower
                # print(speed)

                # Calculating ttc, if ttc is negative it means ttc is effectavily inf
                if speed <= 1e-6 or distance <= 0:
                    ttc = -1  # Handle invalid TTC
                else:
                    ttc = distance / speed

                if ttc < 0 or ttc > 6:
                    ttc_bin = 3
                elif ttc < 2:
                    ttc_bin = 0
                elif ttc < 4:
                    ttc_bin = 1
                else:  # 4 <= ttc <= 6
                    ttc_bin = 2

                # Getting the lane
                y_pos = round(car_obs[2], 2)
                if abs(y_pos) < 0.04 / 2:  # Small tolerance for floating-point precision
                    lane_index = 0  # Same lane as ego vehicle
                
                # Use rounding to determine the lane index
                lane_index = np.clip(round(y_pos / 0.04), -2, 2)
                # 0 same lane, 1 immediate right, 2 Far right etc
                
                # Getting the heading
                y_speed = round(car_obs[4], 4)
                if y_speed > 0:
                    heading = 1
                elif y_speed < 0:
                    heading = -1
                else:
                    heading = 0
            else:
                # Default values if car isnt on screen
                ttc_bin = 3
                lane_index = 0
                heading = 0

            new_obs[index] = [ttc_bin, lane_index, heading]
        return new_obs
    
def train(method, num_episodes = 40_000, alpha = 0.1, gamma = 0.9, epsilon_start = 0.8, epsilon_decay = 0.9999, epsilon_min = 0.1, showProgress = False):             
    env = gym.make(
        'highway-v0', 
        render_mode='rgb_array', 
        config = {
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "heading"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "heading": [-22/7, 22/7]
            },
            "absolute": False,
        },
        "vehicles_count": 30,

    }
    )

    env = CustomHighwayObs(env)

    agent = HighwayAgent(method, env, alpha, gamma, epsilon_start, epsilon_decay, epsilon_min)

    episode_rewards = []
    smoothed_rewards = []
    window = 1000
    for episode in range(num_episodes):
        state = env.reset()[0]
        action = agent.getAction(state)
        total_reward = 0
        done = False

        # Episode Start
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.getAction(next_state)
            done = terminated or truncated
            total_reward += reward

            agent.update(state, action, reward, next_state, next_action, done)

            action = next_action
            state =  next_state
        # Episode end
        
        episode_rewards.append(total_reward)
        agent.decayEpsilon(episode)
        smoothed_rewards.append(np.mean(episode_rewards[-window:]))
        if showProgress and episode % (num_episodes/10) == 0:
            print(f'\r{method}: {int(episode/num_episodes*100)}%', end='', flush=True)
    env.close()
    return smoothed_rewards, env, agent

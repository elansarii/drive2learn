import gymnasium as gym
import highway_env
import highway_env.envs
import highway_env.envs.common
import highway_env.envs.common.observation
from matplotlib import pyplot as plt
import numpy as np

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
                speed = -car_obs[3] # Relative speed between us and car, (-) means we are slower

                # Calculating ttc, if ttc is negative it means ttc is effectavily inf

                # print(distance)
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
    "vehicles_count": 1,

}
)
env = CustomHighwayObs(env)

env.reset()
for _ in range(1000):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    print(obs[0])
    env.render()
print(env.action_space.n)

plt.imshow(env.render())

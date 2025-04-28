import gymnasium
import highway_env
import highway_env.envs
import highway_env.envs.common
import highway_env.envs.common.observation
from matplotlib import pyplot as plt
import numpy as np

def calculateTTC(d, v):
    return d/v

def calculateHeading(vy):
    if vy > 0.01:
        return 1
    elif vy < -0.01:
        return -1
    else:
        return 0
    
def getLane(y):
    y = np.round(y, decimals=2)
    if y < 0.25:
        lane = 0
    elif y < 0.5:
        lane = 1
    elif y < 0.75:
        lane = 2
    else:
        lane = 3

    return lane


env = gymnasium.make('highway-v0', render_mode='rgb_array',config={"type": "KinematicObservation", "absolute":True, "manual_control": True})
env.reset()
for _ in range(100):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    
    if obs[1, 0] != 0:
        d = obs[1, 0] - obs[0, 0]   # If (-) than we are in front
        v = obs[1, 0] - obs[0, 0]   # If (-) then we are faster
        vy = obs[1, 4]
        y = obs[1, 2]

        isFront = bool(d < 0)
        isFaster = bool(v < 0)
        d = abs(d)
        v =  abs(v)
        if (isFront and not isFaster) or (not isFront and isFaster):
            ttc = calculateTTC(d, v)
        else:
            ttc = float("inf")

        heading = calculateHeading(vy)
        lane = getLane(y)
        new_obs = [ttc, lane, heading, isFront]


    env.render()
    print(f'{obs[1, 2]:.2f}')

plt.imshow(env.render())

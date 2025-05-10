## **Team Members**

- Mohamed Elansari (ID: 202209852)
- Ali Ghazi (ID: 202209865)
- Abdullah Jamali (ID: 202104080)
## 1. **Abstract**

This project utilizes the `highway-v0` environment from Gymnasium to simulate autonomous driving in a multi-lane highway scenario. The environment consists of four lanes, with the agent vehicle and four other vehicles interacting within this space. The other vehicles in the environment perform simple actions, such as changing lanes (up or down), while the agent has a broader set of actions at its disposal: changing lanes (up or down), speeding up, slowing down, or idling to maintain its current speed and lane.

The primary objective of this project is to demonstrate the agent's ability to learn driving strategies through two distinct methods: the tabular method and functional approximation. This allows us to compare how the agent performs in discrete and continuous state spaces, respectively.

By training the agent in both discrete and continuous settings, we explore the effectiveness of different state representations in achieving safe and efficient driving behavior. The project highlights the agent's learning process, comparing its performance across both approaches while emphasizing the trade-offs between simplicity and accuracy in the observation space.


## 2. **Introducing the environment**

We opted for the `highway-fast-v0` variation to speed up simulation time, making training more efficient for testing different parameters. However, this comes at the cost of reduced realism, which may limit the agent's ability to generalize to more complex, real-world driving scenarios. The environment will consist of 5 vehicles beside to the agent's vehicle.

### 2.1 Action Space
We employ the `DiscreteMetaAction` space, which abstracts high-level driving maneuvers into five discrete actions:

```bash
0: 'LANE_LEFT'   # Change to the left lane
1: 'IDLE'        # Maintain current speed and lane
2: 'LANE_RIGHT'  # Change to the right lane
3: 'FASTER'      # Increase speed
4: 'SLOWER'      # Decrease speed
```

### 2.2 Reward Function
The reward function is designed to balance the trade-off between maintaining high speed and avoiding collisions. It is defined as:
$$
R(s, a) = a \cdot \frac{v - v_{\min}}{v_{\max} - v_{\min}} - b \cdot \text{collision}

$$
Where:
$$
\begin{aligned}
v & : \text{ current speed of the ego-vehicle} \\
v_{\min}, v_{\max} & : \text{ minimum and maximum speeds for normalization} \\
\text{collision} & : \text{ binary indicator (1 = collision, 0 = no collision)} \\
a, b & : \text{ weighting coefficients (speed, safety)}
\end{aligned}
$$

This formulation encourages the agent to drive at higher speeds while penalizing collisions, promoting efficient and safe driving behavior.

## 3. **Algorithms and Methodology** 

### 3.1 Tabular Method

In the tabular method, we use the `TimeToCollision` observation space, where the agent bases its decisions on the time-to-collision (TTC) with nearby vehicles. The TTC is represented as three arrays: one for speeds ≤15, one for 15–20, and one for >20. Each array has 3 rows (lanes: left, center, right) and 10 columns indicating seconds to collision (0 = imminent). A value of 1 marks a detected vehicle. Our implementation simplifies this by only using the array corresponding to the agent’s current speed range, ignoring the other two. We compared mainly between Expected SARSA and Q-Learning.

### 3.2 Functional Approximation Method

In contrast, the functional approximation method leverages the `Kinematics` observation space, which provides a continuous representation of the environment. Here, the agent observes detailed kinematic features, such as its position (`x`, `y`), velocity components (`vx`, `vy`), and orientation. This continuous state space allows the agent to make more nuanced decisions based on its position, speed, and the dynamics of the surrounding vehicles, offering a more realistic and complex decision-making process. We compared between both Semi-Gradient SARSA and Differential SARSA.

- **Presence**: 0 or 1, indicating if a vehicle is present (1 for the agent's car).
- **x [0, 1]**: Normalized position of the vehicle along the road (longitudinal).
- **y [0, 1]**: Normalized position of the vehicle across the lanes (lateral).
- **vx [0, 1]**: Normalized speed of the vehicle along the road (longitudinal speed).
- **vy [0, 1]**: Normalized speed of the vehicle across lanes (lateral speed).

These features are scaled to [0, 1] to help the agent's model (like a neural network) learn effectively.



## 4. Training Setup

The experiments were conducted on the `highway-fast-v0` environment from the Highway-env library. Both tabular methods and function approximation methods were tested under comparable conditions, with the same number of vehicles (**5**) and initial hyperparameter settings. The tabular approach was based on a classic tabular Expected SARSA implementation, while the function approximation approach utilized tile coding combined with linear value function approximation.

In the initial phase, we used **10 maximum steps per episode** to accelerate early testing and parameter tuning. The tabular method was run for **20,000 episodes**, while the function approximation method was run for **10,000 episodes**. A summary of the key training parameters for both approaches is presented in Table 1.

To further explore the capability of the tabular method to achieve convergence, the number of episodes was subsequently increased to **50,000**, and the learning rate (`alpha`) was raised from **0.1 to 0.3** to enable larger updates to the Q-values. Additionally, the maximum steps per episode were extended from **10 to 50**, allowing the agent to gather more reward information within each episode. This modification resulted in a substantial performance improvement, with the tabular agent’s average reward over the last 1,000 episodes increasing from **9 to 19**. However, despite this improvement, the learning process remained slow and computationally expensive, and full convergence was not achieved.

|parameter|Tabular|Functional approx|
|---|---|---|
|vehicles count|5|5|
|episodes|20,000|10,000|
|epsilon|0.8|0.8|
|gamma|0.9|-|
|epsilon decay|0.9999|0.999|
|alpha|0.3|0.005|
|max steps|50|50|
|tiles per tiling|-|32|
|tilings|-|16|
|beta|-|0.1|
|feature vector size|-|200,000|

---

## 5. **Experimental Results**

### 5.3 Graphs

<table>
<tr>
<td><img src="tabular-first.png" width="300"/><br><b>Figure 1a:</b> Tabular Method Initial Results</td>
<td><img src="functional-first.png" width="300"/><br><b>Figure 1b:</b> Differential SARSA Initial Results</td>
</tr>
<tr>
<td><img src="tabular-tuned.png" width="300"/><br><b>Figure 2a:</b> Expected SARSA After Tuning</td>
<td><img src="functional-tuned.png" width="300"/><br><b>Figure 2b:</b> Differential SARSA After Tuning</td>
</tr>
</table>



### 5.2 Initial Testing

The initial results of the tabular and function approximation methods are shown in **Figure 1**. The tabular method (Figure 1a) demonstrated a gradual but consistent increase in performance, reaching an average reward of approximately **8.19** after 20,000 episodes. The corresponding plot for Differential SARSA with function approximation (Figure 1b) showed faster learning, achieving comparable performance with only **10,000 episodes**, but at the cost of significantly higher variance. The Differential SARSA learning curve displayed substantial fluctuations with pronounced peaks and valleys, indicating instability in the learning process.

#### 5.3 Parameter Tuning and Extended Experiments

After tuning the parameters and increasing max steps to **50**, we observed a dramatic improvement in the Expected SARSA agent’s performance (Figure 2a). The Expected SARSA agent achieved an average reward of **20.04** over the last 1,000 episodes and maintained a smooth upward trend throughout the training period of 20,000 episodes. This was in stark contrast to the tuned Differential SARSA agent (Figure 2b), which, despite improvements, only reached an average reward of **12.32** and continued to exhibit high variance and inconsistent learning behavior.

We also experimented with Q-Learning under the same conditions. While Q-Learning achieved performance comparable to Expected SARSA, the learning curve was noticeably more noisy, reflecting the known overestimation bias associated with Q-Learning, especially in stochastic environments.

---

## 6. **Analysis and Discussion**

The results indicate clear differences in the practical performance of the algorithms. Initially, Differential SARSA showed promising learning speed, matching the performance of Expected SARSA in fewer episodes. However, the extreme variance rendered it impractical for stable policy learning. Even with extensive parameter tuning—including a reduction of the learning rate to **0.005** to dampen oscillations—the algorithm failed to achieve the stability or final performance levels of Expected SARSA.

These findings led us to hypothesize that the primary limitation of Differential SARSA lies in its sensitivity to noise. The `highway-fast-v0` environment is inherently characterized by **stochastic kinematics, multi-agent interactions, and partial observability**, all of which introduce random fluctuations that are amplified by Differential SARSA’s reliance on the average reward update controlled by `beta`. In contrast, Expected SARSA uses expected value estimates rather than sampled next-state actions, resulting in more stable learning and reduced variance. Attempts to switch to semi-gradient SARSA as an alternative also yielded similar high variance and poor performance, further supporting this conclusion.

Overall, **Expected SARSA with function approximation proved to be the most effective and reliable algorithm** for this environment, demonstrating superior stability and highest final reward performance. While Q-Learning produced comparable returns, its increased noise makes Expected SARSA the preferable choice for applications requiring smooth and stable control policies.
## Conclusion


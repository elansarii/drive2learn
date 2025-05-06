## **Team Members**

- Mohamed Elansari (ID: 202209852)
- Ali Ghazi (ID: 202209865)
- Abdullah Jamali (ID: 202104080)
## **Abstract**

This project utilizes the `highway-v0` environment from Gymnasium to simulate autonomous driving in a multi-lane highway scenario. The environment consists of four lanes, with the agent vehicle and four other vehicles interacting within this space. The other vehicles in the environment perform simple actions, such as changing lanes (up or down), while the agent has a broader set of actions at its disposal: changing lanes (up or down), speeding up, slowing down, or idling to maintain its current speed and lane.

The primary objective of this project is to demonstrate the agent's ability to learn driving strategies through two distinct methods: the tabular method and functional approximation. This allows us to compare how the agent performs in discrete and continuous state spaces, respectively.

In the tabular method, we utilize the `TimeToCollision` observation space, where the agent’s decision-making process is based on the time-to-collision (TTC) values between the ego-vehicle and surrounding vehicles. This approach focuses on providing the agent with discrete information, allowing it to make decisions based on the proximity of other vehicles, which is critical for avoiding collisions.

In contrast, the functional approximation method leverages the `Kinematics` observation space, which provides a continuous representation of the environment. Here, the agent observes detailed kinematic features, such as its position (`x`, `y`), velocity components (`vx`, `vy`), and orientation. This continuous state space allows the agent to make more nuanced decisions based on its position, speed, and the dynamics of the surrounding vehicles, offering a more realistic and complex decision-making process.

By training the agent in both discrete and continuous settings, we explore the effectiveness of different state representations in achieving safe and efficient driving behavior. The project highlights the agent's learning process, comparing its performance across both approaches while emphasizing the trade-offs between simplicity and accuracy in the observation space.


## **Introducing the environment**

We opted for the `highway-fast-v0` variation to speed up simulation time, making training more efficient for testing different parameters. However, this comes at the cost of reduced realism, which may limit the agent's ability to generalize to more complex, real-world driving scenarios. The environment will consist of 5 vehicles beside to the agent's vehicle.

### **Action Space**
We employ the `DiscreteMetaAction` space, which abstracts high-level driving maneuvers into five discrete actions:

```bash
0: 'LANE_LEFT'   # Change to the left lane
1: 'IDLE'        # Maintain current speed and lane
2: 'LANE_RIGHT'  # Change to the right lane
3: 'FASTER'      # Increase speed
4: 'SLOWER'      # Decrease speed
```

### **Reward Function**
The reward function is designed to balance the trade-off between maintaining high speed and avoiding collisions. It is defined as:

$R(s, a) = a \cdot \frac{v - v_{\min}}{v_{\max} - v_{\min}} - b \cdot \text{collision}$

Where:

| Symbol              | Description                                           |
|---------------------|-------------------------------------------------------|
| $v$                 | Current speed of the ego-vehicle                      |
| $v_{\min}, v_{\max}$| Minimum and maximum speeds for normalization         |
| $\text{collision}$  | Binary indicator ($1$ = collision, $0$ = no collision)|
| $a, b$              | Weighting coefficients (speed, safety)                |


This formulation encourages the agent to drive at higher speeds while penalizing collisions, promoting efficient and safe driving behavior.

### Observation space

### Tabular Method

In the **tabular method**, the agent uses **Time to Collision (TTC)** with vehicles ahead, left, and right. These values help the agent decide on actions based on how close a collision might be. It’s simple and works well when the observation space is discretized.

### Functional Approximation Method

For **functional approximation**, you use a set of continuous features to represent the state:

- **Presence**: 0 or 1, indicating if a vehicle is present (1 for the agent's car).
- **x [0, 1]**: Normalized position of the vehicle along the road (longitudinal).
- **y [0, 1]**: Normalized position of the vehicle across the lanes (lateral).
- **vx [0, 1]**: Normalized speed of the vehicle along the road (longitudinal speed).
- **vy [0, 1]**: Normalized speed of the vehicle across lanes (lateral speed).

These features are scaled to [0, 1] to help the agent's model (like a neural network) learn effectively.

By using continuous features, functional approximation allows for better generalization compared to the tabular method, especially in complex environments.




## **Team Members**

- Mohamed Elansari (ID: 2209852)
- Ali Ghazi (ID: 202209865)
- Abdullah Jamali (ID: 202104080)
## **Project Abstract**

### **Environment Overview**

Our project utilizes the `highway-v0` environment from the [Highway-env](https://highway-env.farama.org/environments/highway/) suite, designed for simulating autonomous driving scenarios. In this environment, the agent controls an ego-vehicle navigating a multi-lane highway populated with other vehicles. The primary objectives are to maintain high speed, avoid collisions, and adhere to safe driving practices.

### **Observation Space**

The default observation type is `Kinematics`, providing a continuous state space that includes features such as position (`x`, `y`), velocity components (`vx`, `vy`), and orientation (`cos_h`, `sin_h`) of the ego-vehicle and surrounding vehicles. This setup allows the agent to perceive its environment effectively.

### **Action Space**

We employ the `DiscreteMetaAction` space, which abstracts high-level driving maneuvers into five discrete actions:

```
0: 'LANE_LEFT'   # Change to the left lane
1: 'IDLE'        # Maintain current speed and lane
2: 'LANE_RIGHT'  # Change to the right lane
3: 'FASTER'      # Increase speed
4: 'SLOWER'      # Decrease speed

```

These meta-actions simplify the control strategy by allowing the agent to focus on strategic decisions rather than low-level controls. 

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

### **Deliverables**
- **Codebase**: A complete implementation of the agent and environment interaction logic using the `highway-env` library. The code will be modular and well-documented for reproducibility.
- **Training Results**: Plots and visualizations of key performance metrics (reward progression, collision rate, average speed) across training episodes.
- **Parameter Exploration**: Experiments comparing the effects of different reward weightings (`a`, `b`), learning algorithms, and environment configurations.
- **Final Report**: A detailed document summarizing methodology, implementation, experiments, results, and analysis.
- **Presentation**: A visual presentation highlighting the project objectives, techniques used, key findings, and demo runs of the trained agent.

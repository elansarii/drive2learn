# 🚗 Autonomous Driving with Reinforcement Learning

This project explores autonomous driving strategies using reinforcement learning in the `highway-fast-v0` simulation environment from the [Highway-env](https://github.com/eleurent/highway-env) library. We evaluate two learning approaches: **Tabular Methods** (with Expected SARSA and Q-Learning) and **Function Approximation** (with Differential SARSA using tile coding).

## 👥 Team Members

* Mohamed Elansari (ID: 202209852)
* Ali Ghazi (ID: 202209865)
* Abdullah Jamali (ID: 202104080)

## 🚦 Environment Setup

* **Environment**: `highway-fast-v0`
* **Actions**: Lane changes, speed control
* **Observation Spaces**:

  * *Tabular*: Time-To-Collision (TTC) grid
  * *Function Approximation*: Continuous kinematic features

## 🧠 Algorithms Used

* **Tabular Methods**:

  * Expected SARSA (best performance)
  * Q-Learning (noisier but comparable)

* **Function Approximation**:

  * Differential SARSA (fast learning, high variance)
  * Tile coding with hashing (200k feature vector, 16 tilings)

## ⚙️ Training Summary

| Parameter         | Tabular | Functional Approx |
| ----------------- | ------- | ----------------- |
| Episodes          | 50,000  | 10,000            |
| Max Steps/Episode | 50      | 50                |
| Learning Rate     | 0.3     | 0.005             |
| Epsilon Decay     | 0.9999  | 0.999             |

## 📊 Results

* **Expected SARSA** achieved the best final performance with smooth and stable learning (avg. reward ≈ 20.04).
* **Differential SARSA** learned faster but suffered from instability and high variance.

## 📌 Key Takeaways

* Tabular methods were more stable but slower to converge.
* Functional approximation offered flexibility but required careful tuning to avoid noisy learning.
* Overall, Expected SARSA was the most reliable algorithm in this setup.

## 🔄 Future Work

* Implement deep RL methods (e.g., DQN, Actor-Critic)
* More advanced hyperparameter tuning
* Explore cooperative multi-agent driving strategies


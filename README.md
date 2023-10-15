# Lunar-Lander

# Deep Q-Learning - Lunar Lander

This project implements the Deep Q-Learning algorithm to train an agent to safely land a lunar lander on a platform on the surface of the moon using the LunarLander simulation environment from OpenAI Gym.

![lunar_lander](https://github.com/alessiopelusi/Lunar-Lander/assets/130958426/79cb563e-5831-4612-b9fe-e7ce2ed7efa8)

# Project Description

The main objective of this project is to train an agent using the Deep Q-Learning algorithm to land a lunar lander on the landing platform. To achieve this goal, several Deep Q-Learning techniques have been employed:

- **Experience Replay:** During training, past experiences are stored in a memory buffer. This buffer is then randomly sampled to train the agent, allowing for better experience management.

- **Q-Network (Neural Network):** To estimate action values, a deep neural network (Deep Q-Network) composed of dense layers is used. The network takes the environment's state as input and produces action value estimates.

- **Target Network:** To enhance training stability, two neural networks are used: the main Q-Network and the target network. The target network is a copy of the Q-Network and is periodically updated softly.

- **Epsilon-Greedy Policy:** During training, the agent follows an epsilon-greedy policy, balancing random exploration (with probability epsilon) with value-maximizing based approaches (with probability 1 - epsilon).

- **Reward Function:** Rewards are based on proximity to the landing platform, velocity, angle, and ground contact. Episodes end if the lander crashes or reaches the platform.

# How to Use

To run the code and train the Lunar Lander agent, follow these steps:


1. Make sure you have installed all the dependencies listed in the "requirements.txt" file.

2. Run the "main.py" file to start agent training.

3. Wait for the agent to complete training. During training, you'll see information on training progress, including scores obtained in each episode.

4. Training is considered complete when the agent achieves an average of 200 points over the last 100 episodes.

# Hyperparameters

- **MEMORY_SIZE:** Size of the experience memory buffer.
- **GAMMA:** Discount factor.
- **ALPHA:** Learning rate.
- **NUM_STEPS_FOR_UPDATE:** Number of learning steps between network updates.

# Neural Network Architecture

The neural network used consists of three dense layers:

- Input Layer: The environment state (observation) is the input to the network.
- Hidden Layers: Two intermediate layers with ReLU activation for feature learning.
- Output Layer: This layer produces action value estimates for each available action.

# Environment Resolution

The environment is considered solved when the agent achieves an average of 200 points over the last 100 episodes.

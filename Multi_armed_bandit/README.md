# Solving the Multi-Armed Bandit (MAD) problem
My examples in python to implement a RL agent for the MAD problem.

##  Supported material
- Medium article explaining the MAD problem, available [here](https://towardsdatascience.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50)
- Book about online learning, with MAD problem as well, available [here](https://arxiv.org/abs/1912.13213)

## Packages and Modules
Standard py.pkg:
```
numpy, pandas, matplotlib, seaborn, scipy
```
No need for special ones.

## Basic examples
- MAD problem with the generic formulation, please run the [script](multi_armed_example.py). It is an example for RL agent using Random, e-Greedy, Thompson-Sampler policies;
- P2P market application as MAD problem, please run the [script](energy_p2p_market.py). The project implements RL_agent to determine the best trading partners in P2P market using Random, epsilon-Greedy and Thompson-Sampler policies;

# Overview
This is the code repository for the following manuscript: ["Improved Bayes Risk Can Yield Reduced Social Welfare Under Competition"](https://arxiv.org/abs/2306.14670) (Meena Jagadeesan, Michael I. Jordan, Jacob Steinhardt, Nika Haghtalab). 

# Paper Abstract
As the scale of machine learning models increases, trends such as scaling laws anticipate consistent downstream improvements in predictive accuracy. However, these trends take the perspective of a single model-provider in isolation, while in reality providers often compete with each other for users. In this work, we demonstrate that competition can fundamentally alter the behavior of these scaling trends, even causing overall predictive accuracy across users to be non-monotonic or decreasing with scale. We define a model of competition for classification tasks, and use data representations as a lens for studying increases in scale. We find many settings where improving data representation quality (as measured by Bayes risk) decreases the overall predictive accuracy across users (i.e., social welfare) for a marketplace of competing model-providers. Our examples range from closed-form formulas in simple settings to simulations with pretrained representations on CIFAR-10. At a conceptual level, our work suggests that favorable scaling trends for individual model-providers need not translate to downstream improvements in social welfare when there is competition.

# Repository Overview

This repository is designed for analyzing game interactions between model providers, focusing on computing Nash equilibria through gradient-based best-response dynamics, evaluating social welfare at equilibrium, and determining the Bayes risk via gradient descent.

## Experimentation Scripts
- `run_synthetic_experiments.py`: Executes experiments on synthetic data.
- `run_cifar_experiments.py`: Executes experiments on CIFAR-10 data.

## Helper Function Files
- `best_response_dynamics.py`: Contains functions aiding in best-response dynamics, Bayes risk calculation, and equilibrium social welfare analysis.
- `cifar_data.py`: Generates CIFAR-10 representations from various models pretrained on ImageNet.
- `cifar_helper.py`: Provides functions for computing equilibria in CIFAR-10 experiments.
- `synthetic_data.py`: Generates synthetic data. 
- `synthetic_helper.py`: Provides functions for computing equilibria in CIFAR-10 experiments.

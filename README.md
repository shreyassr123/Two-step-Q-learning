# Two-Step Q-Learning

This repository contains the Python files to reproduce the results from the paper titled "Two-Step Q-learning". 

## Maximization Example


The first example is analogous to the classic maximization example provided in the book - Reinforcement Learning: An Introduction (Chapter 6, Example 6.1). 

The directory Max_Bias has all the files required to run the algorithms and, consequently, files to compare the algorithms with plots. 

## Random MDPs

The second example comprises of generating a hundred random MDPs using mdptoolbox by [pymdptoolbox] (https://github.com/sawcordwell/pymdptoolbox). More specifically, a hundred random MDPs are generated, each with restriction and without restrictions on the structure of the MDPs. Further, using the runfile.py, the algorithms are compared in terms of average error. 

#### Step 1: Download and Install Python MDP Toolbox from https://pymdptoolbox.readthedocs.io/en/latest/api/mdptoolbox.html.

#### Step 2: Replace the mdp.py and example.py file in original MDP Toolbox with the updated mdp.py and example.py files provided respectively.

#### Step 3: Run runfile_mdp.py for the results.

## Roulette as Multi-Armed Bandit Example

The third example is a classic roulette problem expressed as a single-state multi-armed bandit problem. Once again, the directory roulette as a multi-armed bandit has all the files required to run the algorithms and plot the graphs. 


#### One can refer to the paper for any further details regarding the parameters and convergence of the proposed two-step Q-learning algorithm.





### Acknowledgments

The Max_Bias example and Roulette as a multi-armed bandit example are based on and adapted from [The-Mean-Squared-Error-of-Double-Q-Learning](https://github.com/wentaoweng/The-Mean-Squared-Error-of-Double-Q-Learning). 





Many thanks to the original authors for their contributions.

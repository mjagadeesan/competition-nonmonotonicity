import torch
from torch.nn.modules.module import T
from torch.distributions import MultivariateNormal
from torch.distributions import Laplace
from torch.distributions import Normal
from torch.distributions import Uniform
from torch.distributions import Bernoulli
from torch.distributions import Categorical
from torch.optim import SGD
from torch import nn
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import best_response_dynamics 
import synthetic_data

## Helper functions to compute Bayes optimal and Nash equilibrium
def compute_bayes_optimal_helper(D, sigma, mean0, mean1, probs, temperature=0.1, learning_rate=1, max_iter_loop=10000, max_iter=100, input_dist='noisy', probs_subpopulations=None):
    c = 0.3
    stdev_input = 0.1
    X, Y = synthetic_data.create_distributions_input(D, sigma, mean0, mean1, probs, probs_subpopulations, input_dist)

    return best_response_dynamics.bayes_optimal_compute(X, Y, D, c, temperature, learning_rate=learning_rate, max_iter_loop=max_iter_loop, stdev=stdev_input, bias_flag=False)


def nash_equilibrium_helper(w, num_players, D, sigma, mean0, mean1, probs, temperature=0.1, learning_rate=1e-1, max_iter_loop=5000, tol=1e-3, max_iter=100, input_dist='noisy', probs_subpopulations=None):
    c = 0.3
    input = 'synthetic'
    X, Y = synthetic_data.create_distributions_input(D, sigma, mean0, mean1, probs, probs_subpopulations, input_dist)

    return best_response_dynamics.nash_equilibrium_compute(w, X, Y, num_players, c, temperature, learning_rate=learning_rate, max_iter_loop=max_iter_loop, tol=tol, max_iter=max_iter, stdev_input=0.1, input=input, bias_flag=False, repeat_flag=False, reinitialize=False)


## Compute Nash equilibrium and Bayes optimal for Setting 1
def compute_nash_equilibria_setting1(m, alpha, trial_num=1, D=1):
    w = torch.ones(m)/m
    probs = torch.tensor([alpha, 1-alpha]) # probability on each class
    input_dist='noisybias'

    steps, models_all, payoffs, social_loss_new, payoffs_array, individual_losses = nash_equilibrium_helper(w, m, D, 1, 1, 1, probs, input_dist=input_dist)

    np.savez('./data-synthetic-experiments/nash_alpha-trial-'+str(trial_num) + 'm-' + str(m) + '-alpha-' + str(alpha) + '.npz', social_loss=social_loss_new)

def compute_bayes_optimal_setting1(alpha, trial_num=1, D=1):
    mean0 = 1
    mean1 = -1
    probs = torch.tensor([alpha, 1-alpha]) # probability on each class
    input_dist='noisybias'

    max_iter_loop = 10000
    (model_parameters, opt_loss) = compute_bayes_optimal_helper(D, 1, mean0, mean1, probs, input_dist=input_dist)
    np.savez('./data-synthetic-experiments/bayes_alpha-trial-'+str(trial_num) + '-alpha-' + str(alpha) + '.npz', opt_loss=opt_loss)


## Compute Nash equilibrium and Bayes optimal for Setting 2
def compute_nash_equilibria_setting2(m, sigma, trial_num=1, D=1):
    w = torch.ones(m)/m
    mean0 = 1
    mean1 = -1
    probs = torch.tensor([0.4, 0.6]) # probability on each class
    input_dist='noisy'
    D=2

    steps, models_all, payoffs, social_loss_new, payoffs_array, individual_losses = nash_equilibrium_helper(w, m, D, sigma, mean0, mean1, probs, input_dist=input_dist)

    np.savez('./data-synthetic-experiments/nash_sigma-trial-'+str(trial_num) + 'm-' + str(m) + '-sigma-' + str(sigma) + '.npz', social_loss=social_loss_new)

def compute_bayes_optimal_setting2(sigma, trial_num=1):
    mean0 = 1
    mean1 = -1
    probs = torch.tensor([0.4, 0.6]) # probability on each class
    input_dist='noisy'
    D = 2

    max_iter_loop = 10000
    (model_parameters, opt_loss) = compute_bayes_optimal_helper(D, sigma, mean0, mean1, probs, input_dist=input_dist)
    np.savez('./data-synthetic-experiments/bayes_sigma-trial-'+str(trial_num) + '-sigma-' + str(sigma) + '.npz', opt_loss=opt_loss)

## Compute Nash equilibrium and Bayes optimal for Setting 3
def compute_nash_equilibria_setting3(m, D, trial_num=1):
    D = int(D)
    w = torch.ones(m)
    mean0 = -1
    mean1 = 1
    sigma = 1.0
    probs = torch.tensor([0.5, 0.5, 0.5, 0.5])
    input_dist='subpopulations'
    probs_subpopulations =  torch.tensor([0.7, 0.15, 0.1, 0.05])

    steps, models_all, payoffs, social_loss_new, payoffs_array, individual_losses = nash_equilibrium_helper(w, m, D, sigma, mean0, mean1, probs, input_dist=input_dist, probs_subpopulations=probs_subpopulations)

    np.savez('./data-synthetic-experiments/nash_Dreal-trial-'+str(trial_num) + 'm-' + str(m) + '-D-' + str(D) + '.npz', social_loss=social_loss_new)

def compute_bayes_optimal_setting3(D, trial_num=1):
    D = int(D)
    mean0 = -1
    mean1 = 1
    sigma = 1.0
    probs = torch.tensor([0.5, 0.5, 0.5, 0.5]) 
    input_dist='subpopulations'
    probs_subpopulations =  torch.tensor([0.7, 0.15, 0.1, 0.05])

    max_iter_loop = 10000
    (model_parameters, opt_loss) = compute_bayes_optimal_helper(D, sigma, mean0, mean1, probs, input_dist=input_dist, probs_subpopulations=probs_subpopulations)
    np.savez('./data-synthetic-experiments/bayes_Dreal-trial-'+str(trial_num) + '-D-' + str(D) + '.npz', opt_loss=opt_loss)


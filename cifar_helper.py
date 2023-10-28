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

def load():
    model_names = ['resnet18', 'alexnet', 'vgg16', 'resnet34', 'resnet50']

    X_all = {model_name: [] for model_name in model_names}

    for model_name in model_names:
      # Load features_all_models from the .npz file
        save_path = "./data-cifar/features_test_train_" + str(model_name) + ".npy"
        features = np.load(save_path, allow_pickle=True)
        X_all[model_name] = torch.tensor(features,dtype=torch.float)[0:50000]

    print(X_all['alexnet'].shape)
    print(X_all['vgg16'].shape)
    print(X_all['resnet18'].shape)
    print(X_all['resnet34'].shape)
    print(X_all['resnet50'].shape)

    # Load 10 class labels
    labels_save_path = "./data-cifar/labels_test_train_all_classes.npy"
    loaded_labels = np.load(labels_save_path)
    Ys_old = torch.tensor(loaded_labels, dtype=torch.float).unsqueeze(1).float()[0:50000]
    Y_multiclass = torch.zeros(50000, 10)
    for i in range(50000):
        index = int(Ys_old[i])
        Y_multiclass[i][index] = 1

    # Load binary classification labels
    labels_save_path = "./data-cifar/labels_test_train.npy"
    loaded_labels = np.load(labels_save_path)
    Y_binary = torch.tensor(loaded_labels, dtype=torch.float).unsqueeze(1).float()[0:50000]
    return X_all, Y_multiclass, Y_binary

# Compute Bayes optimal 

def compute_bayes_optimal_multiclass(trial_num=1):
    X_all, Y_multiclass, Y_binary = load()
    losses = []
    for model in ['alexnet', 'vgg16', 'resnet18', 'resnet34', 'resnet50']:
        X = X_all[model]
        Y = Y_multiclass
        print(X.shape)
        D = X.shape[1]
        c = 0.1
        temperature = 1.0
        max_iter_loop = 70000
        learning_rate = 0.1

        (parameters, loss) = best_response_dynamics.bayes_optimal_compute(X, Y, D, c, temperature, learning_rate=learning_rate, max_iter_loop=max_iter_loop, tol=1e-10, max_iter=100, output_dim=10)
        losses.append(loss)
        print(f"opt loss: {loss}")
    print(losses)
    np.savez('./data-cifar-experiments/bayes_optimal_all_classes' + '-trial' + str(trial_num) + '.npz', model_names=model_names_to_iterate, losses=losses)

    
# Compute Nash equilibria

def compute_nash_equilibria_multiclass(num_players, trial_num=1):
    X_all, Y_multiclass, Y_binary = load()
    w = torch.ones(num_players)/num_players
    print(torch.sum(w))
    social_losses = []
    for model_name in ['alexnet', 'vgg16', 'resnet18', 'resnet34', 'resnet50']:
        X = X_all[model_name]
        Y = Y_multiclass
        print(f"Model name: {model_name}")
        D = X.shape[1]
        c = 0.1
        temperature = 1.0
        max_iter_loop = 2000
        learning_rate = 1.0
        input = 'cifar'

        learning_rate_schedule = []

        stdev_input=0.5

        learning_rate_schedule.append((0.5, 5.0))
        learning_rate_schedule.append((0.4, 15.0))
        learning_rate_schedule.append((0.3, 20.0))

        i, models_all, payoffs, social_loss_new, payoffs_array, individual_losses = best_response_dynamics.nash_equilibrium_compute(w, X, Y, num_players, c, temperature, output_dim=10, learning_rate=learning_rate, max_iter_loop=max_iter_loop, tol=1e-3, max_iter=100, input=input, learning_rate_schedule=learning_rate_schedule, stdev_input=stdev_input)
        social_losses.append(social_loss_new)
    print(social_losses)
    np.savez('./data-cifar-experiments/equilibrium-multiclass' + '-trial' + str(trial_num) + '-m-' + str(num_players) + '.npz', model_names=model_names_to_iterate, social_losses=social_losses)
    
# Compute Bayes optimal

def compute_bayes_optimal_binary(trial_num=1):
    X_all, Y_multiclass, Y_binary = load()
    losses = []
    for model in ['alexnet', 'vgg16', 'resnet18', 'resnet34', 'resnet50']:
        X = X_all[model]
        Y = Y_binary
        print(X.shape)
        D = X.shape[1]
        c = 0.1
        temperature = 1.0
        max_iter_loop = 50000
        learning_rate = 0.1

        (parameters, loss) = best_response_dynamics.bayes_optimal_compute(X, Y, D, c, temperature, learning_rate=learning_rate, max_iter_loop=max_iter_loop, max_iter=100)
        losses.append(loss)
        print(f"opt loss: {loss}")

    print(losses)

    np.savez('./data-cifar-experiments/bayes_optimal' + '-trial' + str(trial_num) + '.npz', model_names=model_names_to_iterate, losses=losses)

    
model_names_to_iterate = ['alexnet', 'vgg16', 'resnet18', 'resnet34', 'resnet50']

def compute_nash_equilibria_binary(num_players, trial_num=1):
    X_all, Y_multiclass, Y_binary = load()
    w = torch.ones(num_players)/num_players
    print(torch.sum(w))
    social_losses = []
    for model_name in ['alexnet', 'vgg16', 'resnet18', 'resnet34', 'resnet50']:
        X = X_all[model_name]
        Y = Y_binary
        print(f"Model name: {model_name}")
        D = X.shape[1]
        c = 0.1
        temperature = 1.0

        input = 'cifar'
        
        max_iter_loop = 2000
        stdev_input=0.5
        
        learning_rate = 1.0
        learning_rate_schedule = []

        
        learning_rate_schedule.append((0.5, 5.0))
        learning_rate_schedule.append((0.4, 15.0))
        learning_rate_schedule.append((0.3, 20.0))

        i, models_all, payoffs, social_loss_new, payoffs_array, individual_losses = best_response_dynamics.nash_equilibrium_compute(w, X, Y, num_players, c, temperature, stdev_input=stdev_input, max_iter_loop=max_iter_loop, tol=1e-3, max_iter=100, input=input, output_dim=1, learning_rate=learning_rate, learning_rate_schedule=learning_rate_schedule)
        social_losses.append(social_loss_new)
    print(social_losses)
    np.savez('./data-cifar-experiments/equilibrium' + '-trial' + str(trial_num) + '-m-' + str(num_players) + '.npz', model_names=model_names_to_iterate, social_losses=social_losses)
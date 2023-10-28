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


def loss_ell_1(predictions, Y):
    probabilities = torch.sigmoid(predictions)
    return torch.abs(Y - probabilities)

def loss_ell_1_multi_class(predictions, Y):
    probabilities = torch.softmax(predictions, dim=1)
    losses = torch.sum(torch.abs(Y - probabilities),dim=1) /2
    return losses

class OneLayerLinear(nn.Module):
    def __init__(self, input_dim, classes=1, biasflag=True, stdev_input=0.005):
        super(OneLayerLinear, self).__init__()
        self.stdev_input = stdev_input
        self.layer1 = torch.nn.Linear(input_dim, classes, bias=biasflag)
        self.to(device)

    def forward(self, x):
        x = self.layer1(x)
        return x

    def initialize_weights(self):
        torch.nn.init.normal_(self.layer1.weight, mean=0.0, std=self.stdev_input)
        


# Compute Bayes Optimal 
def loss_single(model, X, Y, c, temperature, output_dim = 1):
    if output_dim == 1:
        loss_all = loss_ell_1(model(X) / temperature, Y)
    else:
        loss_all = loss_ell_1_multi_class(model(X) / temperature, Y)
    return torch.mean(loss_all)


def bayes_optimal_compute(X, Y, D, c, temperature, learning_rate=1e-2, max_iter_loop=50000, tol=1e-15, max_iter=100, stdev=0.005, output_dim=1, bias_flag=True):

    X = X.to(device)
    Y = Y.to(device)
    model = None
    model = OneLayerLinear(D, output_dim, biasflag=bias_flag, stdev_input=stdev).to(device)
    model.initialize_weights()

    optimizer = SGD(model.parameters(), lr=learning_rate)

    loss_vals = []
    steps = max_iter_loop
    grad_norm = 0.0
    for i in range(max_iter_loop):
        optimizer.zero_grad()
        loss = loss_single(model, X, Y, c, temperature, output_dim=output_dim)
        loss_vals.append(loss.item())
        loss.backward()
        optimizer.step()

    print(f"Gradient norm: {grad_norm}")

    # Get test accuracy:
    if output_dim == 1:
        Y_pred = torch.zeros(X.shape[0], 1).to(device)
        probabilities = torch.sigmoid(model(X) / temperature)
        for i in range(X.shape[0]):
            if probabilities[i] > 0.5:
                Y_pred[i] = 1
        print(f"test accuracy: {torch.mean(torch.abs(Y- Y_pred))}")

    return (model.parameters(), 1 * loss.detach().cpu().numpy())


# Best response dynamics 


## Compute payoff 
def payoff_multi(j, num_players, models_all, w, X, Y, c, temperature, output_dim=1):
    loss_all = [None] * num_players
    for i in range(num_players):
        if output_dim == 1:
            loss_all[i] = loss_ell_1(models_all[i](X) / temperature, Y)
        else:
            loss_all[i] = loss_ell_1_multi_class(models_all[i](X) / temperature, Y)
    # the rest of the function continues as before
    shifted_loss_all = [None] * num_players
    for i in range(num_players):
        shifted_loss_all[i] = torch.log(w[i]) - loss_all[i] / c

    log_p_all = torch.stack(shifted_loss_all, dim=0)
    logsumexp_pall = torch.logsumexp(log_p_all, dim=0)
    log_pj_minus_logsumexp = shifted_loss_all[j] - logsumexp_pall
    probabilities = torch.exp(log_pj_minus_logsumexp)
    return torch.mean(probabilities)

## Compute best response 
def best_response_multi(j, num_players, models_all, w, X, Y, c, temperature, learning_rate=1e-1, tol=1e-11, max_iter=10000, output_dim=1, initialize_flag=True):

    # Set requires_grad to be True for only model j
    for i in range(num_players):
        if i != j:
            for param in models_all[i].parameters():
                param.requires_grad = False
        else:
            for param in models_all[i].parameters():
                param.requires_grad = True
            if initialize_flag:
                models_all[j].initialize_weights()

    optimizer = SGD(models_all[j].parameters(), lr=learning_rate)

    loss_vals = []

    steps = max_iter
    grad_norm = 0.0
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = -1 * payoff_multi(j, num_players, models_all, w, X, Y, c, temperature, output_dim=output_dim)
        loss_vals.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f"Gradient norm: {grad_norm}")

    return (models_all, -1 * loss.detach().cpu().numpy())

## Compute social loss 
def social_loss_multi(num_players, models_all, w, X, Y, c, temperature, output_dim=1):
    loss_all = [None] * num_players

    for i in range(num_players):
        if output_dim == 1:
            loss_all[i] = loss_ell_1(models_all[i](X) / temperature, Y)
        else:
            loss_all[i] = loss_ell_1_multi_class(models_all[i](X) / temperature, Y)


    shifted_loss_all = [None] * num_players
    for i in range(num_players):
        shifted_loss_all[i] = torch.log(w[i]) - loss_all[i] / c

    log_p_all = torch.stack(shifted_loss_all, dim=0)
    logsumexp_pall = torch.logsumexp(log_p_all, dim=0)
    social_loss_val = 0
    payoffs = []
    individual_losses = []
    for i in range(num_players):
        log_pi_minus_logsumexp = shifted_loss_all[i] - logsumexp_pall
        probabilities = torch.exp(log_pi_minus_logsumexp)
        payoffs.append(torch.mean(probabilities).item())
        social_loss_val += torch.matmul(probabilities.T, loss_all[i])
        individual_losses.append(loss_single(models_all[i], X, Y, c, temperature, output_dim=output_dim).item())
    print(f"individual losses: {individual_losses}")
    social_loss_val = social_loss_val / X.shape[0]
    return social_loss_val.item(), payoffs, individual_losses


# w = market tiebreaking for each model-provider
# Xs = representations for each model-provider
# Y = labels
# c = noise in user responses
# learning_rate_schedule = schedule of learning rates
# temperature = temperature parameter in loss function
# bias_flag = include bias in linear network
# repeat_flag = run best response multiple times if the loss is high
def nash_equilibrium_compute(w, X, Y, num_players, c, temperature, learning_rate=1e-2, max_iter_loop=50000, tol=1e-2, max_iter=100, input='synthetic', learning_rate_schedule=None, stdev_input=0.005, output_dim=1, bias_flag=True, repeat_flag=True, reinitialize=False):
    X = X.to(device)
    Y = Y.to(device)

    models_all = []

    for i in range(num_players):
        D = X.shape[1]
        model = OneLayerLinear(D, output_dim, biasflag=bias_flag, stdev_input=stdev_input).to(device)
        model.initialize_weights()
        models_all.append(model)

    lr = learning_rate
    payoffs = np.zeros((num_players, max_iter))
    per_player_losses = 0.5 * np.ones(num_players)
    for i in range(max_iter):
        flag_finished = True
        for j in range(num_players):
            initialize_flag = True

            # Only reinitialize if the loss is very poor
            if per_player_losses[j] < 0.3 and input == 'cifar' and output_dim == 1:
                initialize_flag = False
            elif per_player_losses[j] < 0.7 and input == 'cifar' and output_dim == 10:
                initialize_flag = False
            elif per_player_losses[j] < 2 and input == 'synthetic':
                initialize_flag = False

            # Change learning rate over time
            if learning_rate_schedule != None:
                for (loss, learning_rate_new) in learning_rate_schedule:
                    if per_player_losses[j] < loss:
                        lr = learning_rate_new

            (models_all, payoff_new) = best_response_multi(j, num_players, models_all, w, X, Y, c, temperature, learning_rate=lr, max_iter=max_iter_loop, initialize_flag=initialize_flag, output_dim=output_dim)
            payoffs[j][i] = payoff_new
            social_loss_new, payoffs_array, individual_losses = social_loss_multi(num_players, models_all, w, X, Y, c, temperature, output_dim=output_dim)
            per_player_losses = individual_losses

            # Run the best response a couple more times if the loss is very high
            if repeat_flag==True:
                for _ in range(2):
                    if per_player_losses[j] > 0.3 and output_dim == 1:
                        print(f"Repeating best response, since loss is high: {per_player_losses[j]}")
                        (models_all, payoff_new) = best_response_multi(j, num_players, models_all, w, X, Y, c, temperature, learning_rate=lr, max_iter=max_iter_loop, initialize_flag=initialize_flag, output_dim=output_dim)
                        payoffs[j][i] = payoff_new
                        social_loss_new, payoffs_array, individual_losses = social_loss_multi(num_players, models_all, w, X, Y, c, temperature, output_dim=output_dim)
                        per_player_losses = individual_losses

            if np.abs(payoff_new-payoffs[j][i - 1]) > tol:
                flag_finished = False

            print(f"Player: {j}, Payoff: {payoffs_array}, Social Loss: {social_loss_new}")

        if flag_finished:
            return i, models_all, payoffs, social_loss_new, payoffs_array, individual_losses


    raise Exception("Nash equilibrium not found within max_iter iterations")

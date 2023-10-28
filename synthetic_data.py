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

# Create synthetic data, each with 1000 data points. 
# Since we need to handle setups where the representations are 0-dimensional, we put the bias parameter into the representations and fit linear models without bias. That is, instead of creating D-dimensional representations and using bias, we instead create (D+1)-dimensional features and do not use bias.


# Setting 1: synthetic distribution with 1000 points where the only dimension corresponds to bias
def create_distributions(D, probs):
    assert D == 1, "this function is intended to be used for dimension D = 1"
    assert torch.sum(probs) == 1, "probabilities in distribution must sum to 1"

    X = torch.empty((10000,D), dtype=torch.float)
    Y = torch.empty((10000,1), dtype=torch.float)
    means = torch.ones(D, dtype=torch.float)
    covariance = torch.eye(D)
    categorical = Categorical(probs)
    for i in range(10000):
        component = categorical.sample()
        if component == 0:
            X[i] = means
            Y[i] = 0.0
        else:
            X[i] = means
            Y[i] = 1.0
    return X, Y

# Setting 2: synthetic distribution with 10000 samples
def create_distributions_bias( D, sigma, mean0, mean1, probs):
    assert torch.sum(probs) == 1, "probabilities in distribution must sum to 1"

    S = 10000
    X_real = torch.empty((S,D - 1), dtype=torch.float)
    X = torch.empty((S,D), dtype=torch.float)
    Y = torch.empty((S,1), dtype=torch.float)
    means0 = mean0 * torch.ones(D-1, dtype=torch.float)
    means1 = mean1 * torch.ones(D-1, dtype=torch.float)
    covariance = torch.eye(D-1)
    for d in range(D-1):
        covariance[d][d] = (sigma * sigma)
    dist0 = MultivariateNormal(means0, covariance)
    dist1 = MultivariateNormal(means1, covariance)
    categorical = Categorical(probs)
    for i in range(S):
        component = categorical.sample()
        if component == 0:
            X_real[i] = dist0.sample()
            Y[i] = 0.0
        else:
            X_real[i] = dist1.sample()
            Y[i] = 1.0

        for d in range(D - 1):
            X[i][d] = X_real[i][d]
        X[i][D-1] = 1 # set last feature = 1 to add a bias term

    return X, Y

# Setting 3: Mixture of synthetic distributions across num_subpopulations subpopulations. Subpopulation i has the first i coordinates equal to 0. 
def create_distributions_subpopulations(D, sigma, mean0, mean1, probs_classes, probs_subpopulations, num_subpopulations=4):
    assert torch.sum(probs_subpopulations) == 1, "probabilities in subpopulations must sum to 1"
    assert probs_subpopulations.size(dim=0) == num_subpopulations, "subpopulations vector does not have dimension num_subpopulations: " + str(num_subpopulations)
    assert len(probs_classes) == num_subpopulations, "classes vector does not have dimension num_subpopulations: " + str(num_subpopulations)
    assert num_subpopulations + 1  >= D, "classes vector does not have dimension num_subpopulations: " + str(num_subpopulations)

    print(num_subpopulations)

    X = torch.empty((10000,D), dtype=torch.float)
    Y = torch.empty((10000,1), dtype=torch.float)

    X_real = torch.empty((10000,num_subpopulations), dtype=torch.float) # 

    # set up categoricals for subpopulations
    categorical_subpopulation = Categorical(probs_subpopulations)

    # set up categories for each class for each subpopulation
    distribution_classes = []
    for i in range(num_subpopulations):
        categorical_classes = Categorical(torch.tensor([probs_classes[i], 1-probs_classes[i]]))
        distribution_classes.append(categorical_classes)

    # Create the distribution for each subpopulation and classes 0 and 1
    distributions_zero = []
    distributions_ones = []
    for i in range(num_subpopulations):
        # Zero out the first i coordinates 
        mean_vector = np.ones(num_subpopulations)
        for j in range(i):
            mean_vector[j] = 0

        means0 = mean0 * torch.tensor(mean_vector, dtype=torch.float)
        means1 = mean1 * torch.tensor(mean_vector, dtype=torch.float)
        covariance = torch.eye(num_subpopulations)
        for d in range(num_subpopulations):
            covariance[d][d] = (sigma * sigma)

        dist0 = MultivariateNormal(means0, covariance)
        dist1 = MultivariateNormal(means1, covariance)

        distributions_zero.append(dist0)
        distributions_ones.append(dist1)

    # Sample from each subpopulation and each class
    for i in range(10000):
        component_subpopulation = categorical_subpopulation.sample()
        component_classes = distribution_classes[component_subpopulation].sample()

        if component_classes == 0:
            X_real[i] = distributions_zero[component_subpopulation].sample()
            Y[i] = 0.0
        else:
            X_real[i] = distributions_ones[component_subpopulation].sample()
            Y[i] = 1.0

        # set features for first D dimensions
        for d in range(D - 1):
            X[i][d] = X_real[i][d]
        X[i][D-1] = 1 # set last feature = 1 to add a bias term

    return X, Y


## Add a function that references all of these functions
def create_distributions_input(D, sigma, mean0, mean1, probs, probs_subpopulations, input_dist):
    if input_dist == 'noisybias':
        X, Y = create_distributions(D, probs)
    elif input_dist == 'noisy':
        X, Y = create_distributions_bias(D, sigma, mean0, mean1,probs)
    elif input_dist == 'subpopulations':
        X, Y = create_distributions_subpopulations(D, sigma, mean0, mean1, probs, probs_subpopulations)
    else:
        raise ValueError("Invalid input distribution value")
    return X, Y

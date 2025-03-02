import numpy as np

def repartition_function(x, theta):
    mu = theta[0]
    sigma = theta[1]
    xi = theta[2]
    arg = 1 + xi * (x - mu)/sigma
    arg[arg <0] = 0
    return np.exp(-(arg**(-1/xi)))

def density(x, theta):
    mu = theta[0]
    sigma = theta[1]
    xi = theta[2]
    arg = 1 + xi * (x - mu) / sigma
    arg[arg < 0] = 0
    return (1/sigma)*(arg**(-1/(xi-1))) * repartition_function(x, theta)

#def single_evaluation(x, theta, data, p):
    # consider theta a matrix with three columns(mu, sigma, theta)
    # consider p a vector of the same length as the columns of theta
    # consider data a vector
    # the likelihood can be calculated with a sample np.prod(density(data, theta[i]))
    # we should find a matricial way to express this calculation


data = np.load("numpy_corsica.npy")

# We have to compute the sup and the inf for the different values of x
# The problem is to find a smart, precise and computational light way for doing it

# then reverse and get the quantiles

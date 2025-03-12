import numpy as np

def repartition_function1(x, theta):
    # theta must be p x 3
    # x must be a number or a simple array
    mu = theta[:, 0]
    sigma = theta[:, 1]
    xi = theta[:, 2]
    if isinstance(x,np.ndarray):
        arg = 1 + xi[:, np.newaxis] * (x[np.newaxis, : ] - mu[:, np.newaxis])/sigma[:, np.newaxis]
        arg[arg < 0] = 0
        arg = np.power(arg, -1/xi[:, np.newaxis], where  = arg > 0)
    else:
        arg = 1 + xi * (x - mu) / sigma
        arg[arg <0] = 0
        arg = np.power(arg, -1/xi,  where  = arg > 0)
    return np.exp(-arg)
    # output of size shape of theta x shape of x


def density1(x, theta):

    mu = theta[:, 0]
    sigma = theta[:, 1]
    xi = theta[:, 2]

    if isinstance(x, np.ndarray):
        arg = 1 + xi[:, np.newaxis] * (x[np.newaxis: 1] - mu[:, np.newaxis]) / sigma[:, np.newaxis]
        arg[arg < 0] = 0
        arg = np.power(arg, -1/xi[:, np.newaxis],  where  = arg > 0)
        return (1/sigma[:, np.newaxis])*arg * repartition_function(x, theta)
    else:
        arg = 1 + xi * (x - mu) / sigma
        arg[arg < 0] = 0
        arg = np.power(arg, -1 / xi,  where  = arg > 0)
    return (1/sigma)*arg * repartition_function(x, theta)


def single_evaluation(x, theta, data, p):
    # x is an array of size m
    # theta is a matrix of size p x 3
    # data is a vector of size n
    # p is a vector of size p
    likelihood = np.prod(density(data, theta) , axis = 1) # size p
    like_p = likelihood*p # size p
    numerator = repartition_function(x, theta) * like_p[:, np.newaxis] # size p x m
    if isinstance(x, np.ndarray):
        return np.sum(numerator, axis = 0) / np.sum(like_p)
    else:
        return np.sum(numerator) / np.sum(like_p)


def constraint(quantiles, theta, p):
    return np.sum(repartition_function(quantiles, theta)*p[:, np.newaxis], axis = 0)

def repartition_function(x, theta):
    # theta must be p x 3
    # x must be a number or a simple array
    x = np.atleast_1d(x)
    mu = theta[..., 0]
    sigma = theta[..., 1]
    xi = theta[..., 2]

    arg = 1 + xi * (x - mu) / sigma
    arg[arg < 0] = 0
    arg = np.power(arg, -1/xi, where=arg > 0)

    return np.exp(-arg)

def density(x, theta):
    x = np.atleast_1d(x)
    mu = theta[..., 0]
    sigma = theta[..., 1]
    xi = theta[..., 2]

    arg = 1 + xi * (x - mu) / sigma
    arg[arg < 0] = 0
    arg = np.power(arg, -1/xi - 1, where=arg > 0)
    return (1 / sigma) * arg * repartition_function(x, theta)

def likelihood(data, theta):
    return np.prod(density(data, theta))
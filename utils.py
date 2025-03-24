import numpy as np


def constraint(quantiles, theta, p):
    return np.sum(repartition_function(quantiles, theta)*p[:, np.newaxis], axis = 0)

def repartition_function(x, theta):
    # theta must be p x 3
    # x must be a number or a simple array of size n
    x = np.atleast_1d(x)
    theta = np.atleast_2d(theta)
    print(theta.shape)
    mu = theta[..., 0]
    sigma = theta[..., 1]
    xi = theta[..., 2]

    arg = 1 + xi[np.newaxis,: ] * (x[:, np.newaxis] - mu[np.newaxis,:]) / sigma[np.newaxis,:]

    arg = np.clip(arg, 1e-10, None)
    neg = (arg == 1e-10)

    with np.errstate(divide='ignore', invalid='ignore'):
        arg = np.power(arg, -1 / xi[np.newaxis, :], where=arg > 0)
    exp = np.exp(-arg)
    pos_xi = (xi > 0)[np.newaxis,:]

    exp[neg & pos_xi] = 0.
    exp[neg & ~pos_xi] = 1.
    #outputs of size of n x p.

    return exp

def density(x, theta):
    # theta must be p x 3
    # x must be a number or a simple array of size n
    x = np.atleast_1d(x)
    mu = theta[..., 0]
    sigma = theta[..., 1]
    xi = theta[..., 2]

    arg = 1 + xi[np.newaxis,:] * (x[: , np.newaxis] - mu[np.newaxis,:]) / sigma[np.newaxis,:]
    arg = np.clip(arg, 1e-10, None)
    neg = arg <= 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = np.power(arg, -1 / xi[np.newaxis, :] -1 , where=arg > 0)

    #shape n x p
    output = (1 / sigma[np.newaxis,:]) * arg * repartition_function(x, theta)
    output[neg] = 0.
    return output


def likelihood(data, theta):

    return np.prod(density(data, theta), axis = 0)

def single_evaluation(x, theta, data, p):
    # x is an array of size m
    # theta is a matrix of size p x 3
    # data is a vector of size n
    # p is a vector of size p
    like = likelihood(data, theta) # size p

    like_p = like*p # size p

    numerator = repartition_function(x, theta) * like_p[np.newaxis, :] #size m x p

    if isinstance(x, np.ndarray):
        return np.sum(numerator, axis = 1) / np.sum(like_p)
    else:
        return np.sum(numerator) / np.sum(like_p)


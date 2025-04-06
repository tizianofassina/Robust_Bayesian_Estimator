import numpy as np


def constraint(quantiles, theta, p):
    return np.sum(repartition_function(quantiles, theta)*p[:, np.newaxis], axis = 0)

def repartition_function(x, theta):
    # theta must be p x 3
    # x must be a number or a simple array of size n
    x = np.atleast_1d(x)
    theta = np.atleast_2d(theta)

    mu = theta[..., 0]
    sigma = theta[..., 1]
    xi = theta[..., 2]

    epsilon = 1e-10
    sigma_safe = np.clip(sigma, epsilon, None)

    arg = 1 + xi[np.newaxis, :] * (x[:, np.newaxis] - mu[np.newaxis, :]) / sigma_safe[np.newaxis, :]

    arg = np.clip(arg, epsilon, None)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        power_exponent = -1 / xi[np.newaxis, :]
        log_arg = np.where(arg > epsilon, np.power(arg, power_exponent), 0)

    exp_term = np.exp(- log_arg)

    pos_xi = (xi > 0)[np.newaxis,:]

    exp_term = np.where((arg == epsilon) & pos_xi, 0., exp_term)
    exp_term = np.where((arg == epsilon) & (~pos_xi), 1., exp_term)
    #outputs of size of n x p.

    return exp_term

def density(x, theta):
    """
        Computes the density.

        Parameters:
            data: The input data of size n.
            theta: The parameters used to compute the likelihood of size p x 3.

        Returns:
            The density of all combinations, size n x p.
        """
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
    """
    Computes the likelihood by first computing the log likelihood
    and then returning its exponentiated value.

    Parameters:
        data: The input data.
        theta: The parameters used to compute the likelihood.

    Returns:
        The likelihood value computed as np.exp(log_like).
    """
    dens = density(data, theta)
    if np.any(np.isnan(dens)):
        print("Warning: density contains NaN values")
    epsilon = 1e-10
    log_dens = np.log(dens + epsilon)

    log_like = np.sum(log_dens, axis=0)

    return np.exp(log_like)

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


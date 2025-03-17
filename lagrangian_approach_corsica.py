import numpy as np
from utils import repartition_function, single_evaluation
from scipy.optimize import minimize
from scipy.optimize import minimize
from functools import partial


corsica_data = np.load("data_meuse_corsica/numpy_corsica.npy")
corsica_data_quantile = np.load("data_meuse_corsica/numpy_corsica_quantile.npy")


length_theta = 3
n = corsica_data_quantile.shape[0]

quantiles = corsica_data_quantile[:,1]
alphas =  corsica_data_quantile[:,0]

x = 10

number_candidates = 1000

def possible_values(n, length_theta, sigma, number_candidates, quantiles, alphas):

    possible_values = []
    matrix = np.zeros((n + 1, n + 1))
    matrix[-1, :] = 1

    while len(possible_values) < number_candidates:
        theta = np.zeros((n+1)*length_theta)
        theta[0::3] = np.random.randn(n + 1)*sigma + np.mean(corsica_data)
        theta[1::3] = np.random.uniform(0,10, size = n+1)
        theta[2::3] = np.random.uniform(-100, 100, size = n+1)

        for i in range(n+1):


            column = repartition_function(quantiles, theta[i*length_theta:(i+1)*length_theta])

            matrix[:-1,i] = column

        if np.abs(np.linalg.det(matrix)) > 1e-10:

            p = np.linalg.solve(matrix, np.append(alphas, 1))
            print(alphas)

            if all(p>=0):
                possible_values.append(np.concatenate((theta, p))
)
            else:
                pass
        else:
            pass


    return np.array(possible_values)



sigma = 4

number_candidates = 1000000
quantiles = corsica_data_quantile[:, 1]
alphas = corsica_data_quantile[:, 0]

results = possible_values(n, length_theta, sigma, number_candidates, quantiles, alphas)

#for result in results:
 #   print(np.sum(result[1]))
#print(results)




def best_values(x, data, possible_values, length_theta, number_best, n ):
    values = []
    for element in possible_values:
        p = element[-(n+1):]
        theta = np.reshape(element[:-(n+1)], shape = (n+1, length_theta))
        values.append(single_evaluation(x, theta, data, p))

    top_k_indices = np.argpartition(np.array(values), -number_best)[-number_best:]  # Get top k indices (unordered)
    top_k_indices = top_k_indices[np.argsort(values[top_k_indices])[::-1]]
    return possible_values[top_k_indices]



def function_to_optimize(x, theta_p, data):
    p = theta_p[-(n + 1):]
    theta = np.reshape(theta_p[:-(n + 1)], shape=(n + 1, length_theta))
    return single_evaluation(x, theta, data, p)

def constraint_f(theta_p):
    matrix = np.zeros(n,n+1)
    for i in range(n+1):
        column = repartition_function(quantiles, theta_p[i*length_theta:(i+1)*length_theta])

        matrix[:,i] = column
    return column - alphas

constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x[-(n + 1):]) - 1},
    {'type': 'ineq', 'fun': lambda x: x[-(n + 1):]},
    {'type': 'eq', 'fun': lambda x: constraint_f(x)},
    {'type': 'ineq', 'fun' : lambda x: x[:-(n + 1)][1::3] }
]

def optimization(function, best_values):
    result = []
    for element in best_values :
        result.append(minimize( function, element, method='trust-constr', constraints=constraints, options={'verbose': 1}).fun)

    return max(result)



def sup_F(x, length_theta, sigma, number_candidates, quantiles, alphas, n, data, number_best ):
    candidate_values = possible_values(n, length_theta, sigma, number_candidates, quantiles, alphas)
    starting_points = best_values(x, data, candidate_values, length_theta, number_best, n )
    objective_with_constants = partial(-1*function_to_optimize, constant1 = x,constant2 = data, constant3 = sup)
    return -optimization(starting_points)

def inf_F(x, length_theta, sigma, number_candidates, quantiles, alphas, n, data, number_best ):
    candidate_values = possible_values(n, length_theta, sigma, number_candidates, quantiles, alphas)
    starting_points = best_values(x, data, candidate_values, length_theta, number_best, n )
    objective_with_constants = partial(function_to_optimize, constant1 = x,constant2 = data)

    return optimization(starting_points)




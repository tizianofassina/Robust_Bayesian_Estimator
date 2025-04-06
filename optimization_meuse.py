import numpy as np
from utils import repartition_function, single_evaluation, likelihood
from scipy.optimize import minimize
from functools import partial
import os
import pickle


meuse_data = np.load("data_meuse_corsica/numpy_meuse.npy")
meuse_data_quantile = np.load("data_meuse_corsica/numpy_meuse_quantile.npy")



length_theta = 3


n = meuse_data_quantile.shape[0]


quantiles = meuse_data_quantile[:,1]


alphas =  meuse_data_quantile[:,0]


def possible_values(n, length_theta, number_candidates, quantiles, alphas, std=10.):
    i = 0
    possible_values = []
    matrix = np.zeros((n + 1, n + 1))
    matrix[-1, :] = 1.

    std = np.std(meuse_data)/2

    while len(possible_values) < number_candidates:
        theta = np.zeros(((n + 1), length_theta))

        mu = np.random.randn(n + 1) * std + np.mean(meuse_data)
        sigma = np.random.lognormal(mean=np.log(std), sigma=0.5, size=n+1)
        xi = np.random.uniform(-20, 20, size=n + 1)

        theta[:, 0] = mu
        theta[:, 1] = sigma
        theta[:, 2] = xi

        repartition = repartition_function(quantiles, theta)

        matrix[:-1, :] = repartition


        if np.abs(np.linalg.det(matrix)) > 1e-10:

            p = np.linalg.solve(matrix, np.append(alphas, 1))
            if all(p >= 0):
                possible_values.append((theta, p))

            else:
                pass
        else:
            pass

    if os.path.exists("possible_values_meuse.pkl"):
        with open("possible_values_meuse.pkl", "rb") as f:
            existing_values = pickle.load(f)

        updated_values = existing_values + possible_values

        with open("possible_values_meuse.pkl", "wb") as f:
            pickle.dump(updated_values, f)
    else:
        with open("possible_values_meuse.pkl", "wb") as f:
            pickle.dump(possible_values, f)

    print(f"Added {number_candidates} new possible candidates")

    return 0


def best_values_for_sup(x, data, possible_values, number_best):
    values = []
    valid_possible_values = []

    for element in possible_values:
        theta = element[0]
        p = element[1]
        value = single_evaluation(x, theta, data, p)

        if not np.isnan(value) and value != 0:
            values.append(value)
            valid_possible_values.append(element)

    values = np.array(values)

    top_k_indices = np.argpartition(values, -number_best)[-number_best:]
    top_k_indices = top_k_indices[np.argsort(values[top_k_indices])[::-1]]

    with open("best_initial_points_for_sup_meuse.pkl", "wb") as f:
        pickle.dump([valid_possible_values[i] for i in top_k_indices], f)
    print(f"Created the file with {number_best} initial points")

    return 0

def best_values_for_inf(x, data, possible_values, number_best):
    values = []
    valid_possible_values = []

    for element in possible_values:
        theta = element[0]
        p = element[1]
        value = single_evaluation(x, theta, data, p)

        # Check that value is neither NaN nor inf (positive or negative)
        if not np.isnan(value) and not np.isinf(value):
            values.append(value)
            valid_possible_values.append(element)

    values = np.array(values)

    top_k_indices = np.argpartition(values, number_best)[:number_best]
    top_k_indices = top_k_indices[np.argsort(values[top_k_indices])]

    with open("best_initial_points_for_inf_meuse.pkl", "wb") as f:
        pickle.dump([valid_possible_values[i] for i in top_k_indices], f)
    print(f"Created the file with {number_best} initial points")

    return 0


def function_for_maximization(theta_p, x , data):
    p = theta_p[-(n + 1):]

    theta = np.reshape(theta_p[:-(n + 1)], shape=(n + 1, length_theta))
    return  - single_evaluation(x, theta, data, p)


def function_for_minimization(theta_p, x , data):
    p = theta_p[-(n + 1):]

    theta = np.reshape(theta_p[:-(n + 1)], shape=(n + 1, length_theta))
    return  single_evaluation(x, theta, data, p)

def constraint_f(theta_p):


    theta = theta_p[:-(n+1)].reshape(n+1 , 3)
    matrix = repartition_function(quantiles, theta)


    return np.dot(matrix, theta_p[-(n + 1):])

constraints = [
    {'type': 'eq', 'fun': lambda theta_p: np.sum(theta_p[-(n + 1):]) - 1},
    {'type': 'ineq', 'fun': lambda theta_p: theta_p[-(n + 1):]},
    {'type': 'eq', 'fun': lambda theta_p: constraint_f(theta_p) - alphas},
    {'type': 'ineq', 'fun' : lambda theta_p: theta_p[:-(n + 1)][1::3] }
]




def optimization(function, best_values, x, data, maximize=True):
    result = []
    best_point = []

    options = {
        'verbose': 0,
        'gtol': 1e-8,
        'xtol': 1e-8,
        'maxiter': 250,
        'initial_tr_radius': 1.0,
    }

    for element in best_values:
        initial_params = np.concatenate([element[0].flatten(), element[1]])
        try :
            if np.any(likelihood(meuse_data, element[0]) != 0):
                optim = minimize(
                    function,
                    x0=initial_params,
                    args=(x, data),
                    constraints=constraints,
                    method='trust-constr',
                    options = options
                )
                value = -optim.fun if maximize else optim.fun
                result.append(value)
                best_point.append(optim.x)
        except:
            pass

    if not result:
        print("No valid result finded.")
        return None, None

    i = np.argmax(result) if maximize else np.argmin(result)

    return result[i], best_point[i]


if __name__ == "__main__":

    number_best = 100


    x_s = np.arange(300., 3400., step=40.)

    with open("possible_values_meuse.pkl", "rb") as f:
        possible = pickle.load(f)


    infimum = []
    argmax = []
    supremum = []
    argmin = []

    i = 0

    for x in x_s:
        i+=1

        if len(supremum) == 0 or supremum[-1] < 1.:
            best_values_for_sup(x, meuse_data, possible, number_best)
            with open("best_initial_points_for_sup_meuse.pkl", "rb") as f:
                bests_max = pickle.load(f)

            result_max, arg_max = optimization(function_for_maximization, bests_max, x, meuse_data, maximize=True)
        supremum.append(float(result_max))
        argmax.append(arg_max)


        best_values_for_inf(x, meuse_data, possible, number_best)
        with open("best_initial_points_for_inf_meuse.pkl", "rb") as f:
            bests_min = pickle.load(f)

        result_min, arg_min = optimization(function_for_minimization, bests_min, x, meuse_data, maximize=False)
        infimum.append(float(result_min))
        argmin.append(arg_min)
        print(f"Progress {i / len(x_s) * 100:.2f}% - Point x: {x}, Sup: {result_max}, Inf: {result_min}")


    print("🏁 Optimization completed.")
    print(supremum)
    print(infimum)

    np.save("Sup_meuse.npy", np.array(supremum))
    np.save("Inf_meuse.npy", np.array(infimum))
    np.save("Argmax_meuse.npy", np.array(argmax))
    np.save("Argmin_meuse.npy", np.array(argmin))
    np.save("x_meuse.npy", np.array(x_s))






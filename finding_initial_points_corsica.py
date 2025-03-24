import numpy as np
from utils import repartition_function, single_evaluation
from scipy.optimize import minimize
from functools import partial
import os
import pickle



corsica_data = np.load("data_meuse_corsica/numpy_corsica.npy")
corsica_data_quantile = np.load("data_meuse_corsica/numpy_corsica_quantile.npy")



length_theta = 3
n = corsica_data_quantile.shape[0]

quantiles = corsica_data_quantile[:,1]


alphas =  corsica_data_quantile[:,0]



if __name__ == "__main__":

    def possible_values(n, length_theta, number_candidates, quantiles, alphas, std = 40.):

        possible_values = []
        matrix = np.zeros((n + 1, n + 1))
        matrix[-1, :] = 1.

        while len(possible_values) < number_candidates:
            theta = np.zeros(((n+1),length_theta))

            # std has to be high > = 40
            mu = np.random.randn(n + 1)*std + np.mean(corsica_data)
            sigma = np.random.uniform(0,1000, size = n+1)
            xi = np.random.uniform(-1000, 1000, size = n+1)

            theta[:,0] = mu
            theta[:,1] = sigma
            theta[:, 2] = xi


            repartition = repartition_function(quantiles, theta)

            matrix[:-1,:] = repartition

            if np.abs(np.linalg.det(matrix)) > 1e-10:


                p = np.linalg.solve(matrix, np.append(alphas, 1))

                if all(p>=0):
                    possible_values.append((theta, p))

                else:
                    pass
            else:
                pass

        if os.path.exists("possible_values.pkl"):
            with open("possible_values.pkl", "rb") as f:
                existing_values = pickle.load(f)

            updated_values = existing_values + possible_values

            with open("possible_values.pkl", "wb") as f:
                pickle.dump(updated_values, f)
        else:
            with open("possible_values.pkl", "wb") as f:
                pickle.dump(possible_values, f)

        return f"Added { number_candidates} new possible candidates"



    def best_values(x, data, possible_values, number_best):
        values = []

        # Evaluate each element in possible_values
        for element in possible_values:
            theta = element[0]
            p = element[1]

            # Assuming single_evaluation returns a numerical value
            values.append(single_evaluation(x, theta, data, p))


        values = np.array(values)


        top_k_indices = np.argpartition(values, -number_best)[-number_best:]


        top_k_indices = top_k_indices[np.argsort(values[top_k_indices])[::-1]]


        with open("best_initial_points.pkl", "wb") as f:
            pickle.dump([possible_values[i] for i in top_k_indices], f)

        return f"Created the file with {number_best} initial points"





    def function_to_optimize(theta_p, x , data):
        p = theta_p[-(n + 1):]
        theta = np.reshape(theta_p[:-(n + 1)], shape=(n + 1, length_theta))
        return single_evaluation(x, theta, data, p)

    def constraint_f(theta_p):


        theta = theta_p[:-(n+1)].reshape(n+1 , 3)
        matrix = repartition_function(quantiles, theta)


        return np.dot(matrix, theta_p[-(n + 1):])  - alphas

    constraints = [
        {'type': 'eq', 'fun': lambda theta_p: np.sum(theta_p[-(n + 1):]) - 1},
        {'type': 'ineq', 'fun': lambda theta_p: theta_p[-(n + 1):]},
        {'type': 'eq', 'fun': lambda theta_p: constraint_f(theta_p)},
        {'type': 'ineq', 'fun' : lambda theta_p: theta_p[:-(n + 1)][1::3] }
    ]

    def optimization(function, best_values, x , data):
        result = []
        for element in best_values :
            initial_params = np.concatenate([element[0].flatten(),element[1]])
            result.append( minimize(
            function,
            x0=initial_params,  # Initial guess
            args=(x, data),  # Additional arguments
            constraints=constraints,
            method='SLSQP'  # You can experiment with other methods if needed
            ).fun)
        print(result)

        return print(max(result))


    with open("best_initial_points.pkl", "rb") as f:
        bests = pickle.load(f)
    optimization(function_to_optimize, bests, 10., corsica_data)



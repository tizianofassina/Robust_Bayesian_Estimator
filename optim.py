import numpy as np
from scipy.optimize import minimize
from scipy.stats import genextreme
from utils import repartition_function, likelihood
import matplotlib.pyplot as plt

true_mu = 0.0
true_sigma = 1.0
true_xi = 0.1
# SciPy's genextreme uses a shape parameter c = -xi.
data = genextreme.rvs(-true_xi, loc=true_mu, scale=true_sigma, size=20)

# --- Set up the quantile constraints ---
# We'll impose constraints at the 25th, 50th, and 75th percentiles.
n = 3                  # number of quantile constraints
m = n + 1              # number of mixture components (each with theta and p)
q_targets = np.array([0.25, 0.5, 0.75])
x_quantiles = np.percentile(data, [25, 50, 75])

# Suppose n is the number of quantile constraints, and we have m = n+1 models.
m = n + 1  

def objective(X, x_val, data):
    # X is the decision variable vector:
    # First 3*m entries are the m theta_j's, then m entries for the p's.
    theta = X[:3 * m].reshape((m, 3))
    p = X[3 * m:]
    
    F_vals = np.array([repartition_function(x_val, theta_j) for theta_j in theta])
    l_vals = np.array([likelihood(data, theta_j) for theta_j in theta])
    num = np.sum(F_vals * l_vals * p)
    den = np.sum(l_vals * p)
    den = np.maximum(den, 1e-10)
    return num / den

def quantile_constraint(X, x_quantiles, q_targets):
    theta = X[: 3 * m].reshape((m, 3))
    p = X[3 * m:]
    cons = []
    for xq, q in zip(x_quantiles, q_targets):
        F_vals = np.array([repartition_function(xq, theta_j) for theta_j in theta])
        cons.append(np.sum(F_vals * p) - q)
    return np.array(cons)

def sum_p_constraint(X):
    p = X[3 * m:]
    return np.sum(p) - 1

# Bounds: set bounds for theta and p
bounds = []
for j in range(m):
    bounds.append((None, None))   # mu_j can be unbounded
    bounds.append((1e-6, None))     # sigma_j > 0
    bounds.append((None, None))     # xi_j can be unbounded (or set bounds if needed)
for j in range(m):
    bounds.append((0, 1))           # p_j in [0,1]

# Collect constraints in the format required by scipy.optimize.minimize
constraints = [
    {'type': 'eq', 'fun': sum_p_constraint},
    {'type': 'eq', 'fun': lambda X: quantile_constraint(X, x_quantiles, q_targets)}
]

# Initial guess: concatenate initial guesses for theta_j's and p's

initial_theta = np.tile(np.array([true_mu, true_sigma, true_xi]), m)
initial_p = np.repeat(1/m, m)
X0 = np.concatenate([initial_theta, initial_p])
# --- Define the x grid over which to optimize ---
x_grid = np.linspace(true_mu - 2*true_sigma, true_mu + 2*true_sigma, 10)

# Arrays to store results for the minimum and maximum objectives.
min_values = []
min_X = []
max_values = []
max_X = []

# We use separate initial guesses (which can be updated as warm starts) for min and max.
X0_min = X0.copy()
X0_max = X0.copy()

# --- Loop over the x grid ---
for x_val in x_grid:
    # Solve the minimization problem (find minimum J(x))
    res_min = minimize(objective, X0_min, args=(x_val, data), method='SLSQP',
                       bounds=bounds, constraints=constraints, options={'ftol': 1e-9, 'disp': False})
    if res_min.success:
        min_values.append(res_min.fun)
        min_X.append(res_min.x)
        X0_min = res_min.x  # warm start for next iteration (min)
    else:
        min_values.append(np.nan)
        min_X.append(None)
        print(f"Minimum optimization failed at x = {x_val}")
    
    # Solve the maximization problem by minimizing the negative objective.
    res_max = minimize(lambda X, x_val, data: -objective(X, x_val, data),
                       X0_max, args=(x_val, data), method='SLSQP',
                       bounds=bounds, constraints=constraints, options={'ftol': 1e-9, 'disp': False})
    if res_max.success:
        max_values.append(-res_max.fun)
        max_X.append(res_max.x)
        X0_max = res_max.x  # warm start for next iteration (max)
    else:
        max_values.append(np.nan)
        max_X.append(None)
        print(f"Maximum optimization failed at x = {x_val}")

# --- Plot the minimum and maximum optimal objective values vs. x ---
plt.figure(figsize=(10, 6))
plt.plot(x_grid, min_values, label='Minimum Objective Value', lw=2)
plt.plot(x_grid, max_values, label='Maximum Objective Value', lw=2, linestyle='--')
plt.xlabel('x')
plt.ylabel('Objective Value')
plt.title('Optimal Objective Values (Min and Max) vs. x')
plt.legend()
plt.grid(True)
plt.show()

# --- Inspect optimal parameters at a selected grid point ---
idx_mid = len(x_grid) // 2
print("\n--- At x = {:.4f} ---".format(x_grid[idx_mid]))
if min_X[idx_mid] is not None:
    theta_opt_min = min_X[idx_mid][:3*m].reshape((m, 3))
    p_opt_min = min_X[idx_mid][3*m:]
    print("Minimum solution:")
    print("Theta values (each row is [mu, sigma, xi]):")
    print(theta_opt_min)
    print("Weights p:")
    print(p_opt_min)
else:
    print("No valid minimum solution at the selected grid point.")

if max_X[idx_mid] is not None:
    theta_opt_max = max_X[idx_mid][:3*m].reshape((m, 3))
    p_opt_max = max_X[idx_mid][3*m:]
    print("\nMaximum solution:")
    print("Theta values (each row is [mu, sigma, xi]):")
    print(theta_opt_max)
    print("Weights p:")
    print(p_opt_max)
else:
    print("No valid maximum solution at the selected grid point.")
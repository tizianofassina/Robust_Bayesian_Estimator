import numpy as np

import matplotlib.pyplot as plt

def make_values_increasing(values):
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]
    return values

def make_values_increasing_reverse(values):
    for i in range(len(values) - 2, -1, -1):
        if values[i] > values[i + 1]:
            values[i] = values[i + 1]
    return values


# --- FOR CORSICA ---

sup_corsica = np.load("Sup_corsica.npy")
inf_corsica = np.load("inf_corsica.npy")

x = np.load("x_corsica.npy")



sup_corsica = make_values_increasing(sup_corsica)
inf_corsica = make_values_increasing_reverse(inf_corsica)

plt.figure(figsize=(10, 6))


plt.plot(x, sup_corsica, label=r'$\sup_{\theta, p} G(x, \theta, p)$', color='blue', linewidth=2)
plt.plot(x, inf_corsica, label=r'$\inf_{\theta, p} G(x, \theta, p)$', color='red', linewidth=2)

plt.title('Plot of Supremum and Infimum of G(x, θ, p) for Corsica Dataset', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('G(x, θ, p)', fontsize=12)
plt.legend()
plt.grid(True)

plt.show()



# --- FOR MEUSE ---


sup_meuse = np.load("Sup_meuse.npy")
inf_meuse = np.load("inf_meuse.npy")

x = np.load("x_meuse.npy")



sup_meuse = make_values_increasing(sup_meuse)
inf_meuse = make_values_increasing_reverse(inf_meuse)


plt.figure(figsize=(10, 6))


plt.plot(x, sup_meuse, label=r'$\sup_{\theta, p} G(x, \theta, p)$', color='blue', linewidth=2)
plt.plot(x, inf_meuse, label=r'$\inf_{\theta, p} G(x, \theta, p)$', color='red', linewidth=2)

plt.title('Plot of Supremum and Infimum of G(x, θ, p) for Meuse Dataset', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('G(x, θ, p)', fontsize=12)
plt.legend()
plt.grid(True)

plt.show()
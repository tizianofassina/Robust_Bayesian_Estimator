import pandas as pd
import numpy as np


with open("data_meuse_corsica/max-meuse.txt") as file:
    data = file
    meuse_data = []

    for element in data:
        elements = element.split()
        if element!="\n":
            meuse_data.append(float(elements[0]))


df = pd.read_csv("data_meuse_corsica/pluviometry-corsica.csv")

with open("data_meuse_corsica/pluviometry-corsica.csv") as file:
    data = file
    corsica_data = []
    header = next(data)

    for element in data:
        elements = element.split(";")
        if element!="\n":
            number = elements[2][:-1]
            number = number.replace(',','.')
            corsica_data.append(float(number))

np.save( "data_meuse_corsica/numpy_meuse.npy", np.array(meuse_data))

np.save( "data_meuse_corsica/numpy_corsica.npy", np.array(corsica_data))
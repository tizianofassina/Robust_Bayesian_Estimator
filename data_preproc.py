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



#The quantiles are saved with the columns : order of the quantile, quantile value, interval
quantile_corsica = np.zeros((3,3))
quantile_meuse = np.zeros((3,3))


quantile_meuse[0,0]  = 0.05
quantile_meuse[0,1] = 1250
quantile_meuse[0,2] = 200
quantile_meuse[1,0] = 0.5
quantile_meuse[1,1] = 2000
quantile_meuse[1,2] = 100
quantile_meuse[2,0] = 0.75
quantile_meuse[2,1] = 2100
quantile_meuse[2,2] = 100


quantile_corsica[0,0]  = 0.25
quantile_corsica[0,1] = 75
quantile_corsica[0,2] = 20
quantile_corsica[1,0] = 0.5
quantile_corsica[1,1] = 100
quantile_corsica[1,2] = 20
quantile_corsica[2,0] = 0.75
quantile_corsica[2,1] = 150
quantile_corsica[2,2] = 20

np.save( "data_meuse_corsica/numpy_corsica_quantile.npy",quantile_corsica)
np.save( "data_meuse_corsica/numpy_meuse_quantile.npy",quantile_meuse)







#leemos los datos de mediciones
#usamos:
#lineal: matplotlib -> hacemos una funcion = libreria (x1, x2)
#cubica: interpolamos x1 y x2 por separaado y los juntamos con un plot

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import CubicSpline, PchipInterpolator, RectBivariateSpline, griddata
# Listas para almacenar los datos
x_ground_truth = []
y_ground_truth = []

try:
    with open('mnyo_ground_truth.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')

        for row in csv_reader:
            valores = row[0].split()
            x_ground_truth.append(float(valores[0]))
            y_ground_truth.append(float(valores[1]))
except FileNotFoundError:
    print("El archivo CSV de ground truth no se encontró.")
except Exception as e:
    print(f"Se produjo un error al leer el archivo de ground truth: {str(e)}")

# Separate lists for x and y in mnyo_mediciones.csv
x_mediciones = []
y_mediciones = []

try:
    with open('mnyo_mediciones.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')

        for row in csv_reader:
            valores = row[0].split()
            x_mediciones.append(float(valores[0]))
            y_mediciones.append(float(valores[1]))
except FileNotFoundError:
    print("El archivo CSV de mediciones no se encontró.")
except Exception as e:
    print(f"Se produjo un error al leer el archivo de mediciones: {str(e)}")

# #hacemos interpolación lineal

#Create the x values for interpolation (e.g., more points for a smoother curve)
x_interp = np.linspace(min(x_mediciones), max(x_mediciones), 10)

# # Perform linear interpolation
y_interp = np.interp(x_interp, x_mediciones, y_mediciones)

# Plot the original data and the interpolated curve
plt.plot(x_mediciones, y_mediciones, 'o', label='Data Points')
plt.plot(x_interp, y_interp, '-', label='Linear Interpolation')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Linear Interpolation of mnyo_mediciones.csv')
plt.legend()
plt.grid(True)
plt.show()
#%%
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Separate lists for x and y in mnyo_mediciones.csv
x_mediciones = []
y_mediciones = []

try:
    with open('mnyo_mediciones.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')

        for row in csv_reader:
            valores = row[0].split()
            x_mediciones.append(float(valores[0]))
            y_mediciones.append(float(valores[1]))
except FileNotFoundError:
    print("El archivo CSV de mediciones no se encontró.")
except Exception as e:
    print(f"Se produjo un error al leer el archivo de mediciones: {str(e)}")

# Create an interpolation function
linear_interpolator = interp1d(x_mediciones, y_mediciones, kind='linear')

# Create the x values for interpolation (e.g., more points for a smoother curve)
x_interp = np.linspace(min(x_mediciones), max(x_mediciones), 100)

# Perform linear interpolation using the interpolation function
y_interp = linear_interpolator(x_interp)

# Plot the original data and the interpolated curve
plt.plot(x_mediciones, y_mediciones, 'o', label='Data Points')
plt.plot(x_interp, y_interp, '-', label='Linear Interpolation')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Linear Interpolation of mnyo_mediciones.csv')
plt.legend()
plt.grid(True)
plt.show()

import csv
from tabulate import tabulate
import math
import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from scipy.interpolate import lagrange, interp1d, CubicSpline, PchipInterpolator

# Listas para almacenar los datos
x_ground_truth = []
y_ground_truth = []

try:
    with open('mnyo_ground_truth.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        
        for row in csv_reader:
            # Dividir la fila en dos valores y convertirlos a números flotantes
            valores = row[0].split()
            x_ground_truth.append(float(valores[0]))
            y_ground_truth.append(float(valores[1]))
except FileNotFoundError:
    print("El archivo CSV no se encontró.")
except Exception as e:
    print(f"Se produjo un error: {str(e)}")

# Listas para almacenar los datos
x1_ti = []
x2_ti = []

try:
    with open('mnyo_mediciones.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        
        for row in csv_reader:
            # Dividir la fila en dos valores y convertirlos a números flotantes
            valores = row[0].split()
            x1_ti.append(float(valores[0]))
            x2_ti.append(float(valores[1]))
except FileNotFoundError:
    print("El archivo CSV no se encontró.")
except Exception as e:
    print(f"Se produjo un error: {str(e)}")

#necesito la variable y el f(variable)
# lo primero que le paso a lagrange es el tiemp(0;range(long lista x))
#lo segundo que le paso son las mediciones de x
#hago lo mismo para y
#ahi tengo una x(t) y una y(t)
#ahi ploteo la curva paramétrica (chat gpt)

# Define an array of time values
time_values = np.linspace(0, len(x1_ti) - 1, 10) # Adjust the number of points as needed

#primero hago lagrange con el tiempo y las mediciones de x
x_t = lagrange(time_values, x1_ti)
y_t = lagrange(time_values, x2_ti)

# Calculate x and y values using Lagrange polynomials
x_values = [x_t(t) for t in time_values]
y_values = [y_t(t)for t in time_values]

# Plot the parametric curve
plt.figure(figsize=(10, 6))
plt.plot(x1_ti, x2_ti, 'o', label='Data Points', color='purple')
plt.plot(x_values, y_values, label='Interpolated Function', color='violet')
plt.plot(x_ground_truth, y_ground_truth, label='Ground Truth Function', linestyle='--', color = "pink")
plt.xlabel('X(t)')
plt.ylabel('Y(t)')
plt.title('Comparison of Interpolated Linear and Ground Truth Functions')
plt.legend()
plt.grid(True)
plt.show()

# Create a cubic spline interpolation function for x1_ti and x2_ti
cs_x1 = CubicSpline(time_values, x1_ti)
cs_x2 = CubicSpline(time_values, x2_ti)

# Generate x values for interpolation
time_interp = np.linspace(0, len(x1_ti) - 1, 1000)  # Adjust the number of points as needed

# Interpolate y values using the cubic spline functions
x1_interp = cs_x1(time_interp)
x2_interp = cs_x2(time_interp)

# Plot the original data and the cubic spline interpolation
plt.figure(figsize=(10, 6))
plt.scatter(x1_ti, x2_ti, label='Data Points', color='purple')
plt.plot(x_ground_truth, y_ground_truth, label='Ground Truth Function', linestyle='--', color='green')
plt.plot(x1_interp, x2_interp, label='Cubic Spline Interpolation', color='blue')
plt.xlabel('X1(t)')
plt.ylabel('X2(t)')
plt.title('Comparison of Interpolated Spline and Ground Truth Functions')
plt.legend()
plt.grid(True)
plt.show()

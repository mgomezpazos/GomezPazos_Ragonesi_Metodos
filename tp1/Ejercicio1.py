# -----------------------------------EJERCICIO 1: MANEJO DE ARCHIVOS-----------------------------------------------------------------------------------------------------------------

import csv
from tabulate import tabulate
import math
import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from scipy.interpolate import CubicSpline, PchipInterpolator, RectBivariateSpline, griddata
from numpy.polynomial.chebyshev import chebpts2
from mpl_toolkits.mplot3d import Axes3D

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

# Ahora deberías tener los datos en x_ground_truth y y_ground_truth

# Combinar las listas en una lista de tuplas
#datos = list(zip(x_ground_truth, y_ground_truth))

# Imprimir la tabla
# tabla = tabulate(datos, headers=["Columna 1", "Columna 2"], tablefmt="pretty")
# print(tabla)

# Listas para almacenar los datos
x1_ti = []
x2_ti = []

try:
    with open('mnyo_mediciones.csv', 'r') as csvfile:
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

# Ahora deberías tener los datos en x_ground_truth y y_ground_truth

# Combinar las listas en una lista de tuplas
#datos = list(zip(x_ground_truth, y_ground_truth))

# Imprimir la tabla
# tabla = tabulate(datos, headers=["X1(ti)", "X2(ti)"], tablefmt="pretty")
# print(tabla)


#función
# x1 = 10
# (0.35*x1) + x2 == 3.6

def function(x1_ti, x2_ti): 
    return (x1_ti, x2_ti)

#hacemos interpolación con los 10 puntos de mediciones provistos
#pregunta: en el caso de la funcion real hay que pasarle el ground truth directamente?
# X1, X2 = np.meshgrid(x1_ti, x2_ti)
# Z = function(X1, X2) #valores reales

# #puntos para interpolar
# X1_interp, X2_interp = np.meshgrid(x1_ti, x2_ti)
# points = np.column_stack((X1_interp.ravel(), X2_interp.ravel()))

# #interpolacíon cúbica
# Z_interp_equi = griddata((X1, X2), Z.ravel(), points, method='cubic')
# # Z_interp_equi = griddata((X1.ravel(), X2.ravel()), Z.ravel(), points, method='cubic')
# Z_interp_equi = Z_interp_equi.reshape(X1_interp.shape)

# #graficos
# fig, axes = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': '3d'})

# surface1 = axes[0].plot_surface(X1, X2, Z, cmap='viridis')
# axes[0].set_title('Función Original')

# #barra de variación
# cbar1 = plt.colorbar(surface1, ax=axes[0], shrink=0.5, aspect=10)
# cbar1.set_label('Variación de Z')

# surface2 = axes[1].plot_surface(X1_interp, X2_interp, Z_interp_equi, cmap='viridis')
# axes[1].set_title('Interpolación con puntos equiespaciados')

# #barra de variación
# cbar2 = plt.colorbar(surface2, ax=axes[1], shrink=0.5, aspect=10)
# cbar2.set_label('Variación de Z')

# plt.show()

# # Calculate relative error
# relative_error = np.abs(Z_interp_equi - function(X1_interp, X2_interp)) / np.abs(function(X1_interp, X2_interp))

# # Plot relative error
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# surface = ax.plot_surface(X1_interp, X2_interp, relative_error, cmap='viridis')
# plt.title('Relative Error of Cubic Interpolation')
# cbar = plt.colorbar(surface, shrink=0.5, aspect=10)
# cbar.set_label('Relative Error')
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# plt.show()

#%% Lo que nos dijo el chat
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Listas para almacenar los datos del archivo 'mnyo_ground_truth.csv'
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
    print("El archivo 'mnyo_ground_truth.csv' no se encontró.")
except Exception as e:
    print(f"Se produjo un error al cargar 'mnyo_ground_truth.csv': {str(e)}")

# Listas para almacenar los datos del archivo 'mnyo_mediciones.csv'
x1_mediciones = []
x2_mediciones = []

try:
    with open('mnyo_mediciones.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        
        for row in csv_reader:
            # Dividir la fila en dos valores y convertirlos a números flotantes
            valores = row[0].split()
            x1_mediciones.append(float(valores[0]))
            x2_mediciones.append(float(valores[1]))
except FileNotFoundError:
    print("El archivo 'mnyo_mediciones.csv' no se encontró.")
except Exception as e:
    print(f"Se produjo un error al cargar 'mnyo_mediciones.csv': {str(e)}")


# Eliminar valores duplicados y ordenar los arreglos
x_ground_truth = np.unique(x_ground_truth)
y_ground_truth = np.array([y_ground_truth[i] for i in np.argsort(x_ground_truth)])
x1_mediciones = np.unique(x1_mediciones)
x2_mediciones = np.array([x2_mediciones[i] for i in np.argsort(x1_mediciones)])

# Crear objetos CubicSpline para interpolar x_ground_truth y y_ground_truth
cs_x1_mediciones = CubicSpline(x1_mediciones, x2_mediciones)

# Definir nuevos puntos de tiempo para la interpolación
nuevos_x1_mediciones = np.linspace(min(x1_mediciones), max(x1_mediciones), 100)

# Evaluar las interpolaciones en los nuevos puntos de tiempo
x2_mediciones_interpolado = cs_x1_mediciones(nuevos_x1_mediciones)

# Graficar los resultados
plt.figure(figsize=(8, 6))
plt.plot(x_ground_truth, y_ground_truth, 'o', label='Ground Truth')
plt.xlabel('X Ground Truth')
plt.ylabel('Y Ground Truth')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x1_mediciones, x2_mediciones, 'o', label='Mediciones')
plt.plot(nuevos_x1_mediciones, x2_mediciones_interpolado, '-', label='Interpolación (Mediciones)')
plt.xlabel('X1 Mediciones')
plt.ylabel('X2 Mediciones')
plt.legend()
plt.grid(True)
plt.show()



# %%
# LO QUE NOS PASO LA SEÑORA 2.0
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

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

# Sort x_mediciones and y_mediciones
x_mediciones, y_mediciones = zip(*sorted(zip(x_mediciones, y_mediciones)))

# Perform spline interpolation
spline_interpolation = CubicSpline(x_mediciones, y_mediciones)

# Evaluate the interpolated function at the same points as ground truth data
x_eval = np.linspace(min(x_ground_truth), max(x_ground_truth), 100)
y_interpolated = spline_interpolation(x_eval)

# Create a plot for comparison
plt.figure(figsize=(10, 6))
plt.plot(x_eval, y_interpolated, label='Interpolated Function')
plt.plot(x_ground_truth, y_ground_truth, label='Ground Truth Function', linestyle='--')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Comparison of Interpolated and Ground Truth Functions')
plt.grid(True)
plt.show()


# %%

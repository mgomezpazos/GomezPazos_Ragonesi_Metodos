#----------------------------------------------------------------------------EJERCICIO 2A-------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from scipy.interpolate import lagrange, interp1d, CubicSpline, PchipInterpolator
#-------------------------------------------------------------------FUNCIÓN A Y GENERADOR DE NODOS CHEBYSHEV----------------------------------------------------------------------------
def functionA(value):
    return 0.05 ** (abs(value)) * np.sin(5 * value) + np.tanh(2 * value) + 2

def generate_chebyshev_nodes(n, a, b):
    k = np.arange(1, n + 1)
    chebyshev_nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return chebyshev_nodes
#-------------------------------------------------------------------INTERPOLACIÓN CON PUNTOS EQUIESPACIADOS---------------------------------------------------------------------------
#puntos de interpolación de functionA en el intervalo [-3, 3]
interpolation_pointsA = np.linspace(-3, 3, 10)
# Calcular los valores reales de la funciónA en los puntos de interpolación
real_valuesA = functionA(interpolation_pointsA)
# Realizar interpolación con polinomio de Lagrange
lagrange_polyA = lagrange(interpolation_pointsA, real_valuesA)
# Realizar interpolación con spline cúbico
spline_cubicA = CubicSpline(interpolation_pointsA, real_valuesA)
# Realizar interpolación con spline quíntico (spline cúbico natural)
spline_quinticA = PchipInterpolator(interpolation_pointsA, real_valuesA)
# Puntos donde se evaluarán los resultados
evaluation_pointsA = np.linspace(-3, 3, 300)  # Más puntos para una representación suave
# Evaluar los métodos de interpolación en los puntos de evaluación
lagrange_interpolatedA = lagrange_polyA(evaluation_pointsA)
spline_cubic_interpolatedA = spline_cubicA(evaluation_pointsA)
spline_quintic_interpolatedA = spline_quinticA(evaluation_pointsA)
#----------------------------------------------------------------------ERROR CON PUNTOS EQUIESPACIADOS------------------------------------------------------------------------------------
# Calculate relative errors
relative_error_lagrange = np.abs(lagrange_interpolatedA - functionA(evaluation_pointsA))
relative_error_spline_cubic = np.abs(spline_cubic_interpolatedA - functionA(evaluation_pointsA))
relative_error_spline_quintic = np.abs(spline_quintic_interpolatedA - functionA(evaluation_pointsA))
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Hacer un for para n cantidad de puntos y agarramos el error máximo absoluto (de cada método de interpolación), 
# # armamos un array con  eso y graficamos con los 3 juntos. 

# # Define una lista de valores de n (cantidad de puntos de interpolación) que deseas probar
# n_values = [10, 20, 30, 40, 50]  # Puedes modificar esta lista según tus necesidades

# # Métodos de interpolación que deseas graficar
# interpolation_methods = ["Lagrange", "Spline Cúbico", "Spline Quíntico"]

# # Arreglo para almacenar los errores máximos relativos para cada valor de n y método de interpolación
# max_relative_errors = {method: [] for method in interpolation_methods}

# # Función que calcula el error máximo relativo
# def calculate_max_relative_error(interpolated_values, true_values):
#     return np.max(np.abs(interpolated_values - true_values) / np.abs(true_values))

# # Puntos de evaluación comunes
# evaluation_pointsA = np.linspace(-3, 3, 300)

# for n in n_values:
#     # Puntos de interpolación
#     interpolation_pointsA = np.linspace(-3, 3, n)
    
#     # Calcular los valores reales de la función en los puntos de interpolación
#     real_valuesA = functionA(interpolation_pointsA)
    
#     for method in interpolation_methods:
#         # Realizar interpolación
#         if method == "Lagrange":
#             interpolation_function = lagrange(interpolation_pointsA, real_valuesA)
#         elif method == "Spline Cúbico":
#             interpolation_function = CubicSpline(interpolation_pointsA, real_valuesA)
#         elif method == "Spline Quíntico":
#             interpolation_function = PchipInterpolator(interpolation_pointsA, real_valuesA)
        
#         # Evaluar el método de interpolación en los puntos de evaluación
#         interpolated_values = interpolation_function(evaluation_pointsA)
        
#         # Calcular el error máximo relativo y almacenarlo
#         max_error_relative = calculate_max_relative_error(interpolated_values, functionA(evaluation_pointsA))
#         max_relative_errors[method].append(max_error_relative)

# # Graficar los resultados de los errores máximos relativos para los métodos de interpolación seleccionados
# plt.figure(figsize=(10, 6))
# for method in interpolation_methods:
#     plt.plot(n_values, max_relative_errors[method], label=method)
# plt.xlabel('Cantidad de Puntos de Interpolación (n)')
# plt.ylabel('Error Máximo Relativo')
# plt.title('Comparación de Errores Máximos Relativos para Diferentes Cantidades de Puntos de Interpolación')
# plt.legend()
# plt.grid(True)
# plt.show()
# Crear una lista de números de puntos de interpolación que deseas probar
n_values = [10, 20, 30, 40, 50]

# Crear diccionarios para almacenar los errores relativos y las sumatorias de errores relativos
relative_errors = {}
sum_of_relative_errors = {}

# Puntos de evaluación comunes
evaluation_points = np.linspace(-3, 3, 300)

for n in n_values:
    # Generar puntos de interpolación (pueden ser equiespaciados o de Chebyshev)
    interpolation_points = np.linspace(-3, 3, n)
    
    # Calcular los valores reales de la función en los puntos de interpolación
    real_values = functionA(interpolation_points)
    
    # Realizar interpolación cúbica
    cs = CubicSpline(interpolation_points, real_values)
    
    # Evaluar el método de interpolación en los puntos de evaluación
    interpolated_values = cs(evaluation_points)
    
    # Calcular el error relativo
    relative_error = np.abs((interpolated_values - functionA(evaluation_points)) / functionA(evaluation_points))
    
    # Almacenar los errores relativos en el diccionario
    relative_errors[n] = relative_error
    
    # Calcular la sumatoria de errores relativos
    sum_relative_error = np.sum(relative_error)
    
    # Almacenar la sumatoria en el diccionario
    sum_of_relative_errors[n] = sum_relative_error

# Calcular la diferencia entre sumatorias de errores relativos
differences = []
previous_sum = None
for n in n_values:
    if previous_sum is not None:
        difference = sum_of_relative_errors[n] - previous_sum
        differences.append(difference)
    previous_sum = sum_of_relative_errors[n]

# Graficar la diferencia entre las sumatorias de errores relativos
plt.figure(figsize=(10, 6))
plt.plot(n_values[1:], differences, marker='o')
plt.xlabel('Cantidad de Puntos de Interpolación (n)')
plt.ylabel('Diferencia en Sumatoria de Errores Relativos')
plt.title('Diferencia en Sumatoria de Errores Relativos vs. Cantidad de Puntos de Interpolación')
plt.grid(True)
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(evaluation_pointsA, functionA(evaluation_pointsA), label='Función Original')
plt.plot(evaluation_pointsA, lagrange_interpolatedA, label='Interpolación Lagrange')
plt.plot(evaluation_pointsA, spline_cubic_interpolatedA, label='Spline Cúbico')
plt.plot(evaluation_pointsA, spline_quintic_interpolatedA, label='Spline Quíntico')
plt.scatter(interpolation_pointsA, real_valuesA, color='red', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('functionA(x)')
plt.title('Comparación de Métodos de Interpolación con Puntos Equiespaciados')
plt.legend()
plt.grid(True)
plt.show()
# Grafico el Error Absoluto
plt.figure(figsize=(10, 6))
plt.plot(evaluation_pointsA, relative_error_lagrange, label='Lagrange')
plt.plot(evaluation_pointsA, relative_error_spline_cubic, label='Spline Cúbico')
plt.plot(evaluation_pointsA, relative_error_spline_quintic, label='Spline Quíntico')
plt.xlabel('x')
plt.ylabel('Error Relativo')
plt.title('Comparación de Error Relativo con Puntos Equiespaciados')
plt.legend()
plt.grid(True)
plt.show()
#-------------------------------------------------------------------INTERPOLACIÓN CON PUNTOS NO EQUIESPACIADOS------------------------------------------------------------------------
# Nodos Chebyshev 
interpolation_pointsA = generate_chebyshev_nodes(10, -3, 3)
interpolation_pointsA = np.sort(interpolation_pointsA) 
# Calcular los valores reales de la funciónA en los puntos de interpolación
real_valuesA = functionA(interpolation_pointsA)
# Realizar interpolación con polinomio de Lagrange
lagrange_polyA = lagrange(interpolation_pointsA, real_valuesA)
# Realizar interpolación con spline cúbico
spline_cubicA = CubicSpline(interpolation_pointsA, real_valuesA)
# Realizar interpolación con spline quíntico (spline cúbico natural)
spline_quinticA = PchipInterpolator(interpolation_pointsA, real_valuesA)
# Puntos donde se evaluarán los resultados
evaluation_pointsA = np.linspace(-3, 3, 300)  # Más puntos para una representación suave
# Evaluar los métodos de interpolación en los puntos de evaluación
lagrange_interpolatedA = lagrange_polyA(evaluation_pointsA)
spline_cubic_interpolatedA = spline_cubicA(evaluation_pointsA)
spline_quintic_interpolatedA = spline_quinticA(evaluation_pointsA)
#-------------------------------------------------------------------------------ERROR RELATIVO---------------------------------------------------------------------------------------------------
# Calculo del error relativo
relative_error_lagrange = np.abs((lagrange_interpolatedA - functionA(evaluation_pointsA)) / functionA(evaluation_pointsA)) 
relative_error_spline_cubic = np.abs((spline_cubic_interpolatedA - functionA(evaluation_pointsA)) / functionA(evaluation_pointsA))
relative_error_spline_quintic = np.abs((spline_quintic_interpolatedA - functionA(evaluation_pointsA)) / functionA(evaluation_pointsA))
# Plotep
plt.figure(figsize=(10, 6))
plt.plot(evaluation_pointsA, functionA(evaluation_pointsA), label='Función Original')
plt.plot(evaluation_pointsA, lagrange_interpolatedA, label='Interpolación Lagrange')
plt.plot(evaluation_pointsA, spline_cubic_interpolatedA, label='Spline Cúbico')
plt.plot(evaluation_pointsA, spline_quintic_interpolatedA, label='Spline Quíntico')
plt.scatter(interpolation_pointsA, real_valuesA, color='red', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('functionA(x)')
plt.title('Comparación de Métodos de Interpolación con Puntos No Equiespaciados')
plt.legend()
plt.grid(True)
plt.show()
# Ploteo del error
plt.figure(figsize=(10, 6))
plt.plot(evaluation_pointsA, relative_error_lagrange, label='Lagrange')
plt.plot(evaluation_pointsA, relative_error_spline_cubic, label='Spline Cúbico')
plt.plot(evaluation_pointsA, relative_error_spline_quintic, label='Spline Quíntico')
plt.xlabel('x')
plt.ylabel('Error Absoluto')
plt.title('Comparación de Error Relativo con Puntos No Equiespaciados')
plt.legend()
plt.grid(True)
plt.show()

#----------------------------------------------------------------------------EJERCICIO 2A-------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from scipy.interpolate import lagrange, interp1d, CubicSpline, PchipInterpolator
#-------------------------------------------------------------------FUNCIÓN A Y GENERADOR DE NODOS CHEBYSHEV----------------------------------------------------------------------------
def functionA(value):
    return 0.05 ** (abs(value)) * np.sin(5 * value) + np.tanh(2 * value) + 2
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
# Define a range of numbers of interpolation points up to 100
num_points_range = np.arange(2, 15) 

# Lists to store cumulative errors for each interpolation method
lagrange_cumulative_error = []
cubic_spline_cumulative_error = []
quintic_spline_cumulative_error = []

for num_points in num_points_range:
    # Generate equidistant interpolation points
    interpolation_points = np.linspace(-3, 3, num_points)
    real_values = functionA(interpolation_points)

    # Interpolate using Lagrange, cubic splines, and quintic splines
    lagrange_poly = lagrange(interpolation_points, real_values)
    cubic_spline = CubicSpline(interpolation_points, real_values)
    quintic_spline = PchipInterpolator(interpolation_points, real_values)

    # Evaluate the interpolations at evaluation points
    evaluation_points = np.linspace(-3, 3, 100)
    lagrange_interpolated = lagrange_poly(evaluation_points)
    cubic_spline_interpolated = cubic_spline(evaluation_points)
    quintic_spline_interpolated = quintic_spline(evaluation_points)

    # Calculate the relative errors
    lagrange_relative_error = (np.abs(functionA(evaluation_points) - lagrange_interpolated) / functionA(evaluation_points)) * 100
    cubic_spline_relative_error = (np.abs(functionA(evaluation_points) - cubic_spline_interpolated) / functionA(evaluation_points)) * 100
    quintic_spline_relative_error = (np.abs(functionA(evaluation_points) - quintic_spline_interpolated) / functionA(evaluation_points)) * 100

    # Calculate cumulative errors
    lagrange_cumulative_error.append(np.sum(lagrange_relative_error))
    cubic_spline_cumulative_error.append(np.sum(cubic_spline_relative_error))
    quintic_spline_cumulative_error.append(np.sum(quintic_spline_relative_error))

# Plot the cumulative errors vs. the number of interpolation points
plt.figure(figsize=(10, 6))
plt.plot(num_points_range, lagrange_cumulative_error, label='Lagrange')
plt.plot(num_points_range, cubic_spline_cumulative_error, label='Cubic Spline')
plt.plot(num_points_range, quintic_spline_cumulative_error, label='Quintic Spline')
plt.xlabel('Number of Interpolation Points')
plt.ylabel('Cumulative Relative Error (%)')
plt.title('Cumulative Relative Error vs. Number of Interpolation Points')
plt.legend()
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
def generate_chebyshev_nodes(n, a, b):
    k = np.arange(1, n + 1)
    chebyshev_nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return chebyshev_nodes
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

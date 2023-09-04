import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from scipy.interpolate import lagrange, interp1d, CubicSpline, PchipInterpolator

def functionA(value):
    return 0.05 ** (abs(value)) * np.sin(5 * value) + np.tanh(2 * value) + 2

def generate_chebyshev_nodes(n, a, b):
    k = np.arange(1, n + 1)
    chebyshev_nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return chebyshev_nodes

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
# Calculate relative errors
absolute_error_lagrange = np.abs(lagrange_interpolatedA - functionA(evaluation_pointsA))
absolute_error_spline_cubic = np.abs(spline_cubic_interpolatedA - functionA(evaluation_pointsA))
absolute_error_spline_quintic = np.abs(spline_quintic_interpolatedA - functionA(evaluation_pointsA))

# Hacer un for para n cantidad de puntos y agarramos el error máximo absoluto (de cada método de interpolación), 
# armamos un array con  eso y graficamos con los 3 juntos. 

# Lista para almacenar los máximos errores absolutos para cada método
max_errors_lagrange = []
max_errors_spline_cubic = []
max_errors_spline_quintic = []

# # Valores de n para diferentes cantidades de puntos de interpolación
n_values = [5, 10, 15, 20, 25]

# # Iterar a través de diferentes valores de n
for n in n_values:
    # Generar puntos de interpolación equiespaciados
    interpolation_pointsA = np.linspace(-3, 3, n)
    # Calcular los valores reales de la funciónA en los puntos de interpolación
    real_valuesA = functionA(interpolation_pointsA)

    # Realizar interpolación con polinomio de Lagrange
    lagrange_polyA = lagrange(interpolation_pointsA, real_valuesA)
    # Realizar interpolación con spline cúbico
    spline_cubicA = CubicSpline(interpolation_pointsA, real_valuesA)
    # Realizar interpolación con spline quíntico (spline cúbico natural)
    spline_quinticA = PchipInterpolator(interpolation_pointsA, real_valuesA)

    # Evaluar los métodos de interpolación en los puntos de evaluación
    lagrange_interpolatedA = lagrange_polyA(evaluation_pointsA)
    spline_cubic_interpolatedA = spline_cubicA(evaluation_pointsA)
    spline_quintic_interpolatedA = spline_quinticA(evaluation_pointsA)

    # Calcular errores absolutos en cada punto de evaluación
    absolute_error_lagrange = np.abs(lagrange_interpolatedA - functionA(evaluation_pointsA))
    absolute_error_spline_cubic = np.abs(spline_cubic_interpolatedA - functionA(evaluation_pointsA))
    absolute_error_spline_quintic = np.abs(spline_quintic_interpolatedA - functionA(evaluation_pointsA))

    # Calcular y almacenar los máximos errores absolutos para cada método
    max_error_lagrange = np.max(absolute_error_lagrange)
    max_error_spline_cubic = np.max(absolute_error_spline_cubic)
    max_error_spline_quintic = np.max(absolute_error_spline_quintic)

    max_errors_lagrange.append(max_error_lagrange)
    max_errors_spline_cubic.append(max_error_spline_cubic)
    max_errors_spline_quintic.append(max_error_spline_quintic)

# Graficar los máximos errores absolutos para cada método en función de n
plt.figure(figsize=(10, 6))
plt.plot(n_values, max_errors_lagrange, label='Máximo Error Lagrange', marker='o')
plt.plot(n_values, max_errors_spline_cubic, label='Máximo Error Spline Cúbico', marker='o')
plt.plot(n_values, max_errors_spline_quintic, label='Máximo Error Spline Quíntico', marker='o')
plt.xlabel('Cantidad de Puntos de Interpolación (n)')
plt.ylabel('Máximo Error Absoluto')
plt.title('Máximo Error Absoluto vs. Cantidad de Puntos de Interpolación')
plt.legend()
plt.grid(True)
plt.show()


# Graficar los resultados
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
# Grafico el Error Absouto
plt.figure(figsize=(10, 6))
plt.plot(evaluation_pointsA, absolute_error_lagrange, label='Lagrange')
plt.plot(evaluation_pointsA, absolute_error_spline_cubic, label='Spline Cúbico')
plt.plot(evaluation_pointsA, absolute_error_spline_quintic, label='Spline Quíntico')
plt.xlabel('x')
plt.ylabel('Error Absoluto')
plt.title('Comparación de Error Absoluto con Puntos Equiespaciados')
plt.legend()
plt.grid(True)
plt.show()

# Generate Chebyshev nodes
interpolation_pointsA = generate_chebyshev_nodes(10, -3, 3)
interpolation_pointsA = np.sort(interpolation_pointsA)  # Sort the points to ensure they are strictly increasing
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
# Calculate relative errors
absolute_error_lagrange = np.abs(lagrange_interpolatedA - functionA(evaluation_pointsA)) 
absolute_error_spline_cubic = np.abs(spline_cubic_interpolatedA - functionA(evaluation_pointsA))
absolutee_error_spline_quintic = np.abs(spline_quintic_interpolatedA - functionA(evaluation_pointsA))
# Plotting the comparison of different interpolation methods
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
# Grafico el Error Relativo
plt.figure(figsize=(10, 6))
plt.plot(evaluation_pointsA, absolute_error_lagrange, label='Lagrange')
plt.plot(evaluation_pointsA, absolute_error_spline_cubic, label='Spline Cúbico')
plt.plot(evaluation_pointsA, absolute_error_spline_quintic, label='Spline Quíntico')
plt.xlabel('x')
plt.ylabel('Error Absoluto')
plt.title('Comparación de Error Absoluto con Puntos No Equiespaciados')
plt.legend()
plt.grid(True)
plt.show()

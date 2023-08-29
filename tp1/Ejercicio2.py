import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from scipy.interpolate import lagrange, interp1d, interp2d, CubicSpline, PchipInterpolator

def functionA(value):
    return 0.05 * abs(value) * np.sin(5 * value) + np.tanh(2 * value) + 2

def functionB(x:tuple):
    x1, x2 = x
    return 0.7 * exp((-(((9*x1)-2)**2)/4) - (((9*x2) - 2)**2)/4) + 0.45*exp((-(((9*x1)+1)**2)/9) - (((9*x2)+1)**2)/5) + 0.55*exp((-(((9*x1)-6)**2)/4)-(((9*x2)-3)**2)/4) - 0.01 * exp((-(((9*x1)-7)**2)/4)-(((9*x2)-3)**2)/4)
"""
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
# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(evaluation_pointsA, functionA(evaluation_pointsA), label='Función Original')
plt.plot(evaluation_pointsA, lagrange_interpolatedA, label='Interpolación Lagrange')
plt.plot(evaluation_pointsA, spline_cubic_interpolatedA, label='Spline Cúbico')
plt.plot(evaluation_pointsA, spline_quintic_interpolatedA, label='Spline Quíntico')
plt.scatter(interpolation_pointsA, real_valuesA, color='red', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('functionA(x)')
plt.title('Comparación de Métodos de Interpolación')
plt.legend()
plt.grid(True)
plt.show()"""

#puntos de interpolación de functionB en el intervalo [-1, 1]
#interpolation_pointsBX1 = np.array([-0.5, -0.25, 0.1, 0.4, 0.5, 0.7, 0.9])
x1= np.linspace(-1, 1, 10)
x2= np.linspace(-1, 1, 10)
interpolation_pointsB = [(x1, x2) for x in x1 for y in x2]
real_valuesB = [(p, functionB(interpolation_pointsB)) for p in interpolation_pointsB]
#interpolation_pointsBx2= np.array([-0.9,-0.7,-0.45,-0.10,0.25,0.30,0.75])

"""
x = linblablabla
y = linblablabla
points = [(x, y) for x in x for y in y]
eval = [(point, f(points)) for point in points] """


# Calcular los valores reales de la funciónB en los puntos de interpolación
#real_valuesB = [functionB(x1, x2) for x1, x2 in interpolation_pointsB]

# Realizar interpolación con polinomio de Lagrange
lagrange_polyB = interp2d(interpolation_pointsB, real_valuesB, kind ='cubic')

# Realizar interpolación con spline cúbico
spline_cubicB = CubicSpline(interpolation_pointsB, real_valuesB)


# Realizar interpolación con spline quíntico (spline cúbico natural)
spline_quinticB = PchipInterpolator(interpolation_pointsB, real_valuesB)

# Puntos donde se evaluarán los resultados
evaluation_pointsB = np.linspace(-1, 1, 300)  # Más puntos para una representación suave

# Evaluar los métodos de interpolación en los puntos de evaluación
#lagrange_interpolatedB = lagrange_polyB(evaluation_pointsB)
#spline_cubic_interpolatedB = spline_cubicB(evaluation_pointsB)
#spline_quintic_interpolatedB = spline_quinticB(evaluation_pointsB)

plt.figure(figsize=(10, 6))
plt.plot(evaluation_pointsB, functionB(evaluation_pointsB), label='Función Original')
#plt.plot(evaluation_pointsB, lagrange_interpolatedB, label='Interpolación Lagrange')
#plt.plot(evaluation_pointsB, spline_cubic_interpolatedB, label='Spline Cúbico')
#plt.plot(evaluation_pointsB, spline_quintic_interpolatedB, label='Spline Quíntico')
plt.scatter(interpolation_pointsB, real_valuesB, color='red', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('functionB(x)')
plt.title('Comparación de Métodos de Interpolación')
plt.legend()
plt.grid(True)
plt.show()

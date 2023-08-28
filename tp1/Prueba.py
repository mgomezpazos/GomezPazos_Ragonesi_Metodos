import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#%%
# lo que le dio a zoe el chat
# Función fa(x)
def fa(x):
    return 0.05 * np.abs(x) * np.sin(5 * x) + np.tanh(2 * x) + 2

# Datos iniciales
x_original = np.linspace(-3, 3, num=100)
y_original = fa(x_original)

# Puntos para la interpolación
x_interp = np.linspace(-3, 3, num=300)

# Diferentes tipos de interpolación
interpolation_methods = ['linear', 'quadratic', 'cubic']
plt.figure(figsize=(10, 8))

for method in interpolation_methods:
    interp_function = interp1d(x_original, y_original, kind=method)
    y_interp = interp_function(x_interp)
    plt.plot(x_interp, y_interp, label=method)

plt.plot(x_original, y_original, 'o', label='Datos originales')
plt.xlabel('x')
plt.ylabel('fa(x)')
plt.title('Comparación de Esquemas de Interpolación')
plt.legend()
plt.grid(True)
plt.show()
# %%
# LO QUE ME DIO A MI EL CHAT
import numpy as np
from scipy.interpolate import lagrange, CubicSpline
import matplotlib.pyplot as plt

# Definir la función original fa(x)
def fa(x):
    return 0.05 * abs(x) * np.sin(5 * x) + np.tanh(2 * x) + 2

# Definir los puntos de interpolación
interpolation_points = np.linspace(-3, 3, 10)  # Por ejemplo, 10 puntos equidistantes

# Calcular los valores reales de la función en los puntos de interpolación
real_values = fa(interpolation_points)

# Realizar interpolación con polinomio de Lagrange
lagrange_poly = lagrange(interpolation_points, real_values)

# Realizar interpolación con spline cúbico
spline_cubic = CubicSpline(interpolation_points, real_values)

# Puntos donde se evaluarán los resultados
evaluation_points = np.linspace(-3, 3, 100)  # Más puntos para una representación suave

# Evaluar los métodos de interpolación en los puntos de evaluación
lagrange_interpolated = lagrange_poly(evaluation_points)
spline_interpolated = spline_cubic(evaluation_points)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(evaluation_points, fa(evaluation_points), label='Función Original')
plt.plot(evaluation_points, lagrange_interpolated, label='Interpolación Lagrange')
plt.plot(evaluation_points, spline_interpolated, label='Spline Cúbico')
plt.scatter(interpolation_points, real_values, color='red', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('fa(x)')
plt.title('Comparación de Métodos de Interpolación')
plt.legend()
plt.grid(True)
plt.show()

# %%

# CON SPLINE QUINTICO
import numpy as np
from scipy.interpolate import lagrange, CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt

# Definir la función original fa(x)
def fa(x):
    return 0.05 * abs(x) * np.sin(5 * x) + np.tanh(2 * x) + 2

# Definir los puntos de interpolación
interpolation_points = np.linspace(-3, 3, 10)  # Por ejemplo, 10 puntos equidistantes

# Calcular los valores reales de la función en los puntos de interpolación
real_values = fa(interpolation_points)

# Realizar interpolación con polinomio de Lagrange
lagrange_poly = lagrange(interpolation_points, real_values)

# Realizar interpolación con spline cúbico
spline_cubic = CubicSpline(interpolation_points, real_values)

# Realizar interpolación con spline quíntico (spline cúbico natural)
spline_quintic = PchipInterpolator(interpolation_points, real_values)

# Puntos donde se evaluarán los resultados
evaluation_points = np.linspace(-3, 3, 100)  # Más puntos para una representación suave

# Evaluar los métodos de interpolación en los puntos de evaluación
lagrange_interpolated = lagrange_poly(evaluation_points)
spline_cubic_interpolated = spline_cubic(evaluation_points)
spline_quintic_interpolated = spline_quintic(evaluation_points)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(evaluation_points, fa(evaluation_points), label='Función Original')
plt.plot(evaluation_points, lagrange_interpolated, label='Interpolación Lagrange')
plt.plot(evaluation_points, spline_cubic_interpolated, label='Spline Cúbico')
plt.plot(evaluation_points, spline_quintic_interpolated, label='Spline Quíntico')
plt.scatter(interpolation_points, real_values, color='red', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('fa(x)')
plt.title('Comparación de Métodos de Interpolación')
plt.legend()
plt.grid(True)
plt.show()

# %%
# PUNTOS NO EQUIESPACAIDOS
import numpy as np
from scipy.interpolate import lagrange, CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt

# Definir la función original fa(x)
def fa(x):
    return 0.05 * abs(x) * np.sin(5 * x) + np.tanh(2 * x) + 2

# Definir los puntos de interpolación no equiespaciados
interpolation_points = np.array([-2, -1, 0, 0.5, 1.2, 2, 2.5])

# Calcular los valores reales de la función en los puntos de interpolación
real_values = fa(interpolation_points)

# Realizar interpolación con polinomio de Lagrange
lagrange_poly = lagrange(interpolation_points, real_values)

# Realizar interpolación con spline cúbico
spline_cubic = CubicSpline(interpolation_points, real_values)

# Realizar interpolación con spline quíntico (spline cúbico natural)
spline_quintic = PchipInterpolator(interpolation_points, real_values)

# Puntos donde se evaluarán los resultados
evaluation_points = np.linspace(-3, 3, 300)  # Más puntos para una representación suave

# Evaluar los métodos de interpolación en los puntos de evaluación
lagrange_interpolated = lagrange_poly(evaluation_points)
spline_cubic_interpolated = spline_cubic(evaluation_points)
spline_quintic_interpolated = spline_quintic(evaluation_points)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(evaluation_points, fa(evaluation_points), label='Función Original')
plt.plot(evaluation_points, lagrange_interpolated, label='Interpolación Lagrange')
plt.plot(evaluation_points, spline_cubic_interpolated, label='Spline Cúbico')
plt.plot(evaluation_points, spline_quintic_interpolated, label='Spline Quíntico')
plt.scatter(interpolation_points, real_values, color='red', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('fa(x)')
plt.title('Comparación de Métodos de Interpolación')
plt.legend()
plt.grid(True)
plt.show()

# %%
# hago un comentario

#hola franny te amo

# <3

import numpy as np
import math
from math import exp

def functionA(value):
    return 0.05 * abs(value) * np.sin(5 * value) + np.tanh(2 * value) + 2

def functionB(x1, x2):
    return 0.7 * exp((-(((9*x1)-2)**2)/4) - (((9*x2) - 2)**2)/4) + 0.45*exp((-(((9*x1)+1)**2)/9) - (((9*x2)+1)**2)/5) + 0.55*exp((-(((9*x1)-6)**2)/4)-(((9*x2)-3)**2)/4) - 0.01 * exp((-(((9*x1)-7)**2)/4)-(((9*x2)-3)**2)/4)

#puntos de interpolación de functionA en el intervalo [-3, 3]
interpolation_pointsA = np.array([-2, -1, 0, 0.5, 1.2, 2, 2.5])

#puntos de interpolación de functionB en el intervalo [-1, 1]
interpolation_pointsBX1 = np.array([-0.5, -0.25, 0.1, 0.4, 0.5, 0.7, 0.9])
#interpolation_pointsBX1= np.array([(-0.9,-0.7),(-0.5,-0.4), (-0.25,-0.10),(0.1,0.3),(0.4,0.45),(0.7,0.9)])
interpolation_pointsBx2= np.array([-0.9,-0.7,-0.45,-0.10,0.25,0.30,0.75])
# Calcular los valores reales de la funciónA en los puntos de interpolación
real_valuesA = functionA(interpolation_pointsA)

# Calcular los valores reales de la funciónB en los puntos de interpolación
real_valuesB = functionB(interpolation_pointsBX1, interpolation_pointsBx2) #ARREGLAR!!!!

# Realizar interpolación con polinomio de Lagrange
lagrange_polyA = lagrange(interpolation_pointsA, real_valuesA)

# Realizar interpolación con polinomio de Lagrange
lagrange_polyB = lagrange(interpolation_pointsBX1, interpolation_pointsBx2, real_valuesB)

# Realizar interpolación con spline cúbico
spline_cubicA = CubicSpline(interpolation_pointsA, real_valuesA)

# Realizar interpolación con spline cúbico
spline_cubicB = CubicSpline(interpolation_pointsBX1, interpolation_pointsBx2, real_valuesB)

# Realizar interpolación con spline quíntico (spline cúbico natural)
spline_quinticA = PchipInterpolator(interpolation_pointsA, real_valuesA)

# Realizar interpolación con spline quíntico (spline cúbico natural)
spline_quinticB = PchipInterpolator(interpolation_pointsBX1, interpolation_pointsBx2, real_valuesB)

# Puntos donde se evaluarán los resultados
evaluation_pointsA = np.linspace(-3, 3, 300)  # Más puntos para una representación suave

# Puntos donde se evaluarán los resultados
evaluation_pointsB = np.linspace(-1, 1, 300)  # Más puntos para una representación suave

# Evaluar los métodos de interpolación en los puntos de evaluación
lagrange_interpolatedA = lagrange_polyA(evaluation_pointsA)
lagrange_interpolatedB = lagrange_polyB(evaluation_pointsB)
spline_cubic_interpolatedA = spline_cubicA(evaluation_pointsA)
spline_cubic_interpolatedB = spline_cubicB(evaluation_pointsB)
spline_quintic_interpolatedA = spline_quinticA(evaluation_pointsA)
spline_quintic_interpolatedB = spline_quinticB(evaluation_pointsB)

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
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(evaluation_pointsB, functionB(evaluation_pointsB), label='Función Original')
plt.plot(evaluation_pointsB, lagrange_interpolatedB, label='Interpolación Lagrange')
plt.plot(evaluation_pointsB, spline_cubic_interpolatedB, label='Spline Cúbico')
plt.plot(evaluation_pointsB, spline_quintic_interpolatedB, label='Spline Quíntico')
plt.scatter(interpolation_pointsBX1, interpolation_pointsBx2, real_valuesB, color='red', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('functionB(x)')
plt.title('Comparación de Métodos de Interpolación')
plt.legend()
plt.grid(True)
plt.show()
# %%

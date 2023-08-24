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
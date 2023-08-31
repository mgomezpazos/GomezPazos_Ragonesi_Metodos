import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from scipy.interpolate import CubicSpline, PchipInterpolator, RectBivariateSpline

def functionB(x1, x2):
    return 0.7 * exp((-(((9*x1)-2)**2)/4) - (((9*x2) - 2)**2)/4) + 0.45*exp((-(((9*x1)+1)**2)/9) - (((9*x2)+1)**2)/5) + 0.55*exp((-(((9*x1)-6)**2)/4)-(((9*x2)-3)**2)/4) - 0.01 * exp((-(((9*x1)-7)**2)/4)-(((9*x2)-3)**2)/4)

#puntos de interpolación de functionB en el intervalo [-1, 1]
x1 = np.linspace(-1, 1, 10)
x2 = np.linspace(-1, 1, 10)
X1, X2 = np.meshgrid(x1, x2)
interpolation_pointsB = [(x1_value, x2_value) for x1_value in x1 for x2_value in x2]
real_valuesB = [functionB(x1_value, x2_value) for x1_value, x2_value in interpolation_pointsB]
# Realizar interpolación con spline cúbico
#spline_cubicB = CubicSpline(interpolation_pointsB, real_valuesB)
# Realizar interpolación con spline quíntico (spline cúbico natural)
#spline_quinticB = PchipInterpolator(interpolation_pointsB, real_valuesB)
# Create 2D interpolator
interp_B = RectBivariateSpline(x1, x2, real_valuesB, kx=3, ky=3)
# Puntos donde se evaluarán los resultados
evaluation_pointsB = np.linspace(-1, 1, 300) 
# Evaluar los métodos de interpolación en los puntos de evaluación
#spline_cubic_interpolatedB = spline_cubicB(evaluation_pointsB)
#spline_quintic_interpolatedB = spline_quinticB(evaluation_pointsB)
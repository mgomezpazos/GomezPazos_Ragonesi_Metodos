import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from scipy.interpolate import CubicSpline, PchipInterpolator, RectBivariateSpline, griddata

def functionB(x1, x2):
    return (0.7 * np.exp((-(((9*x1)-2)**2)/4) - (((9*x2) - 2)**2)/4)+
            0.45*np.exp((-(((9*x1)+1)**2)/9) - (((9*x2)+1)**2)/5)+
            0.55*np.exp((-(((9*x1)-6)**2)/4)-(((9*x2)-3)**2)/4)-
            0.01 * np.exp((-(((9*x1)-7)**2)/4)-(((9*x2)-3)**2)/4))

#puntos equiespaciados
x1 = np.linspace(-1, 1, 10)
x2 = np.linspace(-1, 1, 10)
X1, X2 = np.meshgrid(x1, x2)
Z = functionB(X1, X2) #valores reales

#puntos para interpolar
x1_interp = np.linspace(-1, 1, 100)
x2_interp = np.linspace(-1, 1, 100)
X1_interp, X2_interp = np.meshgrid(x1_interp, x2_interp)
points = np.column_stack((X1_interp.ravel(), X2_interp.ravel()))

#Interpolacion cubica con griddata
Z_interp_equi = griddata((X1.ravel(), X2.ravel()), Z.ravel(), points, method='cubic')
Z_interp_equi = Z_interp_equi.reshape(X1_interp.shape)

#graficos
fig, axes = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': '3d'})

surface1 = axes[0].plot_surface(X1, X2, Z, cmap='viridis')
axes[0].set_title('Función Original')

#barra de variación
cbar1 = plt.colorbar(surface1, ax=axes[0], shrink=0.5, aspect=10)
cbar1.set_label('Variación de Z')

surface2 = axes[1].plot_surface(X1_interp, X2_interp, Z_interp_equi, cmap='viridis')
axes[1].set_title('Interpolación con puntos equiespaciados')

#barra de variación
cbar2 = plt.colorbar(surface2, ax=axes[1], shrink=0.5, aspect=10)
cbar2.set_label('Variación de Z')

plt.show()

#calculamos el error relativo
# evaluation_pointsA = np.linspace(-3, 3, 300) 
# # Evaluar los métodos de interpolación en los puntos de evaluación
# lagrange_interpolatedA = lagrange_polyA(evaluation_pointsA)
# spline_cubic_interpolatedA = spline_cubicA(evaluation_pointsA)
# spline_quintic_interpolatedA = spline_quinticA(evaluation_pointsA)
# # Calculate relative errors
# relative_error_lagrange = np.abs(lagrange_interpolatedA - functionA(evaluation_pointsA)) / np.abs(functionA(evaluation_pointsA))
# relative_error_spline_cubic = np.abs(spline_cubic_interpolatedA - functionA(evaluation_pointsA)) / np.abs(functionA(evaluation_pointsA))
# relative_error_spline_quintic = np.abs(spline_quintic_interpolatedA - functionA(evaluation_pointsA)) / np.abs(functionA(evaluation_pointsA))

error_relativo = np.abs(Z - Z_interp_equi) / np.abs(Z)
average_error = np.mean(error_relativo)

# Mostrar el error relativo promedio
print("Error Relativo Promedio:", average_error)

# Gráfico del error relativo
plt.imshow(error_relativo, cmap='viridis', extent=(-1, 1, -1, 1), origin='lower')
plt.colorbar(label='Error Relativo')
plt.title('Error Relativo entre la Función Original y la Interpolación')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Puntos de Chebyshev no equiespaciados
x1_cheb = chebpts2(10)
x2_cheb = chebpts2(10)
X1_cheb, X2_cheb = np.meshgrid(x1_cheb, x2_cheb)
Z_cheb = functionB(X1_cheb, X2_cheb)

# Puntos para interpolar (también en Chebyshev)
x1_interp_cheb = chebpts2(100)
x2_interp_cheb = chebpts2(100)
X1_interp_cheb, X2_interp_cheb = np.meshgrid(x1_interp_cheb, x2_interp_cheb)
points_cheb = np.column_stack((X1_interp_cheb.ravel(), X2_interp_cheb.ravel()))

# Interpolación cúbica con griddata
Z_interp_cheb = griddata((X1_cheb.ravel(), X2_cheb.ravel()), Z_cheb.ravel(), points_cheb, method='cubic')
Z_interp_cheb = Z_interp_cheb.reshape(X1_interp_cheb.shape)

# Gráficos
fig, axes = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': '3d'})

# Gráfico 1: Función Original con puntos Chebyshev
surface1 = axes[0].plot_surface(X1_cheb, X2_cheb, Z_cheb, cmap='viridis')
axes[0].set_title('Función Original (Chebyshev)')

# Barra de variación
cbar1 = plt.colorbar(surface1, ax=axes[0], shrink=0.5, aspect=10)
cbar1.set_label('Variación de Z')

# Gráfico 2: Interpolación con puntos Chebyshev
surface2 = axes[1].plot_surface(X1_interp_cheb, X2_interp_cheb, Z_interp_cheb, cmap='viridis')
axes[1].set_title('Interpolación con puntos Chebyshev')

# Barra de variación
cbar2 = plt.colorbar(surface2, ax=axes[1], shrink=0.5, aspect=10)
cbar2.set_label('Variación de Z')

plt.show()

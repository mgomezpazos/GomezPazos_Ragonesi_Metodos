import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from scipy.interpolate import CubicSpline, PchipInterpolator, RectBivariateSpline, griddata
from numpy.polynomial.chebyshev import chebpts2
from mpl_toolkits.mplot3d import Axes3D

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

# Calculate relative error
relative_error = np.abs(Z_interp_equi - functionB(X1_interp, X2_interp)) / np.abs(functionB(X1_interp, X2_interp))

# Plot relative error
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X1_interp, X2_interp, relative_error, cmap='viridis')
plt.title('Relative Error of Cubic Interpolation')
cbar = plt.colorbar(surface, shrink=0.5, aspect=10)
cbar.set_label('Relative Error')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
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

# Calculate relative error for Chebyshev interpolation
relative_error_cheb = np.abs(Z_interp_cheb - functionB(X1_interp_cheb, X2_interp_cheb)) / np.abs(functionB(X1_interp_cheb, X2_interp_cheb))

# Plot relative error for Chebyshev interpolation
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X1_interp_cheb, X2_interp_cheb, relative_error_cheb, cmap='viridis')
plt.title('Relative Error of Chebyshev Interpolation')
cbar = plt.colorbar(surface, shrink=0.5, aspect=10)
cbar.set_label('Relative Error')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()

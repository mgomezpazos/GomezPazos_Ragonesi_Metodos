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

axes[0].plot_surface(X1, X2, Z, cmap='viridis')
axes[0].set_title('Función Original')

axes[1].plot_surface(X1_interp, X2_interp, Z_interp_equi, cmap='viridis')
axes[1].set_title('Interpolación con puntos equiespaciados')

plt.show()


if _name_ == "_main_":
    main()







# #se crea una malla 3d
# x1 = np.linspace(-1, 1, 10)
# x2 = np.linspace(-1, 1, 10)
# X1, X2 = np.meshgrid(x1, x2)
# Z = functionB(x1, x2)
# #interpolation_pointsB = [(x1_value, x2_value) for x1_value in x1 for x2_value in x2]
# #real_valuesB = [functionB(x1_value, x2_value) for x1_value, x2_value in interpolation_pointsB]

# #defino los puntos donde voy a hacer la interpolación 
# x1_interp = np.linspace(-1, 1, 100)
# x2_interp = np.linspace(-1, 1, 100)
# X1_interp, X2_interp = np.meshgrid(x1_interp, x2_interp)
# points = np.column_stack((X1_interp.ravel(), X2_interp.ravel()))

# # Realizar la interpolación
# Z_interp = griddata((X1.ravel(), X2.ravel()), Z.ravel(), points, method='cubic')
# Z_interp = Z_interp.reshape(X1_interp.shape)

# # Graficar los resultados
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot_surface(X1, X2, Z, cmap='viridis')
# ax1.set_title('Función Original')

# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(X1_interp, X2_interp, Z_interp, cmap='viridis')
# ax2.set_title('Interpolación')

# # Realizar interpolación con spline cúbico
# #spline_cubicB = CubicSpline(interpolation_pointsB, real_valuesB)
# # Realizar interpolación con spline quíntico (spline cúbico natural)
# #spline_quinticB = PchipInterpolator(interpolation_pointsB, real_valuesB)
# # Create 2D interpolator
# interp_B = RectBivariateSpline(x1, x2, real_valuesB, kx=3, ky=3)
# # Puntos donde se evaluarán los resultados
# evaluation_pointsB = np.linspace(-1, 1, 300) 
# # Evaluar los métodos de interpolación en los puntos de evaluación
# #spline_cubic_interpolatedB = spline_cubicB(evaluation_pointsB)
# #spline_quintic_interpolatedB = spline_quinticB(evaluation_pointsB)
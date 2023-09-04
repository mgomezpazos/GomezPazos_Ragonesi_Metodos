#--------------------------------------------------------------------EJERCICIO 1------------------------------------------------------------------------------------------------
import csv
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import lagrange, CubicSpline
from scipy.optimize import newton
#-------------------------------------------------------------------MANEJO DE ARCHIVOS--------------------------------------------------------------------------------------------
# Creación de listas vacías
x_ground_truth = []
y_ground_truth = []
x1_ti = []
x2_ti = []
#Lectura de archivos y almacenamiento de los datos
try:
    with open('mnyo_ground_truth.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        
        for row in csv_reader:
            valores = row[0].split()
            x_ground_truth.append(float(valores[0]))
            y_ground_truth.append(float(valores[1]))
except FileNotFoundError:
    print("El archivo CSV no se encontró.")
except Exception as e:
    print(f"Se produjo un error: {str(e)}")

try:
    with open('mnyo_mediciones.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        
        for row in csv_reader:
            valores = row[0].split()
            x1_ti.append(float(valores[0]))
            x2_ti.append(float(valores[1]))
except FileNotFoundError:
    print("El archivo CSV no se encontró.")
except Exception as e:
    print(f"Se produjo un error: {str(e)}")

#--------------------------------------------------------------------INTERPOLACIÓN---------------------------------------------------------------------------------------------------------
#necesito la variable y el f(variable)
# lo primero que le paso a lagrange es el tiemp(0;range(long lista x))
#lo segundo que le paso son las mediciones de x
#hago lo mismo para y
#ahi tengo una x(t) y una y(t)
#ahi ploteo la curva paramétrica (chat gpt)

# Array con valores de tiempo
time_values = np.linspace(0, len(x1_ti) - 1, 10)
#Lagrange con el tiempo y las mediciones de x
x_t = lagrange(time_values, x1_ti)
y_t = lagrange(time_values, x2_ti)
# Valores de x e y usando polinomios de Lagrange
x_values = [x_t(t) for t in time_values]
y_values = [y_t(t)for t in time_values]
#Cálculo de uno de los límites del terreno
def custom_function(x1):
    return 3.6 - 0.35 * x1
x1_custom = np.linspace(0, 20, 100)
x2_custom = custom_function(x1_custom)
indices_within_limits = np.where((x2_custom >= min(y_ground_truth)) & (x2_custom <= max(y_ground_truth)))
# Spline Cubico para x1_ti y x2_ti
cs_x1 = CubicSpline(time_values, x1_ti)
cs_x2 = CubicSpline(time_values, x2_ti)
# Valores para interpolar
time_interp = np.linspace(0, len(x1_ti) - 1, 1000)
# Valores de Y interpolados
x1_interp = cs_x1(time_interp)
x2_interp = cs_x2(time_interp)
# Ploteo
plt.figure(figsize=(10, 6))
plt.plot(x1_ti, x2_ti, 'o', label='Data Points', color='purple')
plt.plot(x_values, y_values, label='Interpolated Function', color='green')
plt.plot(x_ground_truth, y_ground_truth, label='Ground Truth Function', linestyle='--', color = "pink")
plt.plot(x1_interp, x2_interp, label='Cubic Spline Interpolation', color='magenta')
plt.plot([10] * len(x_values), y_values, label='x1 = 10', linestyle='-.', color='blue', linewidth=2)  # Add x1 = 10
plt.plot(x1_custom[indices_within_limits], x2_custom[indices_within_limits], label='0.35*x1 + x2 = 3.6', linestyle='-.', color='grey')
plt.xlabel('X(t)')
plt.ylabel('Y(t)')
plt.title('Comparison of Interpolated Linear and Ground Truth Functions')
plt.legend()
plt.grid(True)
plt.show()
# Ploteo
plt.figure(figsize=(10, 6))
plt.scatter(x1_ti, x2_ti, label='Data Points', color='purple')
plt.plot(x_ground_truth, y_ground_truth, label='Ground Truth Function', linestyle='--', color='green')
plt.plot([10] * len(x1_interp), x2_interp, label='x1 = 10', linestyle='-.', color='red', linewidth=2)  # Add x1 = 10
plt.plot(x1_custom[indices_within_limits], x2_custom[indices_within_limits], label='0.35*x1 + x2 = 3.6', linestyle='-.', color='orange')
plt.plot(x1_interp, x2_interp, label='Cubic Spline Interpolation', color='blue')
plt.xlabel('X1(t)')
plt.ylabel('X2(t)')
plt.title('Comparison of Interpolated Spline and Ground Truth Functions')
plt.legend()
plt.grid(True)
plt.show()
#------------------------------------------------------------------BÚSQUEDA DE RAÍCES---------------------------------------------------------------------------------------------
# Define la función que representa la línea recta
def line_equation(x, y):
    return 0.35 * x + y - 3.6

# Función para encontrar un punto de cruce utilizando el método de Newton-Raphson
def find_intersection(interpolation_function, line_equation, x_guess, y_guess):
    # Función que representa la diferencia entre la función interpolada y la línea recta
    def equation_to_solve(point):
        x, y = point
        return interpolation_function(x) - line_equation(x, y)

    # Encuentra las coordenadas (x, y) donde la función cruza la línea recta
    intersection_point = newton(equation_to_solve, (x_guess, y_guess))

    return intersection_point

# Supongamos que deseas encontrar el primer punto de cruce en el punto inicial (x, y)
x_initial_guess = 5.0
y_initial_guess = 2.0

# Encuentra el primer punto de cruce
first_intersection_point = find_intersection(x_t, line_equation, x_initial_guess, y_initial_guess)

print("Primer punto de cruce:", first_intersection_point)

# Modifica la ecuación para excluir el primer punto de cruce
def modified_line_equation(x, y):
    return 0.35 * x + y - 3.6 if (x, y) != first_intersection_point else 0

# Supongamos que deseas encontrar el segundo punto de cruce en el punto inicial (x, y)
x_initial_guess = 10.0
y_initial_guess = 4.0

# Encuentra el segundo punto de cruce
second_intersection_point = find_intersection(x_t, modified_line_equation, x_initial_guess, y_initial_guess)

print("Segundo punto de cruce:", second_intersection_point)

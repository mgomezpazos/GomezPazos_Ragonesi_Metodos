# -----------------------------------EJERCICIO 1: MANEJO DE ARCHIVOS-----------------------------------------------------------------------------------------------------------------

import csv
from tabulate import tabulate

# Listas para almacenar los datos
x_ground_truth = []
y_ground_truth = []

try:
    with open('mnyo_ground_truth.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        
        for row in csv_reader:
            # Dividir la fila en dos valores y convertirlos a números flotantes
            valores = row[0].split()
            x_ground_truth.append(float(valores[0]))
            y_ground_truth.append(float(valores[1]))
except FileNotFoundError:
    print("El archivo CSV no se encontró.")
except Exception as e:
    print(f"Se produjo un error: {str(e)}")

# Ahora deberías tener los datos en x_ground_truth y y_ground_truth

# Combinar las listas en una lista de tuplas
datos = list(zip(x_ground_truth, y_ground_truth))

# Imprimir la tabla
tabla = tabulate(datos, headers=["Columna 1", "Columna 2"], tablefmt="pretty")
print(tabla)

# Listas para almacenar los datos
x1_ti = []
x2_ti = []

try:
    with open('mnyo_mediciones.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        
        for row in csv_reader:
            # Dividir la fila en dos valores y convertirlos a números flotantes
            valores = row[0].split()
            x_ground_truth.append(float(valores[0]))
            y_ground_truth.append(float(valores[1]))
except FileNotFoundError:
    print("El archivo CSV no se encontró.")
except Exception as e:
    print(f"Se produjo un error: {str(e)}")

# Ahora deberías tener los datos en x_ground_truth y y_ground_truth

# Combinar las listas en una lista de tuplas
datos = list(zip(x_ground_truth, y_ground_truth))

# Imprimir la tabla
tabla = tabulate(datos, headers=["X1(ti)", "X2(ti)"], tablefmt="pretty")
print(tabla)

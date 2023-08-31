import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from scipy.interpolate import lagrange, CubicSpline, PchipInterpolator

def functionB(x1, x2):
    return 0.7 * exp((-(((9*x1)-2)**2)/4) - (((9*x2) - 2)**2)/4) + 0.45*exp((-(((9*x1)+1)**2)/9) - (((9*x2)+1)**2)/5) + 0.55*exp((-(((9*x1)-6)**2)/4)-(((9*x2)-3)**2)/4) - 0.01 * exp((-(((9*x1)-7)**2)/4)-(((9*x2)-3)**2)/4)

inter
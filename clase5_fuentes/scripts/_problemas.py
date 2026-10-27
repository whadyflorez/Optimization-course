"""
Problemas de prueba de la Clase 5 (convencion:  min f(x)  s.a.  h(x)=0,  g(x)<=0).

P1 (solo igualdad, tipo Nocedal & Wright cap. 12):
    min  x1 + x2      s.a.  h(x) = x1^2 + x2^2 - 2 = 0
    minimo en (-1,-1), f* = -2, lambda* = 1/2   (el punto (1,1) tambien cumple KKT: es el MAXIMO)

P2 (solo desigualdades, problema convexo con las DOS restricciones activas):
    min  f(x) = (x1-2)^2 + (x2-1)^2
    s.a. g1(x) = x1^2 - x2   <= 0
         g2(x) = x1 + x2 - 2 <= 0
    minimo en (1,1), f* = 1, mu* = (2/3, 2/3)
"""
import numpy as np

# ------------------------------------------------------------------ P1
def f1(x):      return x[0] + x[1]
def grad1(x):   return np.array([1.0, 1.0])
def hess1(x):   return np.zeros((2, 2))
def h1(x):      return np.array([x[0]**2 + x[1]**2 - 2.0])
def Jh1(x):     return np.array([[2 * x[0], 2 * x[1]]])
def hessh1(x):  return [2 * np.eye(2)]

# ------------------------------------------------------------------ P2
def f2(x):      return (x[0] - 2)**2 + (x[1] - 1)**2
def grad2(x):   return np.array([2 * (x[0] - 2), 2 * (x[1] - 1)])
def hess2(x):   return 2 * np.eye(2)
def g2(x):      return np.array([x[0]**2 - x[1], x[0] + x[1] - 2.0])
def Jg2(x):     return np.array([[2 * x[0], -1.0], [1.0, 1.0]])
def hessg2(x):  return [np.array([[2.0, 0.0], [0.0, 0.0]]), np.zeros((2, 2))]

X1_STAR = np.array([-1.0, -1.0]); LAM1_STAR = 0.5
X2_STAR = np.array([1.0, 1.0]);   MU2_STAR = np.array([2 / 3, 2 / 3]); F2_STAR = 1.0

# ------------------------------------------------------------------ P3
# Misma f que P2 pero con UNA restriccion curva: el disco x1^2 + x2^2 <= 1.
# El optimo cae en la frontera curva (no es un "vertice"):  x* = (2,1)/sqrt(5), f* = (sqrt(5)-1)^2, mu* = sqrt(5)-1
def g3(x):      return np.array([x[0]**2 + x[1]**2 - 1.0])
def Jg3(x):     return np.array([[2 * x[0], 2 * x[1]]])
def hessg3(x):  return [2 * np.eye(2)]
X3_STAR = np.array([2.0, 1.0]) / np.sqrt(5); F3_STAR = (np.sqrt(5) - 1)**2; MU3_STAR = np.sqrt(5) - 1

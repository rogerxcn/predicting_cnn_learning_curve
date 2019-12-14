import numpy as np

###############################################
## Curve Definition
###############################################

def Vapor_Pressure(x, a, b, c):
    return np.exp(a + b/x + c*np.log(x))

def Pow3(x, c, a, alpha):
    return c - a * np.power(x, -alpha)

def Log_Log_Linear(x, a, b):
    return np.log(np.abs(a * np.log(x) + b))

def Log_Power(x, a, b, c):
    return a/(1+np.power(x/np.exp(b), c))

def Pow4(x, a, b, c, alpha):
    return c - np.power(np.abs(a*x+b), -alpha)

def MMF(x, alpha, beta, delta, kar):
    return alpha - (alpha-beta)/(1+np.power(np.abs(kar*x), delta))

def Exp4(x, a, b, c, alpha):
    return c - np.exp(-a*np.power(x, alpha) + b)

def Janoschek(x, alpha, beta, delta, kar):
    return alpha - (alpha - beta)*np.exp(-kar*np.power(x, delta))

def Weibull(x, alpha, beta, delta, kar):
    return alpha - (alpha - beta) * np.exp(-np.power(np.abs(kar*x), delta))

def ILog2(x, a, b):
    return b - a/(np.log(x + 1e-10))

def curve_set():
    return [Vapor_Pressure, Pow3, Log_Log_Linear, Log_Power, Pow4, MMF, Exp4, Janoschek, Weibull, ILog2]

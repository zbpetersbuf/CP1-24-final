import os
import math as m
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
import glob

def fit_BS(v,w_e,x_e,c):
    #return w_e*(v+1/2) - w_e*x_e*(v+1/2)**2
    return w_e-2*w_e*x_e*(v+2)+c

def Birge_Sponer_funk(v):
    lamdas = [5224, 5239, 5253, 5267, 5282, 5298, 5314, 5331, 5350, 5368, 5387, 5406, 5428, 5449, 5472, 5494, 5519, 5546, 5571, 5597,
              5624, 5652, 5680, 5709, 5739, 5771, 5801, 5833, 5866, 5898, 5933, 5968, 6004, 6040, 6075, 6112]
    
    #lamdas = [5224, 5239, 5253, 5267, 5282, 5298, 5314, 5331, 5350, 5368, 5387, 5406, 5428, 5449,
          #5472, 5494, 5519, 5546, 5571, 5597, 5624, 5652, 5680, 5709, 5739, 5771]
    
    #lamdas = [1.10608*l/0.985 + 35.505 -556.13681 for l in lamdas]
    #lamdas = lamdas[::-1]

    G_v = [1/(x*10**(-8)) for x in lamdas]

    w_e_guess = 197.99
    x_e_guess = 0.0020
    c_guess = 1000


    popt, _ = curve_fit(fit_BS, v, G_v, p0=[w_e_guess, x_e_guess,c_guess])
    
    return popt


def Birge_Sponer_funk(v):
    lamdas = [5224, 5239, 5253, 5267, 5282, 5298, 5314, 5331, 5350, 5368, 5387, 5406, 5428, 5449, 5472, 5494, 5519, 5546, 5571, 5597,
              5624, 5652, 5680, 5709, 5739, 5771, 5801, 5833, 5866, 5898, 5933, 5968, 6004, 6040, 6075, 6112]
    
    #lamdas = [5224, 5239, 5253, 5267, 5282, 5298, 5314, 5331, 5350, 5368, 5387, 5406, 5428, 5449,
          #5472, 5494, 5519, 5546, 5571, 5597, 5624, 5652, 5680, 5709, 5739, 5771]
    
    #lamdas = [1.10608*l/0.985 + 35.505 -556.13681 for l in lamdas]
    #lamdas = lamdas[::-1]

    G_v = [1/(x*10**(-8)) for x in lamdas]

    w_e_guess = 197.99
    x_e_guess = 0.0020
    c_guess = 1000


    popt, _ = curve_fit(fit_BS, v, G_v, p0=[w_e_guess, x_e_guess,c_guess])
    
    return popt
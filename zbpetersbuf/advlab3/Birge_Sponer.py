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



def fit_BS_v3(v,w_e,x_e,c):
    #return w_e*(v+1/2) - w_e*x_e*(v+1/2)**2
    return w_e-2*w_e*x_e*(v+2)+c


def Birge_Sponer_funk_v3(v):
    lamdas = [5224, 5239, 5253, 5267, 5282, 5298, 5314, 5331, 5350, 5368, 5387, 5406, 5428, 5449, 5472, 5494, 5519, 5546,
              5571, 5597, 5624, 5652, 5680, 5709, 5739, 5771, 5801, 5833, 5866, 5898, 5933, 5968, 6004, 6040, 6075, 6112]
    
    #lamdas = [5224, 5239, 5253, 5267, 5282, 5298, 5314, 5331, 5350, 5368, 5387, 5406, 5428, 5449,
          #5472, 5494, 5519, 5546, 5571, 5597, 5624, 5652, 5680, 5709, 5739, 5771]
    
    #lamdas = [1.10608*l/0.985 + 35.505 -556.13681 for l in lamdas]
    #lamdas = lamdas[::-1]

    G_v_1 = [1/(x*10**(-8)) for x in lamdas]

    i = 0
    l = len(lamdas)
    G_v = np.zeros(l-1)

    while i<l:
        G_v[i] = G_v_1[i]-G_v_1[i+1]
        i+=1
    w_e_guess = 197.99
    x_e_guess = 0.0020
    c_guess = 1000
    popt, _ = curve_fit(fit_BS_v2, v, G_v, p0=[w_e_guess, x_e_guess,c_guess])
    
    return popt, G_v


def fit_BS_v2(v,w_e,x_e):
    #return w_e*(v+1/2) - w_e*x_e*(v+1/2)**2
    return w_e-2*w_e*x_e*(v+2)


def Birge_Sponer_funk_v2(v):    
    lamdas = [5172, 5186, 5200, 5215, 5230, 5245, 5261, 5278, 5296, 5314, 5333, 5352, 5373, 5394, 5415, 5439, 5464, 5490,
          5515, 5541, 5568, 5595, 5623, 5651, 5682, 5712, 5743, 5775, 5807, 5840, 5873, 5908, 5943, 5980, 6015, 6052]
    lamdas = lamdas[::-1]

    G_v_1 = [1/(x*10**(-8)) for x in lamdas]

    i = 0
    l = len(lamdas)
    G_v = np.zeros(l-1)
    while i<l-1:
        G_v[i] = G_v_1[i+1] - G_v_1[i]
        i+=1

    w_e_guess = 197.99
    x_e_guess = 0.0020
    popt, _ = curve_fit(fit_BS_v2, v, G_v, p0=[w_e_guess, x_e_guess])
    w_e = popt[0]
    x_e=popt[1]
    return w_e, x_e, G_v



def Birge_Sponer_funk_v4(v):    
    lamdas = [5172, 5186, 5200, 5215, 5230, 5245, 5261, 5278, 5296, 5314, 5333, 5352, 5373, 5394, 5415, 5439, 5464, 5490,
              5515, 5541, 5568, 5595, 5623, 5651, 5682, 5712, 5743, 5775, 5807, 5840, 5873, 5908, 5943, 5980, 6015, 6052]
    lamdas = lamdas[::-1]

    G_v_1 = [1/(x*10**(-8)) for x in lamdas]
    G_v = np.array([G_v_1[i+1] - G_v_1[i] for i in range(len(G_v_1)-1)])

    w_e_guess = 197.99
    x_e_guess = 0.0020
    popt, pcov = curve_fit(fit_BS_v2, v, G_v, p0=[w_e_guess, x_e_guess])
    perr = np.sqrt(np.diag(pcov))  # standard deviations (errors) of the fit parameters

    w_e, x_e = popt
    w_e_err, x_e_err = perr
    return w_e, x_e, w_e_err, x_e_err, G_v

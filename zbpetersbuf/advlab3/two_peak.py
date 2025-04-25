import os
import math as m
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
import glob


def twin_gaussian(x, a, mu1, b, c, mu2, d, e):
    return a * np.exp(-(x - mu1) ** 2 / (2 * b ** 2)) + c * np.exp(-(x - mu2) ** 2 / (2 * d ** 2)) + e

def fit_twin_gaussian(xax, yax, x_min, x_max):

    coords = [(x, y) for x, y in zip(xax, yax)]
    filtered_coords = [(x, y) for x, y in coords if x_min <= x <= x_max]
    
    if not filtered_coords:
        return None

    x_data, y_data = zip(*filtered_coords)

    #a_guess = max(y_data) - min(y_data) + (max(y_data) + min(y_data))/4
    #mu1_guess = np.mean(x_data) - (max(y_data) + min(y_data))/4
    #b_guess = (max(x_data) - min(x_data)) / 4 
    #c_guess = max(y_data) - min(y_data) - (max(y_data) + min(y_data))/4
    #mu2_guess = np.mean(x_data) + (max(y_data) + min(y_data))/4
    #d_guess = (max(x_data) - min(x_data)) / 4 
    #e_guess = min(y_data)
    #popt, _ = curve_fit(twin_gaussian, x_data, y_data, p0=[a_guess, mu1_guess, b_guess, c_guess, mu2_guess, d_guess, e_guess])
    
    popt, _ = curve_fit(twin_gaussian, x_data, y_data, p0=[0.15, 5802, 1, 0.1, 5808, 1, -0.1])
    #mu_fit = popt[1]
    return popt
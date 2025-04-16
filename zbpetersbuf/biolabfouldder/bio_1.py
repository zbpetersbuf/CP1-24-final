"""

"""
import re
import os
import glob
import numpy as np
#import pandas as pd
from scipy.optimize import curve_fit
#import workinh as wrk

def filenamelister(exp_name, filetype = '.md'):
    pattern = os.path.join('/workspaces/CP1-24-final/zbpetersbuf/biodata/', f"*{exp_name.strip()}*{filetype.strip()}")
    files = glob.glob(pattern)
    return files

def findmdfromcsv(filepath):
    """
    Extracts and returns a portion of the filename from a given CSV file path.
    The function expects the file path to be in the format 'LL<digits>_<file_name>.csv'.
    It extracts the part after 'LL<digits>_' and before the '.csv' extension, 
    returning it prefixed with an underscore ('_').
    Parameters:
    filepath (str): The path to the CSV file.
    Returns:
    str: The extracted portion of the filename prefixed with an underscore.
    """
    match = re.search(r'LL\d+_(.*)\.csv', filepath)
    file_name = match.group(1)
    return f"_{file_name}"

def gaussian(x, a, b, c, e):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + e

def fit_gaussian(x_data, y_data):
    initial_guess = [max(y_data), np.mean(x_data), np.std(x_data), np.min(y_data)]  # Initial guesses for [a, b, c, e]
    popt, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)  # Fit the Gaussian model
    a_fit, b_fit, c_fit, e_fit = popt  # Optimized parameters
    return b_fit, c_fit  # Return mean (b) and standard deviation (c)

def process_multiple_files(exp_name):
    files = filenamelister(exp_name, '.csv')  # Get list of files with .csv extension
    results = {}
    
    for file in files:
        # Load the data from the CSV file and skip the header (skiprows=1)
        data = np.loadtxt(file, delimiter=',', skiprows=1)  # Skip the header
        x_data = data[:, 0]  # First column: Distance_(microns)
        y_data = data[:, 1]  # Second column: Gray_Value
        
        # Fit the Gaussian and get the mean and std deviation
        mean, std_dev = fit_gaussian(x_data, y_data)
        
        # Store the results with the file name key
        file_key = findmdfromcsv(file)
        results[file_key] = {'mean': mean, 'std_dev': std_dev}
    
    return results
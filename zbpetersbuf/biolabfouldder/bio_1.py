"""

"""
import re
import os
import glob
import numpy as np
import pandas as pd
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
    try:
        popt, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)  # Fit the Gaussian model
        a_fit, b_fit, c_fit, e_fit = popt  # Optimized parameters
        return b_fit, c_fit  # Return mean (b) and standard deviation (c)
    except Exception as e:
        print(f"Warning: Gaussian fitting failed: {e}")
        return np.nan, np.nan  # Return NaN if fitting fails

def process_multiple_files(exp_name):
    files = filenamelister(exp_name, '.csv')  # Get list of files with .csv extension
    tot_mean = 0  # Initialize total mean
    tot_std_dev = 0  # Initialize total standard deviation
    num_files = 20  # Initialize a counter for the number of files processed

    for file in files:
        try:
            data = pd.read_csv(file, header=0)  # Assuming your CSVs have a header row
            
            if data.empty:
                print(f"Warning: File {file} has no data to process!")
                continue

            distance_data = data.iloc[:, 0].values  # First column: Distance_(microns)
            gray_value_data = data.iloc[:, 1].values  # Second column: Gray_Value

            mean, std_dev = fit_gaussian(distance_data, gray_value_data)

            # Sum up the mean and standard deviation
            tot_mean += mean
            tot_std_dev += std_dev

        except Exception as e:
            print(f"Error processing file {file}: {e}")

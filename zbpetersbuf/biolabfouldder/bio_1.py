"""

"""
import re
import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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
    means = []  # List to store the means
    std_devs = []  # List to store the standard deviations

    for file in files:
        try:
            data = pd.read_csv(file, header=0)  # Assuming your CSVs have a header row
            
            if data.empty:
                print(f"Warning: File {file} has no data to process!")
                continue

            distance_data = data.iloc[:, 0].values  # First column: Distance_(microns)
            gray_value_data = data.iloc[:, 1].values  # Second column: Gray_Value

            mean, std_dev = fit_gaussian(distance_data, gray_value_data)

            # Append the mean and standard deviation for this file
            if not np.isnan(mean) and not np.isnan(std_dev):
                means.append(mean)
                std_devs.append(std_dev)
            else:
                print(f"Warning: Gaussian fitting failed for file {file}")

        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Calculate the averages of the means and standard deviations
    if means and std_devs:
        avg_mean = np.mean(means)
        avg_std_dev = np.mean(std_devs)

        # Print the results
        print(f"Means from each file: {means}")
        print(f"Standard Deviations from each file: {std_devs}")
        print(f"Average Mean: {avg_mean}")
        print(f"Average Std Dev: {avg_std_dev}")
    else:
        print("No valid data to calculate averages.")

def fcs():
    # File path to your Excel file
    file_path = '/workspaces/CP1-24-final/zbpetersbuf/biodata/FCS_hundert.xlsx'

    # Read the Excel file, skipping the first two rows
    df = pd.read_excel(file_path, skiprows=1)

    # Assuming your data has columns 'Time' and 'Count Rate Channel 1 [kCounts/s]'
    # You can adjust these column names based on your actual file structure
    x = df['Time']  # Time column (x-values)
    y = df['Count Rate Channel 1 [kCounts/s]']  # Count Rate Channel 1 [kCounts/s] (y-values)

    # Calculate the average of the y-values
    avg_y = y.mean()

    # Subtract the average from each y-value


    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_adjusted, label='Adjusted Count Rate', color='b')
    plt.title('Time vs Adjusted Count Rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Adjusted Count Rate (kCounts/s)')
    plt.legend()
    plt.grid(True)
    plt.show()
import os
import math as m
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
import glob


def get_file_paths(exp_name, filetype, base_dir='/workspaces/CP1-24-final/zbpetersbuf/advlab3data/'):
    """
    This function generates a file path pattern and returns a list of matching file paths
    for the given experiment name (exp_name) and file type (filetype).

    :param exp_name: Experiment name string (e.g., 'merq')
    :param filetype: File type (extension) string (e.g., '.txt')
    :param base_dir: Base directory where the files are stored
    :return: List of file paths matching the pattern
    """
    pattern = os.path.join(base_dir, f"*{exp_name.strip()}*{filetype.strip()}")
    file_paths = glob.glob(pattern)
    return file_paths

def read_file(file_path):
    """
    Reads the content of a file and parses it into a list of tuples (time, value).

    :param file_path: Path to the file to be read
    :return: A list of tuples where each tuple contains (time, value)
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line by the tab character and convert to a tuple of (time, value)
            try:
                time, value = line.strip().split('\t')
                data.append((float(time), float(value)))
            except ValueError:
                continue  # Skip lines that can't be parsed into two floats
    return data

def read_files(exp_name, filetype):
    # List the specific files you want to read
    file_names = [f"merq_1{filetype}", f"merq_2{filetype}", f"merq_3{filetype}", f"merq_4{filetype}", f"merq_5{filetype}"]

    file_contents = {}

    # Loop through the file names and read the data
    for file_name in file_names:
        file_path = os.path.join('/workspaces/CP1-24-final/zbpetersbuf/advlab3data/', file_name)
        
        # Read the data from the file and store it in a dictionary
        data = np.loadtxt(file_path)
        file_contents[file_path] = data

    return file_contents



def gaussian(x, a, mu, sigma, b):
    return a * np.exp(- (x - mu)**2 / (2 * sigma**2)) + b

def fit_gaussian_and_calculate_mean(exp_name, filetype):
    """
    Fit Gaussian curves to data in the specified files and calculate the average of the means.

    Args:
    exp_name (str): The experiment name (e.g., "merq").
    filetype (str): The file extension (e.g., ".txt").

    Returns:
    average_mean (float): The average of the means from all fits.
    """
    
    # Call the function to read the files
    file_contents = read_files(exp_name, filetype)

    # List to store the means of the fits
    means = []

    # Define a set of distinct colors from the 'tab10' colormap
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # If there are more than 10 files, we will cycle through the colors
    if len(file_contents) > 10:
        color_cycle = distinct_colors * (len(file_contents) // 10 + 1)  # Repeat the cycle if more than 10 files
    else:
        color_cycle = distinct_colors[:len(file_contents)]

    # Plot the data and fit Gaussian
    plt.figure(figsize=(10, 6))

    # Iterate over files and data
    for idx, (file_path, data) in enumerate(file_contents.items()):
        # Extract time (x) and value (y) from the data
        x_data = np.array([point[0] for point in data])  # Wavelength in Angstroms
        y_data = np.array([point[1] for point in data])  # Intensity

        # Get the color for this plot from the cycle
        color = color_cycle[idx]

        # Fit the data to the Gaussian function
        try:
            # Initial guess for the parameters: amplitude, mean, std dev, and baseline offset
            initial_guess = [np.max(y_data), np.mean(x_data), np.std(x_data), np.min(y_data)]
            
            # Fit the curve to the data
            popt, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
            
            # Extract the mean (mu) from the fit parameters
            a, mu, sigma, b = popt
            means.append(mu)  # Store the mean value for calculating the average later

            # Plot the original data with the assigned color
            plt.plot(x_data, y_data, label=f"{file_path.split('/')[-1]} - Data", color=color)
            
            # Plot the fitted Gaussian curve with the assigned color
            plt.plot(x_data, gaussian(x_data, *popt), label=f"{file_path.split('/')[-1]} - Fit", color=color, linestyle='--')

        except Exception as e:
            print(f"Error fitting data from {file_path}: {e}")

    # Add labels and title
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Intensity')
    plt.title(f'Intensity vs Wavelength for {exp_name}')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()

    # Calculate the average of the means from all 5 files
    average_mean = np.mean(means)
    print(f"Average mean of the Gaussian fits: {average_mean}")
    
    return average_mean


def round_sf(x, sig_figs):
    if x == 0:
        return 0
    else:
        return round(x, sig_figs - int(m.floor(m.log10(abs(x)))) - 1)
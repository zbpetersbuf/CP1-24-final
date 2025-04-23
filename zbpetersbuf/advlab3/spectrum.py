import os
import math as m
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
import glob


def get_file_path(file_name, base_dir='/workspaces/CP1-24-final/zbpetersbuf/advlab3data/'):
    """
    This function returns the full path for the given file name.

    :param file_name: The file name (e.g., 'merq_full.txt')
    :param base_dir: Base directory where the file is stored
    :return: The full path for the given file name
    """
    file_path = os.path.join(base_dir, file_name)
    
    # Check if the file exists
    if os.path.isfile(file_path):
        return file_path
    else:
        print(f"Warning: {file_name} not found in {base_dir}")
        return None


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


def read_files(file_name):
    """
    Reads a single file and stores its data in a dictionary.

    :param file_name: The file name (e.g., 'merq_full.txt')
    :return: A dictionary with the file path as key and data as value
    """
    file_contents = {}

    # Get the valid file path for the given file name
    file_path = get_file_path(file_name)
    
    if file_path:  # Proceed only if the file exists
        # Read the data from the file and store it in the dictionary
        data = read_file(file_path)
        file_contents[file_path] = data

    return file_contents

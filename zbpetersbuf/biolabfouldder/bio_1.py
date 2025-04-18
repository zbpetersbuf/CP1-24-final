"""

"""
import re
import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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


def fcs3():
    # File path to your Excel file
    file_path = '/workspaces/CP1-24-final/zbpetersbuf/biodata/FCS_hundert.xlsx'

    # Read the Excel file, skipping the first row
    df = pd.read_excel(file_path, skiprows=1)

    df = df.iloc[200:]

    x = df['Time']  # Time column (x-values)
    y = df['Count Rate Channel 1 [kCounts/s]']  # Count Rate Channel 1 [kCounts/s] (y-values)

    correlation = np.correlate(y, x, mode='full')

    lag = np.arange(-len(x) + 1, len(x))

    adv_correlation = correlation.mean()
    correlation = correlation / adv_correlation

    # Plot the correlation with a log scale for the x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(lag, correlation, label='Auto-correlation', color='b')
    plt.title('Auto-correlation of Count Rate vs Time')
    plt.xlabel('Lag (Time Shift)')
    plt.ylabel('Correlation')

    # Set the x-axis to logarithmic scale
    plt.xscale('log')

    # Adjust plot appearance
    plt.legend()
    plt.grid(True)
    plt.show()


def custom_model(x, t_D, t_f,a):
    return ((1 / (1 + x / t_D)) * (1 / (1 + x / (4 * t_D))**(1/2)) * np.exp(-(x / t_f)**2 / (1 + x / t_D))*a)

def fcs(k1, k2):
    file_path = '/workspaces/CP1-24-final/zbpetersbuf/biodata/FCS_hundert.xlsx'

    df = pd.read_excel(file_path, skiprows=1)

    x = df['Time']  # Time column (x-values)
    y = df['Count Rate Channel 1 [kCounts/s]']  # Count Rate Channel 1 [kCounts/s] (y-values)
    
    lag = np.array([i * 10**(-3) for i in x])  # Convert lag to a NumPy array
    # Auto-correlation of y
    correlation = np.correlate(y, y, mode='full')  # Auto-correlation of y

    # Normalize the correlation
    adv_correlation = correlation.mean()  # Maximum correlation for normalization
    correlation = correlation/adv_correlation

    mid_point = len(correlation) // 2
    correlation_positive = correlation[mid_point:]

    mask = (lag > k1) & (lag < k2)
    lag = lag[mask]  # Filtered lag values
    correlation_positive = correlation_positive[mask]  # Filtered correlation values

    p0 = [0.1, 0.02, 0.5]

    # Set bounds for each parameter [lower, upper]
    lower_bounds = [0, 0, 0]  # Lower bounds for [t_D, t_f, a]
    upper_bounds = [1, 1, 200]  # Upper bounds for [t_D, t_f, a]

    # Fit the custom model with bounds
    popt, pcov = curve_fit(custom_model, lag, correlation_positive, p0=p0, bounds=(lower_bounds, upper_bounds))

    t_D_fit, t_f_fit, a_fit = popt  # Unpack fitted parameters
    print(f"Fitted t_D: {t_D_fit}")
    print(f"Fitted t_f: {t_f_fit}")
    print(f"Fitted a: {a_fit}")

    # Compute the fitted correlation using the fitted parameters
    fitted_correlation = custom_model(lag, t_D_fit, t_f_fit, a_fit)

    plt.figure(figsize=(10, 6))
    plt.plot(lag, correlation_positive, 'b.', label='Auto-correlation Data')  # Plot original data points
    plt.plot(lag, fitted_correlation, 'r-', label='Fitted Curve')  # Plot fitted curve
    plt.title('Auto-correlation of Count Rate vs Time and Fitted Model')
    plt.xlabel('Lag (Time Shift)')
    plt.ylabel('Correlation')
    plt.xscale('log')  # Set x-axis to log scale
    plt.legend()
    plt.grid(True)
    plt.show()




'''real'''



def custom_model1(x, t_D, t_f,a):
    return ((1 / (1 + x / t_D)) * (1 / (1 + x / (4 * t_D))**(1/2)) * np.exp(-(x / t_f)**2 / (1 + x / t_D))/a)

def fcs1(k):
    # File path to your Excel file
    file_path = '/workspaces/CP1-24-final/zbpetersbuf/biodata/FCS_hundert.xlsx'

    df = pd.read_excel(file_path, skiprows=1)

    x = df['Time']  # Time column (x-values)
    y = df['Count Rate Channel 1 [kCounts/s]']  # Count Rate Channel 1 [kCounts/s] (y-values)

    correlation = np.correlate(y, y, mode='full')  # Auto-correlation of y
    #lag = np.arange(-len(x) + 1, len(x))/1000

    lag = np.arange(-len(x) + 1, len(x))
    

    adv_correlation = correlation.mean()  # Maximum correlation for normalization
    correlation = correlation / adv_correlation

    mask = lag > k  # Mask for lag > 1
    lag_filtered = lag[mask]  # Filtered lag values
    correlation_filtered = correlation[mask]  # Filtered correlation values

    popt, pcov = curve_fit(custom_model1, lag_filtered, correlation_filtered, p0=[1000, 1000, 2])  # Initial guesses
    t_D_fit, t_f_fit, a_fit = popt  # Unpack fitted parameters
    print(f"Fitted t_D: {t_D_fit}")
    print(f"Fitted t_f: {t_f_fit}")
    print(f"Fitted a: {a_fit}")

    # Compute the fitted correlation using the fitted parameters
    fitted_correlation = custom_model1(lag_filtered, t_D_fit, t_f_fit, a_fit)

    # Plot the correlation and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(lag_filtered, correlation_filtered, 'b.', label='Auto-correlation Data')  # Plot original data points
    plt.plot(lag_filtered, fitted_correlation, 'r-', label='Fitted Curve')  # Plot fitted curve
    plt.title('Auto-correlation of Count Rate vs Time and Fitted Model')
    plt.xlabel('Lag (Time Shift)')
    plt.ylabel('Correlation')
    plt.xscale('log')  # Set x-axis to log scale
    #plt.ylim(-0.1, 100)  # Set y-axis limits for better visibility
    plt.legend()
    plt.grid(True)
    plt.show()



def exp_decay(x, a, tau, b):
    return a * np.exp(-x / tau) + b


def load_decay_set(set1, set2, k, k2):
    file_path = '/workspaces/CP1-24-final/zbpetersbuf/biodata/FlipperFLIM.xlsx'
    df = pd.read_excel(file_path, skiprows=1)

    def process_set(set_index, lower_bound, upper_bound):
        time_col = 2 * set_index
        count_col = 2 * set_index + 1
        x = df.iloc[:, time_col]
        y = df.iloc[:, count_col]
        # Apply mask for x values between lower_bound and upper_bound
        mask = (x > lower_bound) & (x < upper_bound)
        return x[mask], y[mask]

    # Get data for both sets with the specified bounds
    x1, y1 = process_set(set1, k, k2)
    x2, y2 = process_set(set2, k, k2)

    # Fit the exponential decay function and get the parameters and covariance matrix
    popt1, pcov1 = curve_fit(exp_decay, x1, y1, p0=(y1.max(), 1, y1.min()))
    popt2, pcov2 = curve_fit(exp_decay, x2, y2, p0=(y2.max(), 1, y2.min()))

    # Compute the standard deviation (error) for the fitted tau values
    perr1 = np.sqrt(np.diag(pcov1))  # Standard deviation of parameters for set1
    perr2 = np.sqrt(np.diag(pcov2))  # Standard deviation of parameters for set2

    # Half-life error bars
    tau_error1 = perr1[1]  # Error for τ from the first fit
    tau_error2 = perr2[1]  # Error for τ from the second fit

    # Generate the fit lines for plotting
    x_fit1 = np.linspace(x1.min(), x1.max(), 500)
    y_fit1 = exp_decay(x_fit1, *popt1)

    x_fit2 = np.linspace(x2.min(), x2.max(), 500)
    y_fit2 = exp_decay(x_fit2, *popt2)

    # Plot the data and fitted curves
    plt.figure(figsize=(10, 6))

    plt.plot(x1, y1, 'b.', label=f'Set {set1} Data')
    plt.plot(x_fit1, y_fit1, 'b-', label=f'Set {set1} Fit (τ ≈ {popt1[1]:.2f} ± {tau_error1:.2f} ns)')

    plt.plot(x2, y2, 'r.', label=f'Set {set2} Data')
    plt.plot(x_fit2, y_fit2, 'r-', label=f'Set {set2} Fit (τ ≈ {popt2[1]:.2f} ± {tau_error2:.2f} ns)')

    plt.xlabel("Time (ns)")
    plt.ylabel("Counts")
    plt.title(f"Fluorescence Decay Fit (Only Time > {k} ns)")
    plt.legend()
    plt.grid(True)
    plt.show()




def fcs2(k1,k2):
    file_path = '/workspaces/CP1-24-final/zbpetersbuf/biodata/FCS_hundert.xlsx'

    df = pd.read_excel(file_path, skiprows=1)

    lag = df['Time']  # Time column (x-values)
    y = df['Count Rate Channel 1 [kCounts/s]']  # Count Rate Channel 1 [kCounts/s] (y-values)
    
    #lag = np.array([i * 10**(-3) for i in x])  # Convert lag to a NumPy array
    # Auto-correlation of y
    correlation = np.correlate(y, y, mode='full')  # Auto-correlation of y

    # Normalize the correlation
    adv_correlation = correlation.mean()  # Maximum correlation for normalization
    correlation = (correlation - adv_correlation)/adv_correlation

    #correlation = np.abs(correlation)/adv_correlation

    # Get the second half of the correlation, which corresponds to positive lags
    mid_point = len(correlation) // 2
    correlation_positive = correlation[mid_point:]

    mask = (lag > k1) & (lag < k2)
    lag = lag[mask]  # Filtered lag values
    correlation_positive = correlation_positive[mask]  # Filtered correlation values

    p0 = [3, 2, 1]

    # Set bounds for each parameter [lower, upper]
    lower_bounds = [0, 0, 0]  # Lower bounds for [t_D, t_f, a]
    upper_bounds = [100, 1000, 2]  # Upper bounds for [t_D, t_f, a]

    # Fit the custom model with bounds
    popt, pcov = curve_fit(custom_model, lag, correlation_positive, p0=p0, bounds=(lower_bounds, upper_bounds))

    t_D_fit, t_f_fit, a_fit = popt  # Unpack fitted parameters
    print(f"Fitted t_D: {t_D_fit}")
    print(f"Fitted t_f: {t_f_fit}")
    print(f"Fitted a: {a_fit}")

    # Compute the fitted correlation using the fitted parameters
    fitted_correlation = custom_model(lag, t_D_fit, t_f_fit, a_fit)

    plt.figure(figsize=(10, 6))
    plt.plot(lag, correlation_positive, 'b.', label='Auto-correlation Data')  # Plot original data points
    plt.plot(lag, fitted_correlation, 'r-', label='Fitted Curve')  # Plot fitted curve
    plt.title('Auto-correlation of Count Rate vs Time and Fitted Model')
    plt.xlabel('Lag (Time Shift)')
    plt.ylabel('Correlation')
    plt.xscale('log')  # Set x-axis to log scale
    plt.legend()
    plt.grid(True)
    plt.show()




def fcs3(k1,k2):
    file_path = '/workspaces/CP1-24-final/zbpetersbuf/biodata/FCS_hundert.xlsx'

    df = pd.read_excel(file_path, skiprows=1)

    lag = df['Time']  # Time column (x-values)
    y = df['Count Rate Channel 1 [kCounts/s]']  # Count Rate Channel 1 [kCounts/s] (y-values)
    
    adv_correlation = y.mean()  # Maximum correlation for normalization
    y = y - adv_correlation
                   
    correlation = np.correlate(y, y, mode='full')  # Auto-correlation of y

    # Normalize the correlation
    adv_correlation = correlation.mean()  # Maximum correlation for normalization
    #correlation = correlation/adv_correlation

    #adv_correlation = correlation.mean()

    correlation = np.abs(correlation/adv_correlation)
    adv_correlation = correlation.max()
    correlation = np.abs(correlation/adv_correlation)

    #correlation = np.abs(correlation)/adv_correlation

    # Get the second half of the correlation, which corresponds to positive lags
    mid_point = len(correlation) // 2
    correlation_positive = correlation[mid_point:]

    mask = (lag > k1) & (lag < k2)
    lag = lag[mask]  # Filtered lag values
    correlation_positive = correlation_positive[mask]  # Filtered correlation values

    p0 = [0.03, 0.002, 0.5]

    # Set bounds for each parameter [lower, upper]
    lower_bounds = [0, 0, 0]  # Lower bounds for [t_D, t_f, a]
    upper_bounds = [1, 10000000, 5]  # Upper bounds for [t_D, t_f, a]

    # Fit the custom model with bounds
    popt, pcov = curve_fit(custom_model, lag, correlation_positive, p0=p0, bounds=(lower_bounds, upper_bounds))

    t_D_fit, t_f_fit, a_fit = popt  # Unpack fitted parameters
    print(f"Fitted t_D: {t_D_fit}")
    print(f"Fitted t_f: {t_f_fit}")
    print(f"Fitted a: {a_fit}")

    # Compute the fitted correlation using the fitted parameters
    fitted_correlation = custom_model(lag, t_D_fit, t_f_fit, a_fit)

    plt.figure(figsize=(10, 6))
    plt.plot(lag, correlation_positive, 'b.', label='Auto-correlation Data')  # Plot original data points
    plt.plot(lag, fitted_correlation, 'r-', label='Fitted Curve')  # Plot fitted curve
    plt.title('Auto-correlation of Count Rate vs Time and Fitted Model')
    plt.xlabel('Lag (Time Shift)')
    plt.ylabel('Correlation')
    plt.xscale('log')  # Set x-axis to log scale
    plt.legend()
    plt.grid(True)
    plt.show()
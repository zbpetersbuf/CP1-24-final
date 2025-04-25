def gaussian(x, A, mu, sigma, c):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + c

def fit_gaussian(coords, x_min, x_max):
    filtered_coords = [(x, y) for x, y in coords if x_min <= x <= x_max]
    
    if not filtered_coords:
        return None  # No data in the range

    x_data, y_data = zip(*filtered_coords)

    A_guess = max(y_data) - min(y_data)  # The amplitude is the difference between max and min y
    mu_guess = np.mean(x_data)           # The mean is the average of x values
    sigma_guess = (max(x_data) - min(x_data)) / 4  # Rough estimate for the width
    c_guess = min(y_data)                # The baseline 'c' is the minimum y value
    popt, _ = curve_fit(gaussian, x_data, y_data, p0=[A_guess, mu_guess, sigma_guess, c_guess])
    mu_fit = popt[1]
    return mu_fit

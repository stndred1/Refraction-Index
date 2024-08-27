import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Function to retrieve data points (experimental data)
def load_experiment_data():
    x_positions = np.array([
        0.87, 1.00, 1.10, 1.26, 1.36, 1.52, 1.63, 1.75, 1.83, 1.94, 2.05, 2.18, 2.26, 
        2.43, 2.50, 2.60, 2.72, 2.88, 2.99, 3.10, 3.23, 3.38, 3.48, 3.62, 3.71, 3.88, 
        3.94, 4.08, 4.18, 4.35, 4.48, 4.67, 4.80, 4.93, 5.10, 5.19, 5.34, 5.49, 5.60, 
        5.75, 5.95, 5.96, 6.10, 6.24, 6.40, 6.56, 6.72, 6.97, 7.12, 7.30, 7.44, 7.57, 
        7.71, 7.83, 7.99, 8.18, 8.37, 8.47, 8.69, 8.88, 9.01, 9.11, 9.22, 9.41, 9.55, 
        9.66, 9.79, 9.90, 10.08, 10.22, 10.36, 10.49, 10.49
    ])
    y_positions = np.array([
        -1.35, -1.35, -1.36, -1.34, -1.38, -1.38, -1.39, -1.40, -1.42, -1.41, -1.42, 
        -1.43, -1.44, -1.46, -1.46, -1.47, -1.50, -1.55, -1.56, -1.58, -1.61, -1.63, 
        -1.66, -1.70, -1.71, -1.77, -1.76, -1.82, -1.84, -1.86, -1.89, -1.93, -1.96, 
        -2.01, -2.04, -2.03, -2.09, -2.14, -2.17, -2.18, -2.19, -2.24, -2.24, -2.34, 
        -2.37, -2.44, -2.48, -2.56, -2.62, -2.66, -2.69, -2.77, -2.79, -2.84, -2.89, 
        -2.92, -2.98, -3.03, -3.11, -3.19, -3.22, -3.28, -3.33, -3.39, -3.42, -3.49, 
        -3.58, -3.59, -3.65, -3.73, -3.80, -3.85, -3.91
    ])
    return x_positions, y_positions

# Data from the experimental setup
x_data, y_data = load_experiment_data()

# Model for the catenary curve considering refraction index n(y) = 1 + αy
def refraction_model(params, x_vals):
    c1, c2, alpha = params
    return (c1 / alpha) * np.cosh((alpha / c1) * (x_vals - c2)) - 1 / alpha

# Function to calculate residuals between model predictions and actual data
def calculate_residuals(params, x_vals, y_vals):
    return refraction_model(params, x_vals) - y_vals

# Initial guesses for the parameters
initial_guesses = [1.0, 5.0, 1.0]  # Starting values for c1, c2, and α

# Optimize the model parameters using least squares fitting
fitting_result = least_squares(calculate_residuals, initial_guesses, args=(x_data, y_data))

# Extract optimized parameter values
c1_fitted, c2_fitted, alpha_fitted = fitting_result.x

# Compute the refractive index n(y) based on the optimized α and y-data points
n_y_values = 1 + alpha_fitted * y_data

# Calculate and display the average refractive index
average_n_y = np.mean(n_y_values)
print(f"Optimized c1 = {c1_fitted:.4f}")
print(f"Optimized c2 = {c2_fitted:.4f}")
print(f"Optimized α (alpha) = {alpha_fitted:.4f}")
print(f"Average refractive index n(y) = {average_n_y:.4f}")

# Generate x-values for smooth curve plotting
x_plot_vals = np.linspace(min(x_data) - 1, max(x_data) + 1, 400)

# Calculate the corresponding y-values for the plot based on the fitted parameters
y_plot_vals = refraction_model([c1_fitted, c2_fitted, alpha_fitted], x_plot_vals)

# Plotting the curve and experimental data points
plt.figure(figsize=(10, 6))

# Plot the fitted refraction model
plt.plot(x_plot_vals, y_plot_vals, color='darkblue', label=f'Fitted Refraction Model\nn(y) = 1 + {alpha_fitted:.4f}y')

# Plot the original data points
plt.scatter(x_data, y_data, color='crimson', label='Experimental Data Points', marker='o', zorder=5)

# Customizing plot appearance
plt.title(f'Fitted Refraction Index Model and Experimental Data\nAverage n(y) = {average_n_y:.4f}', fontsize=14)
plt.xlabel('x-position (cm)')
plt.ylabel('y-position (cm)')
plt.legend(loc='best')
plt.grid(True, linestyle='--', linewidth=0.5)

# Display the plot
plt.show()

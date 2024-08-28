import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = .25  # Example value for alpha
c1 = .25     # Example value for c1
c2 = 0     # Example value for c2

# Define the function
def f(x, alpha, c1, c2):
    return (alpha / c1**2) * (x - c2)**2 + c1**2 / alpha - 1 / alpha

# Generate x values
x = np.linspace(-2, 2, 400)

# Calculate y values
y = f(x, alpha, c1, c2)

# Plotting the function
plt.figure(figsize=(8, 6))
plt.grid()
plt.plot(x, y, label=r'$y = \frac{\alpha}{c_1^2}(x - c_2)^2 + \frac{c_1^2}{\alpha} - \frac{1}{\alpha}$')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('[n(y)]^2=1+α Refractive Index Model x(m) vs y(m)')
plt.legend()


plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Create a range of x-values (standardized rewards)
x_vals = np.linspace(-3, 3, 300)
y_vals = np.tanh(x_vals)

# Identify special points
x_mid, y_mid = 0, np.tanh(0)
x_left, y_left = -3, np.tanh(-3)
x_right, y_right = 3, np.tanh(3)

# Set up the plot
plt.figure(figsize=(8, 6))
plt.axhline(0, color='black', linestyle='--')
plt.axvline(0, color='black', linestyle='--')

# Plot tanh curve
plt.plot(x_vals, y_vals, color='orange', label=r'$y=\tanh(x)$')

# Mark the midpoint (0,0) with a red X
plt.scatter(x_mid, y_mid, color='red', marker='x', s=100, label='Midpoint (0, 0)')

# Mark the “endpoints” at x = ±3 with blue X’s
plt.scatter([x_left, x_right], [y_left, y_right],
            color='blue', marker='x', s=100,
            label=f'Scaled Endpoints (-3, {y_left:.2f}), (3, {y_right:.2f})')

plt.title("Tanh Scaling Function for GPRO Reward Adjustment")
plt.xlabel("Standardized Reward (z-score)")
plt.ylabel(r"Scaled Reward $\tanh(x)$")
plt.legend()
plt.grid(True)
plt.show()
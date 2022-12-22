import numpy as np
import matplotlib.pyplot as plt
from spectral_derivative_class import spectral_derivative

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
x_grid, y_grid = np.meshgrid(x, y, sparse=True)
seto4ka = [x_grid, y_grid]
z_grid = np.sin(x_grid + y_grid)
z_div = spectral_derivative(z_grid, seto4ka).spectral_derivative_nd(13, 0.7)
div_z = z_div[0] + z_div[1]
plt.interactive(True)
h = plt.contourf(x, y, div_z)
plt.axis('scaled')
plt.colorbar()
plt.show(block=True)

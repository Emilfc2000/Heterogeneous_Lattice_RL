import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Outcomment the section of code corresponding to data desired to plot. 4 options:
# Npr 1, Npr2, E_c and Controlled Ux


# # Negative poisson ratio Attempt 1
# # Original data
# first30 = [-2.01, -3.38, -5, -3.48, -4.65, -4.37, -2.17, -3.42, -2.59, -3.67,
#            -2.86, -1.85, -5, -3.49, -2.96, -1.68, -1.57, -2.19, -1.84, -1.23,
#            -1.67, -2.72, -1.78, -1.56, -1.62, -1.41, -2.11, -3.27, -1.31, -1.40]
# # File list
# file_list = ['npr_30_to_60.csv','npr_60_to_100.csv','npr_100_to_200.csv','npr_200_to_300.csv','npr_300_to_350.csv']
# # Load all arrays
# arrays = [np.asarray(first30)] + [np.loadtxt(fname, delimiter=",") for fname in file_list]
# long_array = np.concatenate(arrays)
# fail_value = -5.0
# plotname = 'first Poissons ratio attempt'


# # Negative poisson ratio Attempt 2
# long_array = np.loadtxt('npr_600.csv', delimiter=',')
# fail_value = -5.0
# plotname = 'second Poissons ratio attempt'


# # E_c optimization:
# file_list = ['Ec_0_to_100_v2.csv','Ec_100_to_200.csv','Ec_200_to_350.csv','Ec_350_to_500.csv']
# arrays = [np.loadtxt(fname, delimiter=",") for fname in file_list]
# long_array = np.concatenate(arrays)
# fail_value = 0.0
# plotname = f'$E_c$'


# Controlled Deformation optimization:
file_list = ['Control_0_to_150.csv','Control_150_to_200.csv','Control_200_to_350.csv','Control_350_to_500.csv']
arrays = [np.loadtxt(fname, delimiter=",") for fname in file_list]
long_array = np.concatenate(arrays)
fail_value = 0.0
plotname = 'Controlled Deformation'

# Plot 1: just the data
x = np.arange(len(long_array))
y = long_array
plt.figure()
plt.plot(x,y)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward over Episodes')
plt.show()

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)
regression_line = slope * x + intercept
r_squared = r_value**2

failed_sims = np.sum(long_array == fail_value)
# Count failures, print slope of linear regreesion
print('Number of failed simulations:', failed_sims)
print(f'The slope of the regressions is {slope:.4g}')

# Plot 2: data as a line with red markers on y == bad score and regression line
plt.figure()

# Plot the full data line in blue
plt.plot(x, y, label='Data', color='blue')

# Overlay red markers at y == bad score positions
mask_zero = y == fail_value
plt.plot(x[mask_zero], y[mask_zero], linestyle='None', marker='x', color='red', label=f'Failed Simulations (total: {failed_sims})')

# Plot regression line
plt.plot(x, regression_line, label=f'Linear Regression, Slope = {slope:.2e}', color='black')

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title(f'Rewards for {plotname}')
plt.legend()
plt.show()

# Program 5: Simple Linear Regression (SLR)

import matplotlib.pyplot as plt
from scipy import stats

# 1. Input data (independent variable x and dependent variable y)
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [4, 7, 1, 2, 4, 5, 7, 8, 5]

# 2. Perform linear regression using scipy
slope, intercept, r, p, std_err = stats.linregress(x, y)

# 3. Define regression function
def slr(x):
    return slope * x + intercept

# 4. Apply regression function to all x values
model = list(map(slr, x))

# 5. Plot regression line and scatter points
plt.scatter(x, y, color='blue', label="Data points")
plt.plot(x, model, color='red', label="Regression line")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# 6. Print model details
print("Slope:", slope)
print("Intercept:", intercept)
print("R-squared:", r**2)

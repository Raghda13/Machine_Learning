import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color='blue', label='Actual points')        # blue = real data
plt.plot(X, y_pred, color='red', linewidth=2, label='Line')     # red = prediction line
plt.xlabel('X values')
plt.ylabel('y values')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()



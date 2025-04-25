
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Reading the data
df = pd.read_csv(r"C:\FuelConsumptionCo2.csv")

# Selecting relevant columns
cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]

# Visualizing the original data
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("CO2 Emissions")
plt.title("Engine Size vs CO2 Emissions (All Data)")
plt.show()

# Splitting into training and testing sets
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Training the model
x_train = train[['ENGINESIZE']]
y_train = train[['CO2EMISSIONS']]
regr = LinearRegression()
regr.fit(x_train, y_train)

# Coefficients
print("Coefficient:", regr.coef_)
print("Intercept:", regr.intercept_)

# Plotting regression line on training data
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_train, regr.predict(x_train), color='red')
plt.xlabel("Engine size")
plt.ylabel("CO2 Emissions")
plt.title("Regression Line on Training Data")
plt.show()

# Predicting on test data
x_test = test[['ENGINESIZE']]
y_test = test[['CO2EMISSIONS']]
y_pred = regr.predict(x_test)

# Visualizing predictions vs actual test values
plt.scatter(x_test, y_test, color='green', label='Actual')
plt.scatter(x_test, y_pred, color='red', label='Predicted')
plt.xlabel("Engine size")
plt.ylabel("CO2 Emissions")
plt.title("Actual vs Predicted CO2 Emissions (Test Set)")
plt.legend()
plt.show()

# Evaluating the model
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y_test) ** 2))
print("Variance score (R²): %.2f" % regr.score(x_test, y_test))

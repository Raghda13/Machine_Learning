import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Reading the data in
df = pd.read_csv(r"C:\FuelConsumptionCo2.csv")
df.head()
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

# Plotting Emission values with respect to Engine size
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# Creating train and test dataset
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Polynomial Regression (Degree 2)
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
print('Degree 2 coefficients:', clf.coef_)
print('Intercept:', clf.intercept_)

# Plotting Polynomial Regression (Degree 2)
plt.figure(figsize=(10, 6))
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1] * XX + clf.coef_[0][2] * np.power(XX, 2)
plt.plot(XX, yy, '-r', label='Polynomial Degree 2')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# Evaluation for Polynomial Degree 2
test_x_poly = poly.transform(test_x)
test_y_ = clf.predict(test_x_poly)
print("Mean absolute error (Degree 2): %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE) (Degree 2): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score (Degree 2): %.2f" % r2_score(test_y, test_y_))

# Polynomial Regression (Degree 3)
poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3, train_y)

# The coefficients for Degree 3
print('Degree 3 coefficients:', clf3.coef_)
print('Intercept (Degree 3):', clf3.intercept_)

# Plotting Polynomial Regression (Degree 3)
plt.figure(figsize=(10, 6))
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf3.intercept_[0] + clf3.coef_[0][1] * XX + clf3.coef_[0][2] * np.power(XX, 2) + clf3.coef_[0][3] * np.power(XX, 3)
plt.plot(XX, yy, '-g', label='Polynomial Degree 3')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# Evaluation for Polynomial Degree 3
test_x_poly3 = poly3.transform(test_x)
test_y3_ = clf3.predict(test_x_poly3)
print("Mean absolute error (Degree 3): %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
print("Residual sum of squares (MSE) (Degree 3): %.2f" % np.mean((test_y3_ - test_y) ** 2))
print("R2-score (Degree 3): %.2f" % r2_score(test_y, test_y3_))

# Show the plots
plt.legend()
plt.show()

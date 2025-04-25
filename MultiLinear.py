import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as Dataframe

# Reading the data
df = pd.read_csv(r"C:\FuelConsumptionCo2.csv")
df.head()

# Selecting relevant columns for analysis
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

# Visualizing the relationship between engine size and CO2 emissions8
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Relation between Engine Size and CO2 Emissions")
plt.show()  # Show the plot

# Visualizing the relationship between cylinders and CO2 emissions
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='green')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.title("Relation between Cylinders and CO2 Emissions")
plt.show()  # Show the plot

# Visualizing the relationship between fuel consumption (combined) and CO2 emissions
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='red')
plt.xlabel("Fuel Consumption (Combined)")
plt.ylabel("Emission")
plt.title("Relation between Fuel Consumption and CO2 Emissions")
plt.show()  # Show the plot

# Visualizing the relationship between fuel consumption in the city and CO2 emissions
plt.scatter(cdf.FUELCONSUMPTION_CITY, cdf.CO2EMISSIONS, color='orange')
plt.xlabel("Fuel Consumption (City)")
plt.ylabel("Emission")
plt.title("Relation between Fuel Consumption (City) and CO2 Emissions")
plt.show()  # Show the plot

# Visualizing the relationship between fuel consumption on highway and CO2 emissions
plt.scatter(cdf.FUELCONSUMPTION_HWY, cdf.CO2EMISSIONS, color='purple')
plt.xlabel("Fuel Consumption (Highway)")
plt.ylabel("Emission")
plt.title("Relation between Fuel Consumption (Highway) and CO2 Emissions")
plt.show()  # Show the plot

# Splitting data into training and test sets
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Visualizing the training data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()  # Show the plot

# Preparing the data for linear regression
from sklearn.linear_model import LinearRegression

# Selecting features and target for training
x_train = train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_train = train[['CO2EMISSIONS']]

# Fitting the model with DataFrame (not NumPy array)
regr = LinearRegression()
regr.fit(x_train, y_train)

# Now you can predict after the model is trained using DataFrame
x_test = test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_hat = regr.predict(x_test)

# Preparing the data for the second model (using other features)
x_train_2 = train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]
y_train_2 = train[['CO2EMISSIONS']]

# Fitting the second model
regr.fit(x_train_2, y_train_2)

# Printing the model coefficients
print('Coefficients:', regr.coef_)

# Making predictions with the second model
x_test_2 = test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]
y_ = regr.predict(x_test_2)
y = np.asanyarray(test[['CO2EMISSIONS']])

# Calculating the residual sum of squares and variance score
print("Residual sum of squares: %.2f" % np.mean((y_ - y) ** 2))
print('Variance score: %.2f' % regr.score(x_test_2, y))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt



## predict house prices (data/housing.csv)

data = pd.read_csv("data/housing.csv")

# quick overview of the dataset 

print(data.head()) # default lines 5 
print(data.info())  # per column the data type, to pre-process the data, to decide which model is the proper one for my data
print(data.columns)
# quick overview of the descriptive analytics of the dataset

# mean, std, min, max, 25%, 50%, 75%

print(data.describe())

# Features and target
# Features -> inputs to the model (region, year, number of rooms etc.) X matrix 
# Target -> the house price y vector [100.000, 200.000, 300.000]

X = data[['sqft_living', 'bedrooms', 'bathrooms', 'yr_built', 'yr_renovated', 
            'sqft_basement']]
y = data['price']

"""
features = [
        'bedrooms',
        'bathrooms',
        'sqft_living,
        'etc...'
]

x = data[features]
"""


# Split the dataset to Train/Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=84)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Evaluation of the model

mse = mean_squared_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Squared error: {mse}")
print(f"Root Mean Squared error: {rmse}")
print(f"Mean Absolute error: {mae}")


# Plot Actual vs Predicted

plt.figure(figsize=(10,5))

plt.plot(y_test.values[:100], label = "Actual Prices")
plt.plot(predictions[:100], label = "Predicted Prices")

plt.title("Actual vs Predicted")
plt.xlabel("House Index")
plt.ylabel("Price")

plt.legend()
plt.show()
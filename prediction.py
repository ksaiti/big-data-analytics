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

# features and target 
# features --> inputs to the model (region, year, number of rooms etc.) X matrix  
# target  --> the house price  y vector [100.000, 200.000, 300.000, etc, etc]

y = data['price']
X = data[['sqft_living','bedrooms','bathrooms', 'yr_built', 'yr_renovated', 'sqft_basement']]

# features = [
#      'bedrooms',
#     'bathrooms',
#      'sqft_living',
#       'etc.']
# X = data[features]

# if you want to use all data as features except the target, you can use function drop
# X = data.drop('price')

# split the dataset to Train/Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(predictions)

# evaluation of the model 

mse = mean_squared_error(y_test, predictions)
print(mse)

mae = mean_absolute_error(y_test, predictions)
print(mae)

rmse = root_mean_squared_error(y_test, predictions)
print(rmse) # 270.000 

# plot Actual vs Predicted

plt.figure(figsize=(10,5))

plt.plot(y_test.values[:50], label="Actual Prices") # x
plt.plot(predictions[:50], label="Predicted Prices") # y


plt.title("Actual vs Predicted")
plt.xlabel("House index")
plt.ylabel("Price")

plt.legend()
plt.show()












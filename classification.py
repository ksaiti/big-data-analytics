import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset

df = pd.read_csv('data/titanic.csv')

print(df.info())
print(df.describe())

print(df.head())

# Features and target
# y = 'Survived'
# x = 'Age', 'Sibsp', 'Parch'

y = df['Survived']
X = df[['Age','SibSp','Parch']]

X = X.fillna(X.mean()) # if we remove the missing values, we lose too much data

# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier

model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions

prediction = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, prediction)

print(accuracy)

# Confusion matrix

cm = confusion_matrix(y_test, prediction)
print(cm)

# plot the confusion matrix

plt.imshow(cm)
plt.title("Consufion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.colorbar()

plt.show()
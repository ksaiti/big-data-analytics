import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# load dataset

data = pd.read_csv('data/titanic.csv')

print(data.info())
print(data.describe())

# features and target
# y = Survived
# X = 'Age', 'SibSp', 'Parch'

y = data['Survived']
X = data[['Age', 'SibSp', 'Parch','Pclass']]

X = X.fillna(X.mean()) # remove the missing values, we are loosing data
                       #.min, .max

# split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train the classifier
model = LogisticRegression()
model.fit(X_train, y_train)

#predictions 

predictions = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

# confusion matrix

cm = confusion_matrix(y_test, predictions)

print(cm)

# plot the confusion matrix

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.colorbar()

plt.show()
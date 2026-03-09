import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv(r"C:\Users\myjyu\Downloads\Ice Cream.csv")
df.head()

print(df.info())
print(df.describe())
print(df.isnull().sum())

plt.figure(figsize=(8,5))
sns.scatterplot(x="Temperature", y="Revenue", data=df)
plt.title("Temperature vs Ice Cream Revenue")
plt.xlabel("Temperature (°C)")
plt.ylabel("Revenue")
plt.show()

X = df[['Temperature']]
y = df['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color="blue", label="Actual Revenue")
plt.plot(X_test, predictions, color="red", label="Predicted Line")
plt.xlabel("Temperature")
plt.ylabel("Revenue")
plt.title("Linear Regression Prediction")
plt.legend()
plt.show()

temperature = [[35]]
predicted_revenue = model.predict(temperature)

print("Predicted Revenue for 35°C:", predicted_revenue[0])

import pickle

with open("icecream_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully")
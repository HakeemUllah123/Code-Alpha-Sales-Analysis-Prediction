# Sales Prediction using Python

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 2: Load Dataset
df = pd.read_csv("advertising.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())


# Step 3: Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values if any
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()


# Step 4: Data Exploration
print("\nStatistical Summary:")
print(df.describe())


# Step 5: Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# Step 6: Feature Selection
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


# Step 7: Split Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 8: Train Model
model = LinearRegression()
model.fit(X_train, y_train)


# Step 9: Predict Sales
y_pred = model.predict(X_test)


# Step 10: Model Evaluation
print("\nModel Evaluation")

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Step 11: Advertising Impact Analysis
print("\nModel Coefficients (Impact on Sales):")

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")


# Step 12: Visualization of Predictions
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()


# Step 13: Example Prediction
print("\nExample Prediction")

new_data = [[150, 20, 30]]   # TV, Radio, Newspaper
prediction = model.predict(new_data)

print("Predicted Sales:", prediction[0])

df = df.drop(columns=['Unnamed: 0'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("train.csv")
print(df.head())

# Select Required Features
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Data Cleaning
print(df.isnull().sum())
df = df.dropna()

# Data Visualization (Optional but important)
sns.pairplot(df)
plt.show()

# Split Data
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Test with Your Own Input
sample = [[2000, 3, 2]]  # sqft, bedrooms, bathrooms
price = model.predict(sample)
print("Predicted Price:", price)

# Predict on test.csv
test_df = pd.read_csv("test.csv")
test_df = test_df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
test_df = test_df.dropna()
test_predictions = model.predict(test_df)
print("Predictions for test.csv:")
for i, pred in enumerate(test_predictions):
    print(f"House {i+1}: ${pred:.2f}")
# Optionally save to CSV
output_df = pd.DataFrame({'Id': test_df.index + 1, 'SalePrice': test_predictions})  # Assuming Id starts from 1
output_df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
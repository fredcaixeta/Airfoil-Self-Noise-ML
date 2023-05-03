#Random forest regression: an ensemble of decision trees that can capture complex interactions between the input features

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# read data into pandas DataFrame
df = pd.read_csv('airfoil_self_noise.dat', sep='\s+', header=None,
                 names=['Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity',
                        'Suction side displacement thickness', 'Scaled sound pressure level'])

# Define features and target
X = df.iloc[:,:-1] # Features
y = df.iloc[:,-1] # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
# Calculate R-squared value
r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)

# Generate feature importances plot
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), features[indices], rotation=10)
plt.show()

# Generate scatter plot of actual vs predicted values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Scaled Sound Pressure Level")
plt.ylabel("Predicted Scaled Sound Pressure Level")
plt.title("Random Forest Regressor Model: Actual vs. Predicted Values")
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# read data into pandas DataFrame
df = pd.read_csv('airfoil_self_noise.dat', sep='\s+', header=None,
                 names=['Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity',
                        'Suction side displacement thickness', 'Scaled sound pressure level'])

# display the first few rows
#print(df.head())

X = df.drop('Scaled sound pressure level', axis=1)
y = df['Scaled sound pressure level']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression object
lr = LinearRegression()

# Fit the model to the training data
lr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the mean squared error
print('Mean Squared Error:', mse)

# Create a scatter plot of predicted versus actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Scaled sound pressure level')
plt.ylabel('Predicted Scaled sound pressure level')
plt.title('Graph - Predicted versus Actual Scaled sound pressure level')
plt.show()



# Plot predicted vs. actual values
plt.scatter(X_test['Angle of attack'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Angle of attack'], y_pred, color='red', label='Predicted')
plt.xlabel('Angle of Attack')
plt.ylabel('Scaled Sound Pressure Level')
plt.title('Predicted vs. Actual Values')
plt.legend()
plt.show()
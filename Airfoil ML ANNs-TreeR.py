#Using ExtraTreesRegressor to rank the feature importance and select the best features

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

# read data into pandas DataFrame
df = pd.read_csv('airfoil_self_noise.dat', sep='\s+', header=None,
                 names=['Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity',
                        'Suction side displacement thickness', 'Scaled sound pressure level'])

X = df.drop('Scaled sound pressure level', axis=1)
y = df['Scaled sound pressure level']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X, y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
best_features = feat_importances.nlargest(3).index.tolist()

# Select best features and split data into training and test sets again
X = df[best_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the neural network model
model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(len(best_features),)))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error'])
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate the model on the test set and make predictions
test_scores = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

# Plot the actual vs predicted values using a scatter plot
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Scaled sound pressure level')
plt.ylabel('Predicted Scaled sound pressure level')
plt.title('Actual vs Predicted values')
plt.show()

# Plot the loss vs epoch for both the training and validation sets
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

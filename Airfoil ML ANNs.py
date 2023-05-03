import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# read data into pandas DataFrame
df = pd.read_csv('airfoil_self_noise.dat', sep='\s+', header=None,
                 names=['Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity',
                        'Suction side displacement thickness', 'Scaled sound pressure level'])

X = df[['Angle of attack', 'Suction side displacement thickness']]
y = df['Scaled sound pressure level']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(2,)))
model.add(layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error'])
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)
test_scores = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

# Plot predicted values versus actual values
plt.scatter(X_test['Angle of attack'], y_test, label='Actual')
plt.scatter(X_test['Angle of attack'], y_pred, label='Predicted')
plt.xlabel('Angle of attack')
plt.ylabel('Scaled sound pressure level')
plt.legend()
plt.show()

# Plot training and validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plot training and validation mean absolute error values
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Create scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()
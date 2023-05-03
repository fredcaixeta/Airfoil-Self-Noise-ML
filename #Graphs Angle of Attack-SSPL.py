#Graphs Angle of Attack-SSPL

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('airfoil_self_noise.dat', sep='\s+', header=None,
                 names=['Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity',
                        'Suction side displacement thickness', 'Scaled sound pressure level'])


# Create a scatter plot
plt.scatter(df['Angle of attack'], df['Scaled sound pressure level'], alpha=0.5)

# Add labels and title
plt.xlabel('Angle of Attack')
plt.ylabel('Scaled sound pressure level')
plt.title('Correlation between Angle of Attack and Scaled sound pressure level')

# Show the plot
plt.show()
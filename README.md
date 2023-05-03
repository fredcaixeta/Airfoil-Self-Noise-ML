# Airfoil-Self-Noise-ML
ML (linear regression) approach to do a correlation between some features of an airfoil and its self noise with NASA data set, obtained from a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic wind tunnel.

Machine learning (ML) project to predict the Scaled Sound Pressure Level of an Airfoil using the features - frequency, angle of attack, chord length, free-stream velocity and suction side displacement thickness.

First, I started by loading the dataset, which was in a .dat format, and transformed it into a .csv format using Python. Then, I split the data into training and testing sets with a ratio of 80:20 using the train_test_split function from the sklearn library.

To predict the scaled sound pressure level, I implemented four different machine learning models. I started with Linear Regression, which gave me a Mean Squared Error (MSE) of 22.13 and an R-squared value of 0.56. Although the model explained 56% of the variability in the scaled sound pressure level, the MSE was relatively high, indicating that the model's predictions were not very accurate.

Next, I tried ANNs with a simple architecture of two hidden layers, each with ten neurons. The model was trained for 100 epochs, and we used the Adam optimizer and mean squared error as a loss function. The ANN model performed better than Linear Regression, achieving an MSE of 66.49.

After that, I implemented the own model with ANNs-ExtraTreesRegressor, which uses a decision tree-based algorithm. However, the model did not perform as well as I had hoped, achieving an MSE of 5681.43.

Finally, I implemented the Random Forest Regressor, which gave me the best results. It achieved an MSE of 3.28 and an R-squared value of 0.93, indicating that the model could explain 93% of the variability in the scaled sound pressure level.

I also visualized each feature importance for the model. This analysis showed that the most critical features in predicting noise were the suction side displacement thickness, frequency, and free-stream velocity.

Finally, I saved the trained Random Forest model using pickle, which allows me to use input data with the pickle file from Python to predict the noise with the parameters given.

Through this machine learning project, I was able to apply various techniques to predict the scaled sound pressure level of an airfoil. Different models gave me different results, with the Random Forest model giving me the most accurate predictions. This project highlights the importance of finding a good model in machine learning and how it can help in accurately predicting outcomes for various problems.

Machine learning is becoming increasingly important in today's world as it can provide valuable insights and predictions for businesses, organizations, and governments, and can lead to improved decision-making. I look forward to applying these techniques to other problems in the future and helping to drive innovation through machine learning. My next following project is related to decision-making in logistics for reducing transportation costs.

Please, feel free to comment!

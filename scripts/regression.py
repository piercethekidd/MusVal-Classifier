import math as mt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# The predict function sets up a linear regression model for an audio feature and returns a data frame
# Feature options: acousticness, danceability, energy, instrumentalness, loudness, tempo

def train(feature, x_train, x_test, y_train, y_test):

	# Vectorize lyrics for computation
	cv = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
	
	x_train_vc = cv.fit_transform(x_train)

	# Fit x and y to Linear Regression model
	model = LinearRegression()
	model.fit(x_train_vc, y_train[feature])

	# Vectorize features of test data
	x_test_vc = cv.transform(x_test)

	# Predict
	pred = model.predict(x_test_vc)
	actual = np.array(y_test[feature])

	# Mean squared error
	mse = mean_squared_error(pred, actual)
	print(feature + ' error: ' + str(mt.sqrt(mse)))
		
	return model
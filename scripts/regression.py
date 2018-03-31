import math as mt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# The predict function sets up a linear regression model for an audio feature and returns a data frame
# Feature options: acousticness, danceability, energy, instrumentalness, loudness, tempo

def predict(feature):

	lyric_dataset = pd.read_csv('./resources/lyric_dataset.csv', sep=',')
	feature_dataset = pd.read_csv('./resources/feature_dataset.csv', sep=',')

	print('Lyrics and features list opened.')

	# Sort by song_id
	lyric_dataset = lyric_dataset.sort_values(by='song')
	feature_dataset = feature_dataset.sort_values(by='song')

	# Separate features and labels from dataset
	data_x = lyric_dataset['lyrics']
	data_y = feature_dataset[feature]

	# Vectorize lyrics for computation
	cv = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
	
	# Split data; 80% for training and 20% for testingx_train_vc = cv.fit_transform(x_train)
	x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

	x_train_vc = cv.fit_transform(x_train)

	# Fit x and y to Linear Regression model
	model = LinearRegression()
	model.fit(x_train_vc, y_train)

	# Vectorize features of test data
	x_test_vc = cv.transform(x_test)

	# Predict
	pred = model.predict(x_test_vc)
	actual = np.array(y_test)

	# Mean squared error
	mse = mean_squared_error(pred, actual)
	print(feature + ' error: ' + str(mt.sqrt(mse)))
	
	df = pd.DataFrame(columns=[feature])
	for i in range(len(pred)):
		df.loc[i] = pred[i]
		
	return df
import math as mt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# The predict function sets up a linear regression model for an audio feature and returns a data frame
# Feature options: acousticness, danceability, energy, instrumentalness, loudness, tempo

def train(feature, x_train, x_test, y_train, y_test):

	# Vectorize lyrics for computation
	cv = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, analyzer=Stem())
	
	x_train_vc = cv.fit_transform(x_train)

	# Fit x and y to Linear Regression model
	model = Lasso(alpha=0.1)
	model.fit(x_train_vc, y_train[feature])

	# Vectorize features of test data
	x_test_vc = cv.transform(x_test)

	# Predict
	pred = model.predict(x_test_vc)
	actual = np.array(y_test[feature])

	# Mean squared error
	mse = mean_squared_error(pred, actual)
	print(feature + ' error: %.2f' % mt.sqrt(mse))
		
	return model

class Stem(object):

	def __init__(self):
		self.stemmer = SnowballStemmer('english')
		self.analyzer = CountVectorizer(min_df=1, stop_words='english', lowercase=True).build_analyzer()

	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in self.analyzer(doc)]

import time as tm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def run():
	lyric_dataset = pd.read_csv('./resources/lyric_dataset.csv', sep=',')
	feature_dataset = pd.read_csv('./resources/feature_dataset.csv', sep=',')

	print('Lyrics and features list opened.')

	# Sort by song_id
	lyric_dataset = lyric_dataset.sort_values(by='song')
	feature_dataset = feature_dataset.sort_values(by='song')


	# Change label to either energy or valence; apply thresholding
	feature_dataset.loc[feature_dataset['valence'] >= 0.5, "valence"] = 1
	feature_dataset.loc[feature_dataset['valence'] < 0.5, "valence"] = 0

	# Separate features and labels from dataset
	data_x = lyric_dataset['lyrics']
	data_y = feature_dataset['valence']


	# Vectorize lyrics for computation
	cv = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
	
	# Split data; 80% for training and 20% for testing
	x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.5)

	x_train_vc = cv.fit_transform(x_train)

	mnb = MultinomialNB()
	print('Multinomial Naive Bayes initialized.')

	y_train	 = y_train.astype('int')

	# Fit x and y to multinomial naive bayes model
	mnb.fit(x_train_vc, y_train)
	print('Training MNB model...')
	tm.sleep(3)

	# Vectorize features of test data
	x_test_vc = cv.transform(x_test)

	# Predict
	pred = mnb.predict(x_test_vc)
	print('Predicting test dataset...')
	tm.sleep(3)

	# Convert to array the sequelized actual labels of test dataset
	actual = np.array(y_test)

	# Accuracy
	print('MNB Accuracy: ' + str(accuracy_score(actual, pred)))

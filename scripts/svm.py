import math as mt
import time as tm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from .regression import train
from sklearn.metrics import mean_squared_error

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
	data_y = feature_dataset


	# Vectorize lyrics for computation
	cv = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
	# cv = CountVectorizer(min_df=1, stop_words='english', lowercase=True)
	
	# Split data; 80% for training and 20% for testing
	x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

	x_train_vc = cv.fit_transform(x_train)

	clf = SVC(kernel='linear', C=1)
	print('SVM initialized.')

	clf.fit(x_train_vc, y_train['valence'])
	print('Training svm model...')

	# Vectorize features of test data
	x_test_vc = cv.transform(x_test)

	# Predict
	pred = clf.predict(x_test_vc)
	print('Predicting test dataset...')

	# Convert to array the sequelized actual labels of test dataset
	actual = np.array(y_test['valence'])

	# Accuracy
	print('SVM Accuracy: ' + str(accuracy_score(actual, pred)))
	print('Performing Cross Validation')

	###### K-Fold Cross Validation
	######
	scores = cross_val_score(clf, cv.transform(data_x), data_y['valence'], cv=10)
	print("Scores: " + str(scores))
	print("Mean: " + str(scores.mean()))
	######
	######



	# Train models from given x and y values
	acousticness = train('acousticness', x_train, x_test, y_train, y_test)
	danceability = train('danceability', x_train, x_test, y_train, y_test)
	energy = train('energy', x_train, x_test, y_train, y_test)
	instrumentalness = train('instrumentalness', x_train, x_test, y_train, y_test)
	loudness = train('loudness', x_train, x_test, y_train, y_test)
	tempo = train('tempo', x_train, x_test, y_train, y_test)

	# Redo data splitting for 
	x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

	x_test_vc = cv.transform(x_train)

	predicted_acoustic_features = acousticness.predict(x_test_vc)
	predicted_danceability_features = danceability.predict(x_test_vc)
	predicted_energy_features = energy.predict(x_test_vc)
	predicted_instrumentalness_features = instrumentalness.predict(x_test_vc)
	predicted_loudness_features = loudness.predict(x_test_vc)
	predicted_tempo_features = tempo.predict(x_test_vc)

	
	# Convert numpy arrays to pandas dataframe then concatenate
	predicted_acoustic_features = convert_to_dataFrame(predicted_acoustic_features, 'acousticness')
	predicted_danceability_features = convert_to_dataFrame(predicted_danceability_features, 'danceability')
	predicted_energy_features = convert_to_dataFrame(predicted_energy_features, 'energy')
	predicted_instrumentalness_features = convert_to_dataFrame(predicted_instrumentalness_features, 
		'instrumentalness')
	predicted_loudness_features = convert_to_dataFrame(predicted_loudness_features, 'loudness')
	predicted_tempo_features = convert_to_dataFrame(predicted_tempo_features, 'tempo')
	

	predicted_features = [predicted_acoustic_features, predicted_danceability_features, predicted_energy_features, 
		predicted_instrumentalness_features, predicted_loudness_features, predicted_tempo_features]

	features = pd.concat(predicted_features, axis=1)

	clf = SVC(kernel='linear', C=1)
	clf.fit(features, y_train['valence'])
	pred = clf.predict(y_test.drop(columns=['id', 'song', 'valence']))
	actual = clf.actual = np.array(y_test['valence'])

	print('SVM Accuracy: ' + str(accuracy_score(actual, pred)))

	


# Converts a numpy array to a pandas dataframe
def convert_to_dataFrame(arr, name):
	temp = pd.DataFrame(columns=[name])
	for i in range(len(arr)):
		temp.loc[i] = arr[i]
	return temp
import math as mt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from .regression import train
from sklearn.svm import SVC
from nltk import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def run():
	lyric_dataset = pd.read_csv('./resources/lyric_dataset.csv', sep=',')
	feature_dataset = pd.read_csv('./resources/feature_dataset.csv', sep=',')

	print('Lyrics and features list opened.')

	# Sort by song_id
	lyric_dataset = lyric_dataset.sort_values(by='song')
	feature_dataset = feature_dataset.sort_values(by='song')


	# Apply thresholding; Change label to either energy or valence
	feature_dataset.loc[feature_dataset['valence'] >= 0.5, "valence"] = 1
	feature_dataset.loc[feature_dataset['valence'] < 0.5, "valence"] = 0

	# Separate features and labels from dataset
	data_x = lyric_dataset['lyrics']
	data_y = feature_dataset

	# Vectorize lyrics for computation, uncomment below to use CountVectorizer instead
	cv = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, analyzer=Stem(), ngram_range=(2,2))
	# cv = CountVectorizer(min_df=1, stop_words='english', lowercase=True, analyzer=Stem())

	# Combine text features with audio features
	x = sp.sparse.hstack((cv.fit_transform(data_x), data_y[['acousticness','danceability',
		'energy', 'instrumentalness', 'loudness','tempo']].values), format='csr')
	x = x.todense()
	y = data_y['valence']


	# Use PCA feature reduction method to reduce number of features and initialize SVM
	# Use pipelining in Scikit-learn
	estimators = [('reduce_dim', PCA()), ('clf', SVC(kernel='linear', C=1))]
	pipe = Pipeline(estimators)

	# Use K-Fold Cross Validation while using the pipeline as estimator
	scores = cross_val_score(pipe, x, y, cv=KFold(n_splits=10, shuffle=True))
	print("Scores: " + str(scores))
	print("Mean Score: %.4f" % scores.mean())

class Stem(object):

	def __init__(self):
		self.stemmer = SnowballStemmer('english')
		self.analyzer = CountVectorizer(min_df=1, stop_words='english', lowercase=True).build_analyzer()

	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in self.analyzer(doc)]

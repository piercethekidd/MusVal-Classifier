import pandas as pd
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.externals import joblib

def run():

	# Read data from csv
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
	x = data_x
	y = data_y['valence']


	"""
	Use Vectorizer for lyrics and TruncatedSVD dimensionality reduction method to reduce number of features and 
	initialize SVM then use pipelining in Scikit-learn
	"""
	estimators = [('cv', TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, analyzer=Stem(), ngram_range=(1,2))),
	('reduce_dim', TruncatedSVD()), 
	('clf', SVC(C=1, kernel='linear'))]
	pipe = Pipeline(estimators)

	# Tune parameters of estimators from pipeline; Uncomment to verify best parameters
	#tune_parameters(pipe, x, y)

	# Use K-Fold Cross Validation while using the pipeline as estimator
	scores = cross_val_score(pipe, x, y, cv=KFold(n_splits=10, shuffle=True))
	print("Scores: " + str(scores))
	print("Mean Score: %.4f" % scores.mean())

	# Model persistence; Save current model for future use
	joblib.dump(pipe, './resources/svm_lyrics.pkl')
	
# Parameter tuning using the pipeline, x, and y as inputs
def tune_parameters(pipe, x, y):
	# Dictionary of parameters to be tuned
	param_grid = dict(reduce_dim__n_components=[None, 2],
		clf__kernel=['linear', 'rbf'],
		clf__C=[0.1, 1, 10])
	# Initialize GridSearchCV
	grid_search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, cv=KFold(n_splits=10, shuffle=True))
	print("Fitting x and y values to GridSearchCV")
	# Fit data and print results
	grid_search.fit(x, y)
	print(grid_search.best_score_)
	for param_name in sorted(param_grid.keys()):
		print("%s: %r" % (param_name, grid_search.best_params_[param_name]))

# Define a stemming callable to provide stemming for Lyrics Vectorizer
class Stem(object):

	def __init__(self):
		self.stemmer = SnowballStemmer('english')
		self.analyzer = CountVectorizer(min_df=1, stop_words='english', lowercase=True).build_analyzer()

	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in self.analyzer(doc)]

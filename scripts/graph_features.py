import pandas as pd
from matplotlib import pyplot as plt


# This script saves graphs of features vs. valence
def run():

	# Read data from csv
	lyric_dataset = pd.read_csv('./res/lyric_dataset.csv', sep=',')
	feature_dataset = pd.read_csv('./res/feature_dataset.csv', sep=',')

	# Sort by song_id
	lyric_dataset = lyric_dataset.sort_values(by='song')
	feature_dataset = feature_dataset.sort_values(by='song')

	# Create features list for iteration
	features_list = ['acousticness','danceability',
		'energy', 'instrumentalness', 'loudness', 'tempo']
	valence = 'valence'


	for feature in features_list:
		plt.scatter(feature_dataset[feature], feature_dataset[valence], color='b')
		plt.xlabel(feature)
		plt.ylabel(valence)
		plt.savefig('./static/img/graph/' + feature +'.png', bbox_inches='tight')
		plt.clf()

	print("Graphs created and saved.")



from songs.resources import *
import csv

def run():
	song_resource = SongResource()
	dataset = song_resource.export()
	text_file = open("./res/dataset.csv", "w")
	text_file.write(dataset.csv)
	text_file.close()

	dataset = SongFeatureResource().export()
	text_file = open("./res/feature_dataset.csv", "w")
	text_file.write(dataset.csv)
	text_file.close()

	dataset = LyricResource().export()
	text_file = open("./res/lyric_dataset.csv", "w")
	text_file.write(dataset.csv)
	text_file.close()
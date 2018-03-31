# More information on how to use the Genius API on: https://github.com/johnwmillr/LyricsGenius

import lyricsgenius as genius
from songs.models import Song, Lyric





def run():
	
	api = genius.Genius('EvIwS8Hujru0G5Oxr8sulv9z5YLaml5gVIR9JlGGDVjomh-9LOmwSJBbyQzOqbZ3')
	fp = open('resources/index_counter.txt', 'a+')
	fp.seek(0)
	index = int(fp.readline())

	songs = Song.objects.order_by('id')[index:1090]
	
	for s in songs:
		artist = s.artist.split('Featuring')[0]
		title = s.title
		song = api.search_song(title, artist)
		l = s.lyric_set.create(lyrics=song.lyrics)
		print("Lyrics of " + s.title + " saved.")
		print(index)
		index += 1
		fp.truncate(0)
		fp.write(str(index))
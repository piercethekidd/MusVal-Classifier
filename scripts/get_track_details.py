# 12183890197 : User ID
# index default value = 0, in case of connection problem, change value of index to when it last stopped

import json
import spotipy
import spotipy.util as util
from songs.models import Song, SongFeature

def run():
	scope = 'user-library-read'
	username = '12183890197'
	token = util.prompt_for_user_token(username, scope, client_id="b226f2bec10e4127a60ec75e26562562", 
		client_secret="22d44152bc9d48c698df6108e1a5525a", redirect_uri="https://example.com/callback/")
	spotify = spotipy.Spotify(auth=token)

	
	fp = open('res/index_counter.txt', 'a+')
	fp.seek(0)
	if fp.readline() == '':
		fp.write('0\n0')
	
	fp.seek(0)
	index = int(fp.readline())
	counter = int(fp.readline())
	
	
	print("Retrieving track details of " + str(Song.objects.count()) + " songs")
	songs = Song.objects.order_by('id')[index:1100]
	
	"""
	for song in songs:
		print(counter)
		print(song.song_title)
		counter += 1
	"""

	#SongFeature.objects.all().delete()
	#print("Deleted SongFeature database.")
	for song in songs:

		query = song.artist.split("Featuring")[0] + " " +  song.title
		results = spotify.search(q=query, type='track')

		"""
		displays the json object as a readable text
		print(json.dumps(results, sort_keys=True, indent=4))
		"""

		
		if results['tracks']['items'] != []: #if query does not return any match
			#retrieve track details
			track = results['tracks']['items'][0]
			track_id = track['id']
			track_name = track['name']
			track_artist = track['artists'][0]['name']

			features = spotify.audio_features([track['id']])	
			
			print("Track " + str(counter))
			print("Track id: " + str(track_id))
			print("Track Name: %s" % track_name)
			print("Track Artist: %s" % track_artist)

			#retrieve track features
			track_acousticness = features[0]['acousticness']
			track_danceability = features[0]['danceability']
			track_energy = features[0]['energy']
			track_instrumentalness = features[0]['instrumentalness']
			track_loudness = features[0]['loudness']
			track_tempo = features[0]['tempo']
			track_valence = features[0]['valence']

			f = song.songfeature_set.create(acousticness=track_acousticness, danceability=track_danceability, 
				energy=track_energy, instrumentalness=track_instrumentalness, loudness=track_loudness, tempo=track_tempo, 
				valence=track_valence)

			print("Energy: " + str(f.energy))
			print("Valence: " + str(f.valence))


			counter = counter + 1
		else:
			print("This track is unavailable in Spotify. Track will be deleted in the database.")
			song.delete()
			index -= 1;

		index += 1;
		fp.truncate(0)
		fp.write(str(index) + '\n')
		fp.write(str(counter))

	print("Final song count: " + str(counter))
	fp.truncate(0)
	fp.write('0')
	fp.close()
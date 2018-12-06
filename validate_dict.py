"""
Make sure our chord dictionary is not missing any keys
"""
import os
import sys
from chords_to_ints import chords_to_ints

root_dir = 'data_minsimmer'

genres = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

missing_chords = []

for d in genres:
	path = os.path.join(root_dir, d)

	# get song files
	songs = [os.path.join(path, f) for f in os.listdir(path) if '.csv' in f]
	#print("Genre: {} #Songs: {}".format(d, len(songs)))

	for s in songs:
		with open(s, 'r') as fh:
			lines = [line.strip() for line in fh if line.strip() != '']
		for l in lines:
			parts = l.split(",")
			chord = parts[1].strip()
			if chord not in chords_to_ints and chord != 'N':
				if chord not in missing_chords:
					missing_chords.append(chord)

for chord in missing_chords:
	print("You are missing: {}".format(chord))

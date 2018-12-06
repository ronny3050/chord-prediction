"""
preprocess data
"""

import os
import sys
import random
import numpy
import numpy as np
from chords_to_ints import chords_to_ints

if len(sys.argv) != 3:
	print("Please pass a valid genre name and a percentage of training examples. E.g. processing_training.py Pop 80")
	sys.exit(1)

root_dir = 'final_data'

genre = sys.argv[1]
num_train = sys.argv[2]

path = os.path.join(root_dir, genre)

# this number can of course change to other window sizes
sequence_length = 4
n_vocab = 14
n_steps = 4

train_input = []
train_output = []
test_input = []
test_output = []

# get song files
songs = [os.path.join(path, f) for f in os.listdir(path) if '.csv' in f]
print(len(songs))
print("Genre: {} #Songs: {}".format(genre, len(songs)))

for idx, s in enumerate(songs):
	with open(s, 'r', encoding='windows-1252') as fh:
            chords = [line.strip().split(',')[1] for line in fh]
            if len(chords) < sequence_length+1:
                continue
            n_i = []
            n_o = []
            # create input sequences and the corresponding outputs
            for i in range(0, len(chords) - sequence_length, n_steps):
                sequence_in = chords[i:i + sequence_length]
                #print(len(sequence_in))
                sequence_out = chords[i + sequence_length]
                n_i.append([int(chords_to_ints[char.replace('"', "")]) for char in sequence_in])
                n_o.append(int(chords_to_ints[sequence_out.replace('"', "")]))
            
            # shuffle sequences
            c = list(zip(n_i, n_o))
            random.shuffle(c)
            n_i, n_o = zip(*c)
            maxim = int(0.8*len(n_i))
            train_input.extend(n_i[0:maxim])
            train_output.extend(n_o[0:maxim])
            test_input.extend(n_i[maxim:])
            test_output.extend(n_o[maxim:])

n_patterns = len(train_input)
# reshape the input into a format compatible with LSTM layers
network_input = numpy.reshape(numpy.array(train_input), (n_patterns, sequence_length, 1))
# normalize input
print(network_input)
network_input = network_input / float(n_vocab)
#network_output = np_utils.to_categorical(network_output)
one_hot_output = numpy.zeros((n_patterns, n_vocab, 1))
for index, o in enumerate(train_output):
	one_hot_output[index, o] = 1.0

print("input shape: {}".format(network_input.shape))
print("output shape: {}".format(one_hot_output.shape))

os.makedirs('data/{}/seq_len_{}'.format(genre, sequence_length))

numpy.save('data/{}/seq_len_{}/x_train.npy'.format(genre, sequence_length), network_input)
numpy.save('data/{}/seq_len_{}/y_train.npy'.format(genre, sequence_length), one_hot_output)

n_patterns = len(test_input)
# reshape the input into a format compatible with LSTM layers
network_input = numpy.reshape(numpy.array(test_input), (n_patterns, sequence_length, 1))
# normalize input
network_input = network_input / float(n_vocab)
#network_output = np_utils.to_categorical(network_output)
one_hot_output = numpy.zeros((n_patterns, n_vocab, 1))
for index, o in enumerate(test_output):
	one_hot_output[index, o] = 1.0

print("test input shape: {}".format(network_input.shape))
print("test output shape: {}".format(one_hot_output.shape))

numpy.save('data/{}/seq_len_{}/x_test.npy'.format(genre, sequence_length), network_input)
numpy.save('data/{}/seq_len_{}/y_test.npy'.format(genre, sequence_length), one_hot_output)

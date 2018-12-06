'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
from time import time
from keras.models import load_model

genre = 'pop'
seq_len = 25
num_classes = 24
max_features = 128

x_test = np.load('data/pop/seq_len_4/x_test.npy')
y_test = np.load('data/pop/seq_len_4/y_test.npy').squeeze()
model = load_model('models/pop/seq_len_4.h5'.format(genre))
y_pred = model.predict(x_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
diff = np.abs(y_true - y_pred)
diff[np.where(diff < 2)] = 0
diff = np.sum(np.mod(np.abs(diff), 13)/12)
acc = 1 - diff/x_test.shape[0]
print(acc)

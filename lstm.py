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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback

genre = 'pop'
seq_len = 4
num_classes = 14
max_features = 128
best = -np.inf
sequential = '' # can be blank or _non

class print_lr(Callback):
    def __init__(self, filepath, validation_data):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        global best

        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))

        y_pred = self.model.predict(self.x_val)
        y_pred = np.argmax(y_pred, axis = 1)
        y_true = np.argmax(self.y_val, axis = 1)
        diff = np.abs(y_true - y_pred)
        diff[np.where(diff < 2)] = 0
        diff = np.mod(diff, num_classes - 1)
        diff = np.sum(diff/np.max(diff))
        acc = round((1-diff/self.x_val.shape[0])*100, 2)

        if acc > best:
            print('acc improved from ' + str(best) + ' to ' + str(acc) + ', saving model to ' +
                    self.filepath)
            self.model.save(self.filepath)
            best = acc
        else:
            print('acc did not impove. still {}'.format(str(best)))
    
        print('LearningRate -- {}\t\t Testing Accuracy -- {}%'.format(K.eval(lr_with_decay), acc))


def train():
    epoch_size = 6000

    print('Loading data...')
    
    x_train = np.load('data/{}/seq_len_{}/x_train{}.npy'.format(genre, seq_len, sequential))
    y_train = np.load('data/{}/seq_len_{}/y_train{}.npy'.format(genre, seq_len, sequential)).squeeze()

    x_test = np.load('data/{}/seq_len_{}/x_test{}.npy'.format(genre, seq_len, sequential))
    y_test = np.load('data/{}/seq_len_{}/y_test{}.npy'.format(genre, seq_len, sequential)).squeeze()
   
    batch_size = 8224*4
    lr = 0.1

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
  
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('Build model...')
    model = Sequential()
    model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
    #model.add(LSTM(512, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    #model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(num_classes, activation='softmax'))

    decay_rate = lr / epoch_size
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr = lr),
                          metrics=['accuracy'])
    print(y_train.shape)
    print(y_test.shape)
    print('Train...')

    tensorboard = TensorBoard(log_dir="logs/{}/seq_len_{}{}".format(genre, seq_len, sequential))


    filepath="models/{}/seq_len_{}{}.h5".format(genre, seq_len, sequential)
    #checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
    plr = print_lr(filepath = filepath, validation_data=(x_test, y_test))

    callback_lists = [tensorboard, plr]

    model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epoch_size,
                      validation_data=(x_test, y_test),
                      callbacks=callback_lists)

    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', best)

train()

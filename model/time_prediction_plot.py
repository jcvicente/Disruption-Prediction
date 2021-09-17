import os
import h5py
import numpy as np
np.random.seed(2000)
from keras.models import *
from keras.layers.core import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.optimizers import *
import matplotlib.pyplot as plt

batch_size = 1
sample_size = 400

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'convlstm_trained_model.h5'
# Load model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
old_model = load_model(model_path)
print('Loaded trained model at %s ' % model_path)

model = Sequential()

model.add(Conv1D(32, 5, batch_input_shape=(batch_size, sample_size, 39)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 5))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.5, return_sequences=True))
model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.5))

model.add(Dense(300, activation='relu'))

model.add(Dense(1, activation='relu'))

old_weights=old_model.get_weights()
model.set_weights(old_weights)



fname = '../data/kb5_data.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

kb5 = dict()
kb5_t = dict()

pulse = '92213'

g = f[pulse]
kb5[pulse] = g['kb5'][:]
kb5_t[pulse] = g['kb5_t'][:]

f.close()

predictions = []
time = []

i = sample_size
while i < len(kb5[pulse]):
        x = kb5[pulse][i-sample_size:i]
        x = np.delete(x,[18,19,20,21,22,23,38,39,46,48,49,50,51,52,53,54,55],1)
        x = np.expand_dims(x, axis=0)
        predict = model.predict(x)
	predictions.append(predict[0][0])
	time.append(kb5_t[pulse][i])
	i += 32


plt.plot(time, predictions)
plt.xlabel("Time (s)")
plt.ylabel("Predicted ttd (s)")
plt.show()

	

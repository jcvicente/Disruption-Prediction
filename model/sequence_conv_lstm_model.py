import os
import h5py
import numpy as np
np.random.seed(2018)
from keras.models import *
from keras.layers.core import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.optimizers import *

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'convlstm_trained_model.h5'

batch_size = 100
sample_size = 400
minimum_time = 5000     # in miliseconds
# --------------------------------------------------------------------------

model = Sequential()

model.add(Conv1D(32, 5, batch_input_shape=(batch_size, sample_size, 56)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 5))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(64, dropout=0.1, stateful=True, recurrent_dropout=0.5, return_sequences=True))
model.add(LSTM(64, dropout=0.1, stateful=True, recurrent_dropout=0.5))

model.add(Dense(64, activation='relu'))

model.add(Dense(1))


model.summary()

opt=RMSprop()
#opt = Adam(lr=1e-4)
loss = 'mae'
model.compile(opt, loss)

# --------------------------------------------------------------------------

fname = '../data/dst_kb5.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

dst = dict()
kb5 = dict()
kb5_t = dict()

train_pulses = []
valid_pulses = []

k = 0
for pulse in f:
    k += 1
    if k % 10 == 0:
        valid_pulses.append(pulse)
    else:
        train_pulses.append(pulse)
    g = f[pulse]
    dst[pulse] = g['dst'][0]
    kb5[pulse] = g['kb5'][:]
    kb5_t[pulse] = g['kb5_t'][:]

f.close()

print 'train_pulses:', len(train_pulses)
print 'valid_pulses:', len(valid_pulses)

# -------------------------------------------------------------------------

fname = '../data/ShotList_ILW_2011-16.txt'
print 'Reading:', fname
f = open(fname, 'r')

dst_real = dict()

k=0
for pulse in f:
    k = k+1
    if (k<4) or (k>1954):
        continue

    data = pulse[25:39].split(' ')
    data.pop(1)
    data.pop(1)
    dst_real[int(data[0])]=float(data[1])

# -------------------------------------------------------------------------

def batch_generator(pulses, batch_size):
    X_batch = []
    Y_batch = []
    while True:
        pulse = np.random.choice(pulses)
        if int(pulse) not in dst_real:
            continue
        if len(kb5[pulse]) < 5000:
            continue
        i = sample_size
        while i < len(kb5[pulse]):
            x = kb5[pulse][i-sample_size:i]
            y = dst_real[int(pulse)] - kb5_t[pulse][i]
            X_batch.append(x)
            Y_batch.append(y)
            i = i + 32
            if len(X_batch) >= batch_size:
                X_batch = np.array(X_batch)
                Y_batch = np.array(Y_batch)
                yield (X_batch, Y_batch)
                X_batch = []
                Y_batch = []


model.fit_generator(batch_generator(train_pulses, batch_size),
                    steps_per_epoch=100,
                    epochs=100,
                    verbose=1,
                    validation_data=batch_generator(valid_pulses, batch_size),
                    validation_steps=10)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

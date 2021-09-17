import h5py
import numpy as np
np.random.seed(2017)
from keras.models import *
from keras.layers.core import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.optimizers import *
from keras.layers import TimeDistributed

batch_size = 256
sample_size = 500
minimum_time = 2500     # in miliseconds
# --------------------------------------------------------------------------

model = Sequential()

model.add(GRU(32, batch_input_shape=(batch_size, sample_size, 56), dropout=0.1, recurrent_dropout=0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

opt = RMSprop(lr=1e-4)
loss = 'mae'
metrics = ['mae','mape']
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

# --------------------------------------------------------------------------


def batch_generator(pulses, batch_size):
    X_batch = []
    Y_batch = []
    while True:
        pulse = np.random.choice(pulses)
        if len(kb5[pulse]) < minimum_time*2:
            continue
        i0 = max(sample_size-1, len(kb5[pulse])-minimum_time)
        i = np.random.randint(i0, len(kb5[pulse])-50)
        x = kb5[pulse][i-(sample_size):i]
        y = dst[pulse] - kb5_t[pulse][i]
        X_batch.append(x)
        Y_batch.append(y)
        if len(X_batch) >= batch_size:
            X_batch = np.array(X_batch)
            Y_batch = np.array(Y_batch)
            Y_batch = np.expand_dims(Y_batch, axis=1)
            yield (X_batch, Y_batch)
            X_batch = []
            Y_batch = []


model.fit_generator(batch_generator(train_pulses, batch_size),
                    steps_per_epoch=100,
                    epochs=1000,
                    verbose=1,
                    validation_data=batch_generator(valid_pulses, batch_size),
                    validation_steps=10)

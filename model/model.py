
import h5py
import numpy as np
np.random.seed(2017)
from keras.models import *
from keras.layers.core import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.optimizers import *

# --------------------------------------------------------------------------

model = Sequential()

model.add(Conv1D(32, 3, input_shape=(500,56)))
model.add(Activation('relu'))
model.add(Conv1D(32, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Conv1D(64, 3))
model.add(Activation('relu'))
model.add(Conv1D(64, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('relu'))

model.summary()

opt = Adam(lr=1e-3)
loss = 'mae'
model.compile(opt, loss)

# --------------------------------------------------------------------------

fname = 'data/dst_kb5.hdf'
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

sample_size = 500

def batch_generator(pulses, batch_size):
    X_batch = []
    Y_batch = []
    while True:
        pulse = np.random.choice(pulses)
        if len(kb5[pulse]) < sample_size*15:
            continue
        i0 = max(sample_size-1, len(kb5[pulse])-10*sample_size)
        i = np.random.randint(i0, len(kb5[pulse]))
        x = kb5[pulse][i-(sample_size-1):i+1]
        y = dst[pulse] - kb5_t[pulse][i]
        X_batch.append(x)
        Y_batch.append(y)
        if len(X_batch) >= batch_size:
            X_batch = np.array(X_batch)
            Y_batch = np.array(Y_batch)
            yield (X_batch, Y_batch)
            X_batch = []
            Y_batch = []

batch_size = 1024

model.fit_generator(batch_generator(train_pulses, batch_size),
                    steps_per_epoch=100,
                    epochs=1000,
                    verbose=1,
                    validation_data=batch_generator(valid_pulses, batch_size),
                    validation_steps=10)

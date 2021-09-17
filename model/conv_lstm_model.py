import os
import h5py
import numpy as np
np.random.seed(2018)
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'convlstm_trained_model.h5'

batch_size = 256
sample_size = 100
minimum_time = 1000     # in miliseconds
# --------------------------------------------------------------------------


class MyCallback(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.minimum = 100
        self.lastepoch = 0
        open('train_log_ttd.txt', 'w').close()

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('val_loss')
        if loss >= self.minimum:
            if epoch > 500 and epoch >= 2*self.lastepoch:
                self.model.stop_training = True
        else:
            self.lastepoch = epoch
            self.minimum = loss
            logfile = open('train_log_ttd.txt', 'a')
            logfile.write(str(epoch+1) + ' - loss: ' + str(logs.get('loss')) + ', val_loss: ' + str(self.minimum) + '\n')
            logfile.close()
            if epoch > 100:
                # Save model and weights
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                model_path = os.path.join(save_dir, model_name)
                model.save(model_path)


# --------------------------------------------------------------------------

model = Sequential()

model.add(Conv1D(32, 5, batch_input_shape=(batch_size, sample_size, 39)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 5))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(300, activation='relu'))

model.add(Dense(1, activation='relu'))


model.summary()

#opt=RMSprop()
opt = Adam(lr=5e-5)
loss = 'mae'
model.compile(opt, loss)

# -------------------------------------------------------------------------

fname = '../data/disruptions.txt'
print 'Reading:', fname
f = open(fname, 'r')

dst_real = dict()

for pulse in f:
    data = pulse[:-1].split(' ')
    dst_real[data[0]]=float(data[1])

f.close()
# --------------------------------------------------------------------------

fname = 'test_pulses.txt'
print 'Reading:', fname
f = open(fname, 'r')

test = dict()

for pulse in f:
    data = pulse[:-1]
    test[data]=1
f.close()

fname = 'valid_pulses.txt'
print 'Reading:', fname
f = open(fname, 'r')

valid = dict()

for pulse in f:
    data = pulse[:-1]
    valid[data]=1
f.close()

fname = '../data/kb5_data_small.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

dst = dict()
kb5 = dict()
kb5_t = dict()

train_pulses = []
valid_pulses = []

for pulse in f:
    if pulse in dst_real and pulse not in test:
        if pulse in valid:
            valid_pulses.append(pulse)
        else:
            train_pulses.append(pulse)
        g = f[pulse]
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
        if len(kb5[pulse]) < 200:
            continue
        i0 = sample_size
        i = np.random.randint(i0, len(kb5[pulse]))
        x = kb5[pulse][i-sample_size:i]
        x = np.delete(x,[18,19,20,21,22,23,38,39,46,48,49,50,51,52,53,54,55],1)
        y = dst_real[pulse] - kb5_t[pulse][i]
        X_batch.append(x)
        Y_batch.append(y)
        if len(X_batch) >= batch_size:
            X_batch = np.array(X_batch)
            Y_batch = np.array(Y_batch)
            yield (X_batch, Y_batch)
            X_batch = []
            Y_batch = []


callback = MyCallback()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=500, min_lr=5e-8)

model.fit_generator(batch_generator(train_pulses, batch_size),
                    steps_per_epoch=100,
                    epochs=9999999,
                    verbose=1,
                    validation_data=batch_generator(valid_pulses, batch_size),
                    validation_steps=10,
                    callbacks=[callback, reduce_lr])

print "Done"

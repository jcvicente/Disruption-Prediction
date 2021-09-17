import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2000)
from keras.models import *
from keras.layers.core import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.optimizers import *

batch_size = 1
sample_size = 100
test_pulse = '86956'

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name_prob = 'prob_trained_model.h5'
# Load model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name_prob)
old_model_prob = load_model(model_path)
print('Loaded trained model at %s ' % model_path)

model_name_ttd = 'convlstm_trained_model.h5'
model_path = os.path.join(save_dir, model_name_ttd)
old_model_ttd = load_model(model_path)


model = Sequential()

model.add(Conv1D(32, 5, batch_input_shape=(batch_size, sample_size, 39)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 5))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(600, dropout=0.2, recurrent_dropout=0.3, return_sequences=True))
model.add(LSTM(600, dropout=0.2, recurrent_dropout=0.3))

model.add(Dense(600, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

old_weights=old_model_prob.get_weights()
model.set_weights(old_weights)


# --------------------------------------------------------------------------

model_ttd = Sequential()

model_ttd.add(Conv1D(32, 5, batch_input_shape=(batch_size, sample_size, 39)))
model_ttd.add(Activation('relu'))
model_ttd.add(MaxPooling1D(pool_size=2))

model_ttd.add(Conv1D(64, 5))
model_ttd.add(Activation('relu'))
model_ttd.add(MaxPooling1D(pool_size=2))

model_ttd.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model_ttd.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))

model_ttd.add(Dense(300, activation='relu'))

model_ttd.add(Dense(1, activation='relu'))

old_weights=old_model_ttd.get_weights()
model_ttd.set_weights(old_weights)


# --------------------------------------------------------------------------

fname = '../data/disruptions.txt'
print 'Reading:', fname
f = open(fname, 'r')

dst = dict()

for pulse in f:
    data = pulse[:-1]
    data = data.split(' ')
    dst[data[0]]=data[1]
f.close()

# --------------------------------------------------------------------------

fname = '../data/kb5_data_small.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

kb5 = dict()
kb5_t = dict()

nondisruptive = []
disruptive = []

for pulse in f:
    if pulse == test_pulse:
        g = f[pulse]
        kb5[pulse] = g['kb5'][:]
        kb5_t[pulse] = g['kb5_t'][:]
        break

f.close()

# --------------------------------------------------------------------------

late = 0
pulse = test_pulse

ttd = []
prob = []
time = []

i = sample_size
x = []
sum_prob = 0
sum_ttd = 0
count = 0

step = 6

while i < len(kb5[pulse]):
    x = kb5[pulse][i-sample_size:i]
    x = np.delete(x,[18,19,20,21,22,23,38,39,46,48,49,50,51,52,53,54,55],1)
    x = np.expand_dims(x, axis=0)
    predict = model.predict(x)
    predict_ttd = model_ttd.predict(x)
    sum_prob += float(predict[0][0])
    sum_ttd += float(predict_ttd[0][0])
    count += 1
    if count >= step:
        mean_prob = sum_prob/step
        mean_ttd = sum_ttd/step
        prob.append(mean_prob)
        ttd.append(mean_ttd)
        time.append(kb5_t[pulse][i])
        count = 0
        sum_prob = 0
        sum_ttd = 0

    i += 1
    #print "prob: " + str(predict) + ", ttd: " + str(predict_ttd)

fig, ax1 = plt.subplots()
t = time
s1 = prob
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('time (s)')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('prob', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
s2 = ttd
ax2.plot(t, s2, 'r.')
ax2.set_ylabel('ttd', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()
















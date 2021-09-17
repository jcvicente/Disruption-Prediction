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

batch_size = 1
sample_size = 100

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

model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(300, activation='relu'))

model.add(Dense(1, activation='relu'))

old_weights=old_model.get_weights()
model.set_weights(old_weights)


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

k = 0
for pulse in f:
    k += 1
    if pulse not in dst:
        if np.random.randint(0,100) > 1 or len(nondisruptive) == 100:
            continue
        nondisruptive.append(pulse)
    else:
        if np.random.randint(0,100) > 8 or len(disruptive) == 100:
            continue
        disruptive.append(pulse)
    g = f[pulse]
    kb5[pulse] = g['kb5'][:]
    kb5_t[pulse] = g['kb5_t'][:]
    print len(nondisruptive), len(disruptive)
    if len(disruptive) == 100 and len(nondisruptive) == 100:
        break

f.close()

print 'non-disruptive pulses:', len(nondisruptive)
print 'disruptive pulses:', len(disruptive)

pulses = np.concatenate((disruptive,nondisruptive))
np.random.shuffle(pulses)

# --------------------------------------------------------------------------

test_results = []
results = []
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
late = 0

for pulse in pulses:
    if len(kb5[pulse]) < 200:
        continue
    i = sample_size
    print results
    results = []
    x = []
    while i < len(kb5[pulse]):
        x = kb5[pulse][i-sample_size:i]
        x = np.delete(x,[18,19,20,21,22,23,38,39,46,48,49,50,51,52,53,54,55],1)
        x = np.expand_dims(x, axis=0)
        predict = model.predict(x)
        print predict
        if predict[0] < 0.100:
            results.append(pulse)
            results.append(kb5_t[pulse][i])
            results.append(predict[0][0])
            if pulse in disruptive:
                results.append(dst[pulse])
                true_positive += 1
            else:
                results.append("error: non-disruptive")
                false_positive += 1
            test_results.append(results)
            break
        i = i+3
    if i >= len(kb5[pulse]):
        results.append(pulse)
        results.append("no disruption")
        if pulse in disruptive:
            results.append("error")
            false_negative += 1
        else:
            results.append("correct")
            true_negative += 1
    test_results.append(results)

print test_results
print "TP: ", true_positive,", FP: ", false_positive,"\nTN: ", true_negative, ", FN: ", false_negative 


















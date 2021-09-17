import os
import sys
import h5py
import numpy as np
np.random.seed(2000)
from keras.models import *
from keras.layers.core import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.optimizers import *

batch_size = 96
sample_size = 100

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
    dst[data[0]]=float(data[1])
f.close()

# --------------------------------------------------------------------------

fname = '../data/kb5_data_small.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

valid_file = open('valid_pulses.txt', 'r')

kb5 = dict()
kb5_t = dict()

nondisruptive = []
disruptive = []

for line in valid_file:
    pulse = line[:-1]
    if pulse in dst:
        disruptive.append(pulse)
    else:
        nondisruptive.append(pulse)
    g = f[pulse]
    kb5[pulse] = g['kb5'][:]
    kb5_t[pulse] = g['kb5_t'][:]

f.close()
valid_file.close()

print 'non-disruptive pulses:', len(nondisruptive)
print 'disruptive pulses:', len(disruptive)

pulses = np.concatenate((disruptive,nondisruptive))
np.random.shuffle(pulses)

# --------------------------------------------------------------------------

def batch_generator(pulse, batch_size):
    X_batch = []
    i = sample_size
    while True:
        x = kb5[pulse][i-sample_size:i]
        x = np.delete(x,[18,19,20,21,22,23,38,39,46,48,49,50,51,52,53,54,55],1)
        X_batch.append(x)
        if len(X_batch) >= batch_size:
            X_batch = np.array(X_batch)
            yield X_batch
            X_batch = []
        i+=1

# --------------------------------------------------------------------------

fname = 'dec_tree_xy.hdf'
print 'Writing:', fname
f = h5py.File(fname, 'w')

n = 0
last = len(pulses)


for pulse in pulses:
    n+=1

    if len(kb5[pulse]) < 200:
        continue

    sys.stdout.write("\rLoading: pulse - %d/%d" % (n, last))
    sys.stdout.flush()

    size = round((len(kb5[pulse])-sample_size)/batch_size)

    ttd = model_ttd.predict_generator(batch_generator(pulse,batch_size), steps=size)
    prob = model.predict_generator(batch_generator(pulse,batch_size), steps=size)
    
    i=0
    count = 0
    step = 6
    sum_prob = 0
    sum_ttd = 0    
    times = []
    inputs = []
    outputs = []
    while i < len(ttd):
        sum_prob += prob[i][0]
        sum_ttd += ttd[i][0]
        count += 1
        if count >= step:
            mean_prob = sum_prob/step
            mean_ttd = sum_ttd/step
            inputpair = [mean_prob, mean_ttd]
            inputs.append(inputpair)
            currenttime = kb5_t[pulse][i+sample_size]
            times.append(currenttime)
            if pulse in dst:
                if (dst[pulse] - float(currenttime)) < 0.500:
                    outputs.append(1)
                else:
                    outputs.append(0)
            else:
                outputs.append(0)
            count = 0
            sum_prob = 0
            sum_ttd = 0
        i+=1


    g = f.create_group(pulse)
    g.create_dataset('input', data=inputs)
    g.create_dataset('output', data=outputs)
    g.create_dataset('time', data=times)

            
            
    '''

    n += 1
    i = sample_size
    x = []
    inputs = []
    outputs = []
    count = 0
    step = 6
    sum_prob = 0
    sum_ttd = 0
    while i < len(kb5[pulse]):
        x = kb5[pulse][i-sample_size:i]
        x = np.delete(x,[18,19,20,21,22,23,38,39,46,48,49,50,51,52,53,54,55],1)
        x = np.expand_dims(x, axis=0)
        predict = model.predict(x)
        predict_ttd = model_ttd.predict(x)
        sum_prob += float(predict[0][0])
        sum_ttd += float(predict_ttd[0][0])
        count += 1

        sys.stdout.write("\rLoading: pulse:%d/%d - %d/%d" % (n, last, i, len(kb5[pulse])))
        sys.stdout.flush()

        #print "prob: " + str(predict) + ", ttd: " + str(predict_ttd)
        if count >= step:
            mean_prob = sum_prob/step
            mean_ttd = sum_ttd/step
            inputpair = [mean_prob, mean_ttd]
            inputs.append(inputpair)
            if pulse in disruptive:
                outputs.append(1)
            else:
                outputs.append(0)
            count = 0
            sum_prob = 0
            sum_ttd = 0

        i += 1
    
    g = f.create_group(pulse)
    g.create_dataset('input', data=inputs)
    g.create_dataset('output', data=outputs)

'''


f.close()

print '\nDone'

import os
import sys
import h5py
import numpy as np
np.random.seed(2000)
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn import tree
from sklearn.externals import joblib

# make tensorflow use cpu instead of gpu to avoid loading data onto the gpu 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

batch_size = 1
sample_size = 100


clf = joblib.load('dec_tree.pkl') 

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name_prob = 'prob_trained_model_30abr.h5'
# Load model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name_prob)
old_model_prob = load_model(model_path)
print('Loaded trained model at %s ' % model_path)

model_name_ttd = 'convlstm_trained_model_30abr.h5'
model_path = os.path.join(save_dir, model_name_ttd)
old_model_ttd = load_model(model_path)
print('Loaded trained model at %s ' % model_path)

# --------------------------------------------------------------------------

prob_model = Sequential()

prob_model.add(Conv1D(32, 5, batch_input_shape=(batch_size, sample_size, 39)))
prob_model.add(Activation('relu'))
prob_model.add(MaxPooling1D(pool_size=2))

prob_model.add(Conv1D(64, 5))
prob_model.add(Activation('relu'))
prob_model.add(MaxPooling1D(pool_size=2))

prob_model.add(LSTM(600, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
prob_model.add(LSTM(600, dropout=0.3, recurrent_dropout=0.2))

prob_model.add(Dense(600, activation='relu'))

prob_model.add(Dense(1, activation='sigmoid'))

old_weights=old_model_prob.get_weights()
prob_model.set_weights(old_weights)

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

merged = Merge([prob_model, model_ttd], mode='concat')
model = Sequential()
model.add(merged)
model.add(Dense(1, activation='sigmoid'))

model_stacked = model

model_name_stacked = 'stacked_trained_model.h5'
model_path = os.path.join(save_dir, model_name_stacked)
model.load_weights(model_path)
#model_stacked = load_model(model_path)
print('Loaded trained model at %s ' % model_path)

#old_weights = model_stacked.get_weights()
#model.set_weights(old_weights)

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

valid_file = open('test_pulses.txt', 'r')

kb5 = dict()
kb5_t = dict()

nondisruptive = []
disruptive = []

for line in valid_file:
    pulse = line[:-1]
    if pulse not in dst:
    #    if np.random.randint(0,100) > 15 or len(nondisruptive) == 100:
    #        continue
        nondisruptive.append(pulse)
    else:
    #    if np.random.randint(0,100) > 55 or len(disruptive) == 100:
    #        continue
        disruptive.append(pulse)
    g = f[pulse]
    kb5[pulse] = g['kb5'][:]
    kb5_t[pulse] = g['kb5_t'][:]
    sys.stdout.write("\rLoading: %d %d" % (len(nondisruptive), len(disruptive)))
    sys.stdout.flush()
    #if len(disruptive) == 100 and len(nondisruptive) == 100:
    #    break

f.close()
valid_file.close()

print '\nnon-disruptive pulses:', len(nondisruptive)
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
    i = 2*sample_size
    print results
    results = []
    x = []
    x2 = []
    count = 0
    step = 6
    sum_prob = 0
    sum_ttd = 0
    while i < len(kb5[pulse]):
        x = kb5[pulse][i-sample_size:i]
        x = np.delete(x,[18,19,20,21,22,23,38,39,46,48,49,50,51,52,53,54,55],1)
        x = np.expand_dims(x, axis=0)
        #x2 = kb5[pulse][i-2*sample_size:i]
        #x2 = np.delete(x2,[18,19,20,21,22,23,38,39,46,48,49,50,51,52,53,54,55],1)
        #x2 = np.expand_dims(x2, axis=0)
        predict = model.predict([x,x])
        sum_prob += float(predict[0][0])
        count += 1

        #print "prob: " + str(predict) + ", ttd: " + str(predict_ttd)
        if count >= step:
            mean_prob = sum_prob/step
            count = 0
            sum_prob = 0
            if mean_prob >= 0.85:
                results.append(pulse)
                results.append(kb5_t[pulse][i])
                results.append(mean_prob)
                if pulse in disruptive:
                    if float(dst[pulse]) - float(kb5_t[pulse][i]) > 0.030:
                        results.append(dst[pulse])
                        true_positive += 1
                    else:
                        results.append("Late - "+str(dst[pulse]))
                        false_negative += 1
                else:
                    results.append("error: non-disruptive")
                    false_positive += 1
                test_results.append(results)
                break
        i = i+1
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

print "TP: ", true_positive,", FP: ", false_positive,"\nTN: ", true_negative, ", FN: ", false_negative 


















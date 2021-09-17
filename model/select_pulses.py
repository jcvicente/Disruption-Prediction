import os
import h5py
import numpy as np
np.random.seed(2018)

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
train_pulses = []
valid_pulses = []
test_pulses = []

k = 0
for pulse in f:
    k += 1
    if pulse not in dst:
        nondisruptive.append(pulse)
    else:
        disruptive.append(pulse)
    g = f[pulse]
    kb5[pulse] = g['kb5'][:]
    kb5_t[pulse] = g['kb5_t'][:]

f.close()

print 'non-disruptive pulses:', len(nondisruptive)
print 'disruptive pulses:', len(disruptive)


logfile = open('valid_pulses.txt', 'w').close()
logfile = open('valid_pulses.txt', 'a')
count_dis = 0
count_nondis = 0

k = 0
for pulse in nondisruptive:
	k += 1
	if k % 3 == 0 and int(pulse) > 85000:
		valid_pulses.append(pulse)
                count_nondis += 1
	else:
		train_pulses.append(pulse)

k = 0
for pulse in disruptive:
	k += 1
	if k % 3 == 0 and int(pulse) > 85000:
		valid_pulses.append(pulse)
                count_dis += 1
	else:
		train_pulses.append(pulse) 

k = 0
for pulse in valid_pulses:
    if k % 2 == 0:
        test_pulses.append(pulse)
        valid_pulses.remove(pulse)


for pulse in valid_pulses:
    logfile.write(pulse+"\n")

logfile.close()

logfile = open('test_pulses.txt', 'w').close()
logfile = open('test_pulses.txt', 'a')

for pulse in test_pulses:
    logfile.write(pulse+"\n")



print 'train:', len(train_pulses)
print 'validation:', len(valid_pulses)
print 'test:', len(test_pulses)

print 'valid non-dis:', count_nondis, ' valid dis:', count_dis

# --------------------------------------------------------------------------

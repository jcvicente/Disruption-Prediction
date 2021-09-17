import h5py

fname = 'ShotList_ILW_2011-16.txt'
print 'Reading:', fname
f = open(fname, 'r')

dst_real = dict()
pulses = []

k=0
for pulse in f:
    k = k+1
    if (k<4) or (k>1954):
        continue

    data = pulse[25:39].split(' ')
    data.pop(1)
    data.pop(1)
    dst_real[int(data[0])]=float(data[1])
    print data

f.close()

print dst_real

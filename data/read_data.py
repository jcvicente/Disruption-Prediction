
import h5py

fname = 'dst_kb5.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

dst = []
kb5 = []
kb5_t = []

for pulse in f:
    g = f[pulse]
    dst.append(g['dst'][0])
    kb5.append(g['kb5'][:])
    kb5_t.append(g['kb5_t'][:])
    print pulse, '%.4f' % dst[-1], kb5[-1].shape, kb5[-1].dtype, kb5_t[-1].shape, kb5_t[-1].dtype

f.close()

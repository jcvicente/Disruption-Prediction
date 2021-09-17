
import h5py
import numpy as np

fname = 'kb5_sample.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

for pulse in f:
    print 'pulse:', pulse
    kb5h = f[pulse]['kb5h'][:]
    kb5v = f[pulse]['kb5v'][:]
    kb5h_t = f[pulse]['kb5h_t'][:]
    kb5v_t = f[pulse]['kb5v_t'][:]
    print 'kb5h:', kb5h.shape, kb5h.dtype
    print 'kb5v:', kb5v.shape, kb5v.dtype
    print 'kb5h_t:', kb5h_t.shape, kb5h_t.dtype
    print 'kb5v_t:', kb5v_t.shape, kb5v_t.dtype

f.close()

assert np.all(kb5h_t==kb5v_t)

start_time = 0#72000
end_time = -1#start_time + 3000

kb5_t = kb5h_t[start_time:end_time]

kb5 = np.hstack([kb5v, kb5h])

print 'kb5:', kb5.shape, kb5.dtype

import matplotlib.pyplot as plt

for j in range(kb5.shape[1]):
    if (j == 15) or (j == 22):
        continue
    plt.plot(kb5_t, kb5[start_time:end_time,j], label=str(j))

#plt.title("KB5 readings for JET pulse no. 92213")
plt.xlabel("Time (s)")
plt.ylabel("Radiated Power (W/m$^2$)")
plt.show()

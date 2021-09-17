
import h5py
import numpy as np

# ---------------------------------------------------------------------------------

from ppf import *

ppfgo()
ppfuid('jetppf', 'r')

def get_data(pulse, dda, dtyp):
    ihdata, iwdata, data, x, t, ier = ppfget(pulse, dda, dtyp, reshape=1)
    return data, t

fname = 'kb5_sample.hdf'
print 'Writing:', fname
f = h5py.File(fname, 'w')

pulse = 92213

kb5h, kb5h_t = get_data(pulse, 'bolo', 'kb5h')
kb5v, kb5v_t = get_data(pulse, 'bolo', 'kb5v')

print 'kb5h:', kb5h.shape, kb5h.dtype
print 'kb5v:', kb5v.shape, kb5v.dtype
print 'kb5h_t:', kb5h_t.shape, kb5h_t.dtype
print 'kb5v_t:', kb5v_t.shape, kb5v_t.dtype

g = f.create_group(str(pulse))
g.create_dataset('kb5h', data=kb5h)
g.create_dataset('kb5v', data=kb5v)
g.create_dataset('kb5h_t', data=kb5h_t)
g.create_dataset('kb5v_t', data=kb5v_t)

f.close()

# ---------------------------------------------------------------------------------

fname = 'kb5_sample.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

for pulse in f:
    print 'pulse:', pulse
    g = f[pulse]
    kb5h = f[pulse]['kb5h'][:]
    kb5v = f[pulse]['kb5v'][:]
    kb5h_t = f[pulse]['kb5h_t'][:]
    kb5v_t = f[pulse]['kb5v_t'][:]
    print 'kb5h:', kb5h.shape, kb5h.dtype
    print 'kb5v:', kb5v.shape, kb5v.dtype
    print 'kb5h_t:', kb5h_t.shape, kb5h_t.dtype
    print 'kb5v_t:', kb5v_t.shape, kb5v_t.dtype

f.close()

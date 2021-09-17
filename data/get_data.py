
import h5py
import numpy as np
from MDSplus import *

host = 'mdsplus.jet.efda.org'
print 'Connecting:', host
conn = Connection(host)

def get_data(pulse, ppf, dim):
    expr = '_s=jet("%s",%d)' % (ppf, pulse)
    data = conn.get(expr).data()
    expr = 'dim_of(_s,%d)' % dim
    t = conn.get(expr).data()
    if (len(data) < 2) or (len(t) < 2):
        raise KeyError
    return data, t

def get_dst(ipla, ipla_t):
    x0 = ipla_t[:-1]
    x1 = ipla_t[1:]
    y0 = ipla[:-1]
    y1 = ipla[1:]
    grad = (y1-y0)/(x1-x0)
    grad_t = x0 + (x1-x0)/2.
    dst = None
    lim = 20e6 # 20 MA/s
    pos = np.where(grad > lim)[0]
    if len(pos) > 0:
        i = pos[0]
        x0 = grad_t[i-1]
        x1 = grad_t[i]
        y0 = grad[i-1]
        y1 = grad[i]
        m = (y1-y0)/(x1-x0)
        b = (y0*x1-y1*x0)/(x1-x0)
        dst = (lim-b)/m
        dst = round(dst, 4)
    return dst

fname = 'dst_kb5.hdf'
print 'Writing:', fname
f = h5py.File(fname, 'w')

pulse0 = 79886
pulse1 = 92504

for pulse in range(pulse0, pulse1+1):
    print pulse,
    
    try:
        ipla, ipla_t = get_data(pulse, 'PPF/MAGN/IPLA', 0)
    except KeyError:
        print 'no ipla'
        continue

    dst = get_dst(ipla, ipla_t)
    if dst == None:
        print 'no dst'
        continue
        
    try:
        kb5h, kb5h_t = get_data(pulse, 'PPF/BOLO/KB5H', 1)
        kb5v, kb5v_t = get_data(pulse, 'PPF/BOLO/KB5V', 1)
    except KeyError:
        print 'no kb5'
        continue
    
    kb5 = np.hstack((kb5h, kb5v))
    kb5_t = kb5h_t
    
    step = round(np.mean(kb5_t[1:]-kb5_t[:-1]), 4)
    if step != 0.0002:
        print 'no step'
        continue
    
    n = 5
    kb5 = np.cumsum(kb5, axis=0)
    kb5 = (kb5[n:]-kb5[:-n])/n
    kb5 = kb5[::n]
    kb5_t = kb5_t[n/2+1::n]

    t = 40.
    i = np.argmin(np.fabs(kb5_t-t))
    if kb5_t[i] < t:
        i += 1
    kb5 = kb5[i:]
    kb5_t = kb5_t[i:]
        
    t = dst
    i = np.argmin(np.fabs(kb5_t-t))
    if kb5_t[i] > t:
        i -= 1
    kb5 = kb5[:i+1]
    kb5_t = kb5_t[:i+1]
    
    print '%.4f' % dst, kb5.shape, '%.4f' % kb5_t[0], '%.4f' % kb5_t[-1]
    
    g = f.create_group(str(pulse))
    g.create_dataset('dst', data=[dst])
    g.create_dataset('kb5', data=kb5, compression='gzip', compression_opts=9)
    g.create_dataset('kb5_t', data=kb5_t, compression='gzip', compression_opts=9)

f.close()

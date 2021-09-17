import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File("kb5_sample.hdf", 'r')

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

start_time = 49500
end_time = start_time + 28960

kb5_t = kb5h_t[start_time:end_time]

kb5 = np.hstack([kb5v, kb5h])

print 'kb5:', kb5.shape, kb5.dtype


numpy_kb5v = np.array(kb5v)
numpy_kb5v = numpy_kb5v[start_time:end_time]
kb5v_averages = np.average(numpy_kb5v, 0)
kb5v_averages = kb5v_averages/100000
kb5v_averages[15]= -1
kb5v_averages[22]= -1

print np.round(kb5v_averages, 3)

plt.plot(kb5v_averages, 'ro', alpha=0.5)
for i in range(kb5v_averages.shape[0]):
    plt.text(i, kb5v_averages[i], str(i+1))
#plt.title("KB5 readings for JET pulse no. 92213")
plt.xlabel("Channels")
plt.ylabel("Radiated Power (W/m$^2$ x1e6)")
plt.show()


numpy_kb5h = np.array(kb5h)
numpy_kb5h = numpy_kb5h[start_time:end_time]
kb5h_averages = np.average(numpy_kb5h, 0)
kb5h_averages = kb5h_averages/100000

print np.round(kb5h_averages, 3)

plt.plot(kb5h_averages, 'o', alpha=0.5)
for i in range(kb5h_averages.shape[0]):
    plt.text(i, kb5h_averages[i], str(i+1))
#plt.title("KB5 readings for JET pulse no. 92213")
plt.xlabel("Channels")
plt.ylabel("Radiated Power (W/m$^2$ x1e6)")
plt.show()


kb5v_min=numpy_kb5v.min(0)
kb5v_min = kb5v_min/100000
kb5v_min[15]=-1
kb5v_min[22]=-1
for i in range(kb5v_min.shape[0]):
  x = np.round(kb5v_min[i],3)
  kb5v_min[i] = x
print kb5v_min

kb5v_max=numpy_kb5v.max(0)
kb5v_max = kb5v_max/100000
kb5v_max[15]=-1
kb5v_max[22]=-1
for i in range(kb5v_max.shape[0]):
  x = np.round(kb5v_max[i],3)
  kb5v_max[i] = x
print kb5v_max


plt.plot(kb5v_min, 'bo', alpha=0.5)
for i in range(kb5v_min.shape[0]):
    plt.text(i, kb5v_min[i], str(i+1))
plt.plot(kb5v_max, 'ro', alpha=0.5)
for i in range(kb5v_max.shape[0]):
    plt.text(i, kb5v_max[i], str(i+1))
plt.xlabel("Channels")
plt.ylabel("Radiated Power (W/m$^2$ x1e6)")
plt.show()



kb5h_min=numpy_kb5h.min(0)
kb5h_min = kb5h_min/100000
for i in range(kb5h_min.shape[0]):
  x = np.round(kb5h_min[i],3)
  kb5h_min[i] = x
print kb5h_min

kb5h_max=numpy_kb5h.max(0)
kb5h_max = kb5h_max/100000
for i in range(kb5h_max.shape[0]):
  x = np.round(kb5h_max[i],3)
  kb5h_max[i] = x
print kb5h_max


plt.plot(kb5h_min, 'bo', alpha=0.5)
for i in range(kb5h_min.shape[0]):
    plt.text(i, kb5h_min[i], str(i+1))
plt.plot(kb5h_max, 'ro', alpha=0.5)
for i in range(kb5h_max.shape[0]):
    plt.text(i, kb5h_max[i], str(i+1))
plt.xlabel("Channels")
plt.ylabel("Radiated Power (W/m$^2$ x1e6)")
plt.show()


kb5v_std = np.std(numpy_kb5v, 0)
kb5v_std = kb5v_std/100000
kb5v_std[15]=-1
kb5v_std[22]=-1
print np.round(kb5v_std,3)

kb5h_std = np.std(numpy_kb5h, 0)
kb5h_std = kb5h_std/100000
print np.round(kb5h_std,3)

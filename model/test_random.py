import numpy as np
import matplotlib.pyplot as P

x = 0
y=[]
while x < 9000:
    i = np.random.exponential(scale=1.0)
    while i > 5:
        i = np.random.exponential(scale=1.0)
    y.append(i)
    
    x+=1

for number in y:
    if number <= 0:
        print number


a = np.clip(y, 0, 5)

number_density, radii = np.histogram(a, bins=100,normed=False)
P.plot(radii[0:-1], number_density)
P.xlabel('$R$')
P.ylabel(r'$\Sigma$')
P.legend()      
P.show()

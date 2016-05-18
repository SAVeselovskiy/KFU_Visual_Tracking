import random
import matplotlib.pyplot as plt
N=31
y=[0.816957]
for i in xrange(N-1):
    y.append(y[-1]+random.random()/10/(i+1)**(1.9))
x=range(1,len(y)+1)
plt.plot(x,y,linewidth=2)
plt.grid(axis='both')
for elem in y:
    print "%0.6f" % elem

plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Question 1
x = np.linspace(1.920,2.080,161)
p1 = x**9 - 18*(x**8) + 144*(x**7) - 672*(x**6) + 2016*(x**5) - 4032*(x**4) + 5376*(x**3) - 4608*(x**2) + 2304*x - 512
p2 = (x-2)**9
plt.plot(x,p1,'-r',label="Expanded Form")
plt.plot(x,p2,'-b',label=r"$(x-2)^9$")
plt.legend(loc = "upper left")
plt.title("Question 1")
plt.xlabel("x")
plt.ylabel(r"$p(x)$")
plt.show()

'''
Expanded form has a lot more static. This is likely due to floating point error,
which is a result of the numerous additions and subtractions. Since some of the coefficients
are notably larger, it is likely that some terms get rounded off which produces the static.
'''

# Question 5
x1 = np.pi
x2 = 10**6
delta = np.array([10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,1])
y1 = (np.cos(2*x1+delta)+np.cos(delta))/2 - (1+np.cos(2*x1))/2
y2 = (np.cos(2*x2+delta)+np.cos(delta))/2 - (1+np.cos(2*x2))/2
plt.semilogx(delta,y1,'-or',label=r"$x=\pi$")
plt.semilogx(delta,y2,'-ob',label=r"$x=10^6$")
plt.legend(loc='lower left')
plt.title("Question 5")
plt.xlabel(r"$\delta$")
plt.ylabel(r"$cos(x+\delta)-cos(x)$")
plt.show()

y3 = -delta*np.sin(x1)-delta**2/2*np.cos(x1+delta/2)
y4 = -delta*np.sin(x2)-delta**2/2*np.cos(x2+delta/2)
plt.semilogx(delta,y3,'-or',label=r"$x=\pi$")
plt.semilogx(delta,y4,'-ob',label=r"$x=10^6$")
plt.legend(loc='lower left')
plt.title("Question 5c")
plt.xlabel(r"$\delta$")
plt.ylabel(r"$cos(x+\delta)-cos(x)$")
plt.show()



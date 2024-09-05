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
# Question 3
x = np.linspace(0,1,1000)
y = x**3/6*(4*np.cos(x/2)-2*np.sin(x/2))
print(np.trapz(y,x))



import numpy as np
import matplotlib.pyplot as plt

# Question 3c
y = np.exp(9.999999995000000*10**(-10))
print(y-1)
y = np.expm1(9.999999995000000*10**(-10))
print(y)

# Question 3d
x = 9.999999995000000*10**(-10)
y2 = x + x**2/2
print(y2/y) 
# This prints 1.0, showing that my result is indistinguishable from expm1...
# and is almost certainly accurate

# Question 4
t = np.linspace(0,np.pi,31)
y = np.cos(t)
s = np.sum(t*y)
print("S =",s)
# part b
tab20_colors = plt.cm.get_cmap('tab20').colors
colors=[]
c2=[]
for i in range(20):
    if i%2 == 0:
        colors.append(tab20_colors[i])
    else:
        c2.append(tab20_colors[i])
colors.extend(c2)
fig = plt.figure(figsize=(11,5))
ax = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
R = 1.2
delr = 0.1
f = 15
p = 0
theta = np.linspace(0,np.pi*2,1000)
x = R*(1+delr*np.sin(f*theta+p))*np.cos(theta)
y = R*(1+delr*np.sin(f*theta+p))*np.sin(theta)
ax.plot(x,y,'-r')
ax.set_xlabel("x")
ax.set_ylabel("y")
p2 = np.random.uniform(0,2,10)
for i in range(10):
    R = i + 1
    f = 3 + i
    x = R*(1+delr*np.sin(f*theta+p2[i]))*np.cos(theta)
    y = R*(1+delr*np.sin(f*theta+p2[i]))*np.sin(theta)
    ax2.plot(x,y,linestyle='-',color=colors[i],label="i = " + str(i+1))
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend(loc="upper right")
ax.set_aspect('equal')
ax2.set_aspect('equal')
fig.savefig("Q4b.png")

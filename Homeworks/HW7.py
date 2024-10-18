import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv 

def f1(x):
    return 1/(1+100*x**2)

def problem1(f):
    n = 18
    h = 2/(n-1)
    x = np.zeros(n)
    for i in range(n):
        x[i] = -1 + (i)*h
    v = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            v[i,j] = x[i]**j
    vinv = inv(v)
    y = np.zeros(n)
    for i in range(n):
        y[i] = f(x[i])
    c = np.matmul(vinv,y)
    neval = 1001
    x2 = np.linspace(-1,1,neval)
    y_fake = np.zeros(neval)
    y_true = np.zeros(neval)
    for i in range(neval):
        y_true[i] = f(x2[i])
        for j in range(n):
            y_fake[i] += c[j]*x2[i]**j

    plt.plot(x,y,'or')
    plt.plot(x2,y_true,'-g')
    plt.plot(x2,y_fake,'-b')
    plt.ylim(-2,2)
    plt.show()

def problem2(f):
    n = 80
    h = 2/(n-1)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[i] = -1 + (i)*h
    for i in range(n):
        y[i] = f(x[i])
    w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            else:
                w[i] *= (x[i]-x[j])

    n_interp = 2000
    x2 = np.linspace(-1,1,n_interp)
    y_fake = np.zeros(n_interp)
    y_true = np.zeros(n_interp)
    phi = np.zeros(n_interp)
    for i in range(n_interp):
        val = 1
        for j in range(n):
            val *= (x2[i]-x[j])
        phi[i] = val
    for i in range(n_interp):
        sum = 0
        for j in range(n):
            if x2[i] == x[j]:
                continue
            else:
                sum += 1/w[j]/(x2[i] - x[j])*f(x[j])
        y_fake[i] = phi[i]*sum
    for i in range(n_interp):
        y_true[i] = f(x2[i])
    
    plt.plot(x,y,'or')
    plt.plot(x2,y_true,'-g')
    plt.plot(x2,y_fake,'-b')
    plt.ylim(-2,2)
    plt.show()

def problem3(f):
    n = 2
    x = np.zeros(n)
    for i in range(n):
        x[i] = np.cos((2*(i+1)-1)*np.pi/2/n)
    v = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            v[i,j] = x[i]**j
    vinv = inv(v)
    y = np.zeros(n)
    for i in range(n):
        y[i] = f(x[i])
    c = np.matmul(vinv,y)
    neval = 1001
    x2 = np.linspace(-1,1,neval)
    y_fake = np.zeros(neval)
    y_true = np.zeros(neval)
    for i in range(neval):
        y_true[i] = f(x2[i])
        for j in range(n):
            y_fake[i] += c[j]*x2[i]**j

    plt.plot(x,y,'or')
    plt.plot(x2,y_true,'-g')
    plt.plot(x2,y_fake,'-b')
    plt.show()

def main():
    problem1(f1)
    problem2(f1)
    problem3(f1)

main()
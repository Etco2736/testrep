import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return 1/(1+x**2)

def trap(f,x):
    size = len(x)
    h = 10/size
    res = 0
    for i in range(size):
        if i == 0 or i == size-1:
            res += f(x[i])
        else:
            res += 2*f(x[i])
    res = res * h/2
    return res

def simps(f,x):
    size = len(x)
    h = 10/size
    res = 0
    for i in range(size):
        if i == 0 or i == size-1:
            res += f(x[i])
        elif i % 2 == 0:
            res += 2*f(x[i])
        else:
            res += 4*f(x[i])
    res = res * h/3
    return res

def main():
    x = np.linspace(-5,5,1291)
    x2 = np.linspace(-5,5,193)
    v1 = trap(f,x)
    v2 = simps(f,x2)
    v3, err, info = quad(f,-5,5,full_output=True)
    neval1 = info['neval']
    v4, err, info = quad(f,-5,5,epsabs=1e-4,full_output=True)
    neval2 = info['neval']
    print(v1,v2,v3,v4)
    print(neval1,neval2)
    return
main()
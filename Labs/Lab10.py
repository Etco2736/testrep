import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def eval_legendre(n,x):
    phi = np.zeros(n+1)
    phi[0] = 1
    if n > 0:
        phi[1] = x
    for i in range(2,n+1):
        phi[i] = 1/(n+1)*((2*n+1)*x*phi[i-1]-n*phi[i-2])
    return phi

def eval_legendre_expansion(f,a,b,w,n,x_val):
    p = eval_legendre(n,x_val)
    pval = 0.0
    for j in range(0,n+1):
    # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: eval_legendre(j,x)[j]
    # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: eval_legendre(j,x)[j]**2*w(x)
    # use the quad function from scipy to evaluate normalizations
        norm_fac,err = quad(phi_j_sq,a=a,b=b)
    # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: phi_j(x)*f(x)*w(x)/norm_fac
    # use the quad function from scipy to evaluate coeffs
        aj,err = quad(func_j,a=a,b=b)
    # accumulate into pval
        pval = pval+aj*p[j]
    return pval


def driver():
    # function you want to approximate
    f = lambda x: 1/(1+x**2)
    # Interval of interest
    a = -1
    b = 1
    # weight function
    w = lambda x: 1.
    # order of approximation
    n = 8
    # Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)
    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
    plt.figure()
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion')
    plt.legend()
    plt.show()
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',label='error')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    driver()
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm

def driver():
    f = lambda x: 1/(1+(10*x)**2)
    N = 20
    ''' interval'''
    a = -1
    b = 1
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    for j in range(N+1):
        xint[j] = np.cos(np.pi*(2*j-1)/2/N)
    ''' create interpolation data'''
    yint = f(xint)
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
    yeval_m = np.zeros(Neval+1)
    '''Initialize and populate the first columns of the
    divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
    for j in range(N+1):
        y[j][0] = yint[j]
    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
        yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)

    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)

    ''' Invert the Vandermonde matrix'''
    # Vinv = inv(V)
    # print('Vinv = ' , Vinv)
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    # coef = Vinv @ yint
    # print('coef = ', coef)
    # No validate the code
    # yeval_m = eval_monomial(xeval,coef,N,Neval)
    ''' create vector with exact values'''
    fex = f(xeval)
    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval_l,'bs--')
    plt.plot(xeval,yeval_dd,'c.--')
    # plt.plot(xeval,yeval_m,'g.--')
    plt.legend()
    plt.figure()
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    err_m = abs(yeval_m-fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.semilogy(xeval,err_dd,'bs--',label='Newton DD')
    # plt.semilogy(xeval,err_m,'gs--',label='Monomial')
    plt.legend()
    plt.show()
    
def eval_lagrange(xeval,xint,yint,N):
    lj = np.ones(N+1)
    for count in range(N+1):
        for jj in range(N+1):
            if (jj != count):
                lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])
    yeval = 0.
    for jj in range(N+1):
        yeval = yeval + yint[jj]*lj[jj]
    return(yeval)
''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
            (x[j] - x[i + j]))
    return y
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    ptmp[0] = 1.
    for j in range(N):
        ptmp[j+1] = ptmp[j]*(xval-xint[j])
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
        yeval = yeval + y[0][j]*ptmp[j]
    return yeval
def eval_monomial(xeval,coef,N,Neval):
    yeval = coef[0]*np.ones(Neval+1)
    # print('yeval = ', yeval)
    for j in range(1,N+1):
        for i in range(Neval+1):
            # print('yeval[i] = ', yeval[i])
            # print('a[j] = ', a[j])
            # print('i = ', i)
            # print('xeval[i] = ', xeval[i])
            yeval[i] = yeval[i] + coef[j]*xeval[i]**j
    return yeval
def Vandermonde(xint,N):
    V = np.zeros((N+1,N+1))
    ''' fill the first column'''
    for j in range(N+1):
        V[j][0] = 1.0
    for i in range(1,N+1):
        for j in range(N+1):
            V[j][i] = xint[j]**i
    return V

driver()
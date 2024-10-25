import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1/(1+x**2)

def fp(x):
    return -2*x/(1+x**2)**2

def lagrange(xp, yp, xsol, ysol):
    n = len(xp)
    n2 = len(xsol)

    for k in range(n2):
        x = xsol[k]
        y = 0
        for i in range(n):
            L_i = 1.0
            for j in range(n):
                if i != j:
                    L_i *= (x - xp[j]) / (xp[i] - xp[j])
            y += L_i * yp[i]
        ysol[k] = y
    
    return ysol

def hermite(xp, yp, dyp, xsol, ysol):
    n = len(xp)
    n2 = len(xsol)
    
    for k in range(n2):
        x = xsol[k]
        y = 0
        for i in range(n):
            h_i = 1
            h_prime = 0

            for j in range(n):
                if i != j:
                    h_i *= (x - xp[j]) / (xp[i] - xp[j])
                    h_prime += 1 / (xp[i] - xp[j])

            H1 = (1 - 2 * h_prime * (x - xp[i])) * h_i**2
            H2 = (x - xp[i]) * h_i**2

            y += H1 * yp[i] + H2 * dyp[i]
        ysol[k] = y
    
    return ysol

def nat_spline(xp, yp, xsol, ysol):
    n = len(xp) - 1
    h = np.diff(xp)

    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)

    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3 * ((yp[i+1] - yp[i]) / h[i] - (yp[i] - yp[i-1]) / h[i-1])

    c = np.linalg.solve(A, b)

    a = yp[:-1]
    b_coeff = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b_coeff[i] = (yp[i+1] - yp[i]) / h[i] - h[i] * (2 * c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])

    ysol = np.zeros_like(xsol)
    for j, x in enumerate(xsol):
        for i in range(n):
            if xp[i] <= x <= xp[i+1]:
                dx = x - xp[i]
                ysol[j] = a[i] + b_coeff[i] * dx + c[i] * dx**2 + d[i] * dx**3
                break

    return ysol

def clamp_spline(xp, yp, xsol, ysol, fp1, fp2):
    n = len(xp) - 1
    h = np.diff(xp)

    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)

    A[0, 0] = 2 * h[0]
    A[0, 1] = h[0]
    b[0] = 3 * ((yp[1] - yp[0]) / h[0] - fp1) 

    A[n, n-1] = h[n-1]
    A[n, n] = 2 * h[n-1]
    b[n] = 3 * (fp2 - (yp[n] - yp[n-1]) / h[n-1]) 

    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3 * ((yp[i+1] - yp[i]) / h[i] - (yp[i] - yp[i-1]) / h[i-1])

    c = np.linalg.solve(A, b)

    a = yp[:-1]
    b_coeff = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b_coeff[i] = (yp[i+1] - yp[i]) / h[i] - h[i] * (2 * c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])

    ysol = np.zeros_like(xsol)
    for j, x in enumerate(xsol):
        for i in range(n):
            if xp[i] <= x <= xp[i+1]:
                dx = x - xp[i]
                ysol[j] = a[i] + b_coeff[i] * dx + c[i] * dx**2 + d[i] * dx**3
                break

    return ysol

def q1():
    x1 = np.linspace(-5,5,5)
    x2 = np.linspace(-5,5,10)
    x3 = np.linspace(-5,5,15)
    x4 = np.linspace(-5,5,20)
    y1 = np.zeros_like(x1)
    y2 = np.zeros_like(x2)
    y3 = np.zeros_like(x3)
    y4 = np.zeros_like(x4)
    dy1 = np.zeros_like(x1)
    dy2 = np.zeros_like(x2)
    dy3 = np.zeros_like(x3)
    dy4 = np.zeros_like(x4)
    n1 = len(x1)
    n2 = len(x2)
    n3 = len(x3)
    n4 = len(x4)
    for i in range(n1):
        y1[i] = f(x1[i])
        dy1[i] = fp(x1[i])
    for i in range(n2):
        y2[i] = f(x2[i])
        dy2[i] = fp(x2[i])
    for i in range(n3):
        y3[i] = f(x3[i])
        dy3[i] = fp(x3[i])
    for i in range(n4):
        y4[i] = f(x4[i])
        dy4[i] = fp(x4[i])

    xsol = np.linspace(-5,5,1000)
    nsol = len(xsol)
    ysol11 = np.zeros_like(xsol)
    ysol12 = np.zeros_like(xsol)
    ysol13 = np.zeros_like(xsol)
    ysol14 = np.zeros_like(xsol)
    ysol21 = np.zeros_like(xsol)
    ysol22 = np.zeros_like(xsol)
    ysol23 = np.zeros_like(xsol)
    ysol24 = np.zeros_like(xsol)
    ysol31 = np.zeros_like(xsol)
    ysol32 = np.zeros_like(xsol)
    ysol33 = np.zeros_like(xsol)
    ysol34 = np.zeros_like(xsol)
    ysol41 = np.zeros_like(xsol)
    ysol42 = np.zeros_like(xsol)
    ysol43 = np.zeros_like(xsol)
    ysol44 = np.zeros_like(xsol)
    ytrue = np.zeros_like(xsol)
    for i in range(nsol):
        ytrue[i] = f(xsol[i])

    ysol11 = lagrange(x1,y1,xsol,ysol11)
    ysol12 = hermite(x1,y1,dy1,xsol,ysol12)
    ysol13 = nat_spline(x1,y1,xsol,ysol13)
    ysol14 = clamp_spline(x1,y1,xsol,ysol14,1,1)
    ysol21 = lagrange(x2,y2,xsol,ysol21)
    ysol22 = hermite(x2,y2,dy2,xsol,ysol22)
    ysol23 = nat_spline(x2,y2,xsol,ysol23)
    ysol24 = clamp_spline(x2,y2,xsol,ysol24,1,1)
    ysol31 = lagrange(x3,y3,xsol,ysol31)
    ysol32 = hermite(x3,y3,dy3,xsol,ysol32)
    ysol33 = nat_spline(x3,y3,xsol,ysol33)
    ysol34 = clamp_spline(x3,y3,xsol,ysol34,1,1)
    ysol41 = lagrange(x4,y4,xsol,ysol41)
    ysol42 = hermite(x4,y4,dy4,xsol,ysol42)
    ysol43 = nat_spline(x4,y4,xsol,ysol43)
    ysol44 = clamp_spline(x4,y4,xsol,ysol44,1,1)

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    ax.plot(xsol, ysol11 - ytrue, '-b', label="Lagrange")
    ax.plot(xsol, ysol12 - ytrue, '-r', label="Hermite")
    ax.plot(xsol, ysol13 - ytrue, '-g', label="Natural Spline")
    ax.plot(xsol, ysol14 - ytrue, '-m', label="Clamped Spline")
    ax2.plot(xsol, ysol21 - ytrue, '-b', label="Lagrange")
    ax2.plot(xsol, ysol22 - ytrue, '-r', label="Hermite")
    ax2.plot(xsol, ysol23 - ytrue, '-g', label="Natural Spline")
    ax2.plot(xsol, ysol24 - ytrue, '-m', label="Clamped Spline")
    ax3.plot(xsol, ysol31 - ytrue, '-b', label="Lagrange")
    ax3.plot(xsol, ysol32 - ytrue, '-r', label="Hermite")
    ax3.plot(xsol, ysol33 - ytrue, '-g', label="Natural Spline")
    ax3.plot(xsol, ysol34 - ytrue, '-m', label="Clamped Spline")
    ax4.plot(xsol, ysol41 - ytrue, '-b', label="Lagrange")
    ax4.plot(xsol, ysol42 - ytrue, '-r', label="Hermite")
    ax4.plot(xsol, ysol43 - ytrue, '-g', label="Natural Spline")
    ax4.plot(xsol, ysol44 - ytrue, '-m', label="Clamped Spline")
    ax.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')
    ax4.legend(loc='upper right')
    ax.set_ylim(-1,1)
    ax2.set_ylim(-1,1)
    ax3.set_ylim(-1,1)
    ax4.set_ylim(-1,1)
    plt.show()

def q2():
    x1 = np.zeros(5)
    x2 = np.zeros(10)
    x3 = np.zeros(15)
    x4 = np.zeros(20)
    for i in range(5):
        x1[i] = 5*np.cos((2*i+1)/2/5*np.pi)
    for i in range(10):
        x2[i] = 5*np.cos((2*i+1)/2/10*np.pi)
    for i in range(15):
        x3[i] = 5*np.cos((2*i+1)/2/15*np.pi)
    for i in range(20):
        x4[i] = 5*np.cos((2*i+1)/2/20*np.pi)

    y1 = np.zeros_like(x1)
    y2 = np.zeros_like(x2)
    y3 = np.zeros_like(x3)
    y4 = np.zeros_like(x4)
    dy1 = np.zeros_like(x1)
    dy2 = np.zeros_like(x2)
    dy3 = np.zeros_like(x3)
    dy4 = np.zeros_like(x4)
    n1 = len(x1)
    n2 = len(x2)
    n3 = len(x3)
    n4 = len(x4)
    for i in range(n1):
        y1[i] = f(x1[i])
        dy1[i] = fp(x1[i])
    for i in range(n2):
        y2[i] = f(x2[i])
        dy2[i] = fp(x2[i])
    for i in range(n3):
        y3[i] = f(x3[i])
        dy3[i] = fp(x3[i])
    for i in range(n4):
        y4[i] = f(x4[i])
        dy4[i] = fp(x4[i])

    xsol = np.linspace(-5,5,1000)
    nsol = len(xsol)
    ysol11 = np.zeros_like(xsol)
    ysol12 = np.zeros_like(xsol)
    ysol13 = np.zeros_like(xsol)
    ysol14 = np.zeros_like(xsol)
    ysol21 = np.zeros_like(xsol)
    ysol22 = np.zeros_like(xsol)
    ysol23 = np.zeros_like(xsol)
    ysol24 = np.zeros_like(xsol)
    ysol31 = np.zeros_like(xsol)
    ysol32 = np.zeros_like(xsol)
    ysol33 = np.zeros_like(xsol)
    ysol34 = np.zeros_like(xsol)
    ysol41 = np.zeros_like(xsol)
    ysol42 = np.zeros_like(xsol)
    ysol43 = np.zeros_like(xsol)
    ysol44 = np.zeros_like(xsol)
    ytrue = np.zeros_like(xsol)
    for i in range(nsol):
        ytrue[i] = f(xsol[i])

    ysol11 = lagrange(x1,y1,xsol,ysol11)
    ysol12 = hermite(x1,y1,dy1,xsol,ysol12)
    ysol13 = nat_spline(x1,y1,xsol,ysol13)
    ysol14 = clamp_spline(x1,y1,xsol,ysol14,1,1)
    ysol21 = lagrange(x2,y2,xsol,ysol21)
    ysol22 = hermite(x2,y2,dy2,xsol,ysol22)
    ysol23 = nat_spline(x2,y2,xsol,ysol23)
    ysol24 = clamp_spline(x2,y2,xsol,ysol24,1,1)
    ysol31 = lagrange(x3,y3,xsol,ysol31)
    ysol32 = hermite(x3,y3,dy3,xsol,ysol32)
    ysol33 = nat_spline(x3,y3,xsol,ysol33)
    ysol34 = clamp_spline(x3,y3,xsol,ysol34,1,1)
    ysol41 = lagrange(x4,y4,xsol,ysol41)
    ysol42 = hermite(x4,y4,dy4,xsol,ysol42)
    ysol43 = nat_spline(x4,y4,xsol,ysol43)
    ysol44 = clamp_spline(x4,y4,xsol,ysol44,1,1)

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    ax.plot(xsol, ysol11 - ytrue, '-b', label="Lagrange")
    ax.plot(xsol, ysol12 - ytrue, '-r', label="Hermite")
    ax.plot(xsol, ysol13 - ytrue, '-g', label="Natural Spline")
    ax.plot(xsol, ysol14 - ytrue, '-m', label="Clamped Spline")
    ax2.plot(xsol, ysol21 - ytrue, '-b', label="Lagrange")
    ax2.plot(xsol, ysol22 - ytrue, '-r', label="Hermite")
    ax2.plot(xsol, ysol23 - ytrue, '-g', label="Natural Spline")
    ax2.plot(xsol, ysol24 - ytrue, '-m', label="Clamped Spline")
    ax3.plot(xsol, ysol31 - ytrue, '-b', label="Lagrange")
    ax3.plot(xsol, ysol32 - ytrue, '-r', label="Hermite")
    ax3.plot(xsol, ysol33 - ytrue, '-g', label="Natural Spline")
    ax3.plot(xsol, ysol34 - ytrue, '-m', label="Clamped Spline")
    ax4.plot(xsol, ysol41 - ytrue, '-b', label="Lagrange")
    ax4.plot(xsol, ysol42 - ytrue, '-r', label="Hermite")
    ax4.plot(xsol, ysol43 - ytrue, '-g', label="Natural Spline")
    ax4.plot(xsol, ysol44 - ytrue, '-m', label="Clamped Spline")
    ax.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')
    ax4.legend(loc='upper right')
    ax.set_ylim(-1,1)
    ax2.set_ylim(-1,1)
    ax3.set_ylim(-1,1)
    ax4.set_ylim(-1,1)
    plt.show()

def main():
    # Question 1
    q1()
    # Question 2
    q2()
main()
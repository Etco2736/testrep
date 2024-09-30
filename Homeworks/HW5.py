import numpy as np
import matplotlib.pyplot as plt

def f1(x,y):
    return 3*x**2-y**2
def g1(x,y):
    return 3*x*y**2-x**3-1
def f1x(x,y):
    return 6*x
def f1y(x,y):
    return -2*y
def g1x(x,y):
    return 3*y**2-3*x**2
def g1y(x,y):
    return 6*x*y
def f3(x,y,z):
    return x**2+4*y**2+4*z**2-16
def f3x(x,y,z):
    return 2*x
def f3y(x,y,z):
    return 8*y
def f3z(x,y,z):
    return 8*z

def q1(f,g,x0,y0,tol,max):
    count = 0
    err = 500
    errs = []
    while err > tol:
        x1 = x0 - (f(x0,y0)/6 - g(x0,y0)/18)
        y1 = y0 - g(x0,y0)/6
        err = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        x0 = x1
        y0 = y1
        count = count + 1
        errs.append(err)
        if count > max:
            print("Did Not Converge")
            break
    return x1,y1,count,errs

def newtons(f,g,fx,fy,gx,gy,x0,y0,tol,max):
    count = 0
    err = 500
    errs = []
    jacobian = np.zeros((2,2))
    j_inv = np.zeros((2,2))
    while err > tol:
        jacobian[0][0] = fx(x0,y0)
        jacobian[0][1] = fy(x0,y0)
        jacobian[1][0] = gx(x0,y0)
        jacobian[1][1] = gy(x0,y0)
        j_inv = np.linalg.inv(jacobian)
        x1 = x0 - (j_inv[0][0]*f(x0,y0)+j_inv[0][1]*g(x0,y0))
        y1 = y0 - (j_inv[1][0]*f(x0,y0)+j_inv[1][1]*g(x0,y0))
        err = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        x0 = x1
        y0 = y1
        count = count + 1
        errs.append(err)
        if count > max:
            print("Did Not Converge")
            break
    return x1,y1,count,errs

def newton_3d(f,fx,fy,fz,x0,y0,z0,tol,max):
    count = 0
    err = 500
    errs = []
    while err > tol:
        x1 = x0 - f(x0,y0,z0)/(fx(x0,y0,z0)**2 + fy(x0,y0,z0)**2 + fz(x0,y0,z0)**2)*fx(x0,y0,z0)
        y1 = y0 - f(x0,y0,z0)/(fx(x0,y0,z0)**2 + fy(x0,y0,z0)**2 + fz(x0,y0,z0)**2)*fy(x0,y0,z0)
        z1 = z0 - f(x0,y0,z0)/(fx(x0,y0,z0)**2 + fy(x0,y0,z0)**2 + fz(x0,y0,z0)**2)*fz(x0,y0,z0)
        err = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
        x0 = x1
        y0 = y1
        z0 = z1
        errs.append(err)
        count += 1
        if count > max:
            print("No convergence")
            break
    return x0,y0,z0,count,np.array(errs)

def main():
    # Question 1
    print("Question 1:")
    # Part A
    x,y,iter,errs = q1(f1,g1,1,1,1e-12,1000)
    print("x: " + str(x), "y: " + str(y), "Iterations: " + str(iter))
    index = np.arange(0,iter,1)
    plt.plot(index,errs,'-b',label="Method Provided")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Convergence Comparison")
    # Part C
    x,y,iter,errs = newtons(f1,g1,f1x,f1y,g1x,g1y,1,1,1e-12,1000)
    print("x: " + str(x), "y: " + str(y), "Iterations: " + str(iter))
    index = np.arange(0,iter,1)
    plt.plot(index,errs,'-r',label="Newtons Method")
    plt.legend(loc="upper right")
    plt.show()

    # Question 2
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.abs(X**2 + 2*X*Y + Y**2 + 1)
    Z2 = np.abs(X**2 - 2*X*Y + Y**2 + 1)
    plt.contourf(X, Y, Z, levels=[-np.inf, 2**(1/3)], colors=['skyblue'], alpha=0.7)
    plt.contourf(X, Y, Z2, levels=[-np.inf, 2**(1/3)], colors=['lightcoral'], alpha=0.7)
    plt.contour(X, Y, Z, levels=[2**(1/3)], colors='blue')
    plt.contour(X, Y, Z2, levels=[2**(1/3)], colors='red')
    plt.title('Domain With Guaranteed Fixed Point')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Question 3
    x,y,z,iter,errs = newton_3d(f3,f3x,f3y,f3z,1,1,1,1e-12,1000)
    print("x: " + str(x), "y: " + str(y), "z: " + str(z), "Iterations: " + str(iter))
    print(f3(x,y,z))
    err1 = errs[1:]
    err2 = errs[:-1]
    ind = np.arange(0,iter,1)
    plt.plot(ind,err1/err2**2,'-r')
    plt.title("Error Analysis")
    plt.xlabel("Iteration number, n")
    plt.ylabel(r"$\frac{Error_{n+1}}{Error_n^2}$")
    plt.show()

main()
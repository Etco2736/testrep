import numpy as np
import matplotlib.pyplot as plt

def f1(x,y):
    return 4*x**2+y**2-4
def fx1(x,y):
    return 8*x
def fy1(x,y):
    return 2*y
def g1(x,y):
    return x+y-np.sin(x-y)
def gx1(x,y):
    return 1 - np.cos(x-y)
def gy1(x,y):
    return 1 + np.cos(x-y)

def lazy_newton(f,g,fx,fy,gx,gy,x0,y0,tol,max):
    count = 0
    err = 500
    errs = []
    jacobian = np.zeros((2,2))
    j_inv = np.zeros((2,2))
    jacobian[0][0] = fx(x0,y0)
    jacobian[0][1] = fy(x0,y0)
    jacobian[1][0] = gx(x0,y0)
    jacobian[1][1] = gy(x0,y0)
    j_inv = np.linalg.inv(jacobian)

    while err > tol:
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

def slacker_newton(f,g,fx,fy,gx,gy,x0,y0,tol,max):
    count = 0
    err = 500
    errs = []
    jacobian = np.zeros((2,2))
    j_inv = np.zeros((2,2))
    jacobian[0][0] = fx(x0,y0)
    jacobian[0][1] = fy(x0,y0)
    jacobian[1][0] = gx(x0,y0)
    jacobian[1][1] = gy(x0,y0)
    j_inv = np.linalg.inv(jacobian)
    x_old = x0
    y_old = y0

    while err > tol:
        if np.sqrt((x0-x_old)**2+(y0-y_old)**2) > 0.001:
            print("Hi")
            jacobian[0][0] = fx(x0,y0)
            jacobian[0][1] = fy(x0,y0)
            jacobian[1][0] = gx(x0,y0)
            jacobian[1][1] = gy(x0,y0)
            j_inv = np.linalg.inv(jacobian)
            x_old = x0
            y_old = y0
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

def approximate_newton(f,g,fx,fy,gx,gy,x0,y0,tol,max,h):
    count = 0
    err = 500
    errs = []
    jacobian = np.zeros((2,2))
    j_inv = np.zeros((2,2))
    while err > tol:
        jacobian[0][0] = (f(x0+h,y0)-f(x0-h,y0))/2/h
        jacobian[0][1] = (f(x0,y0+h)-f(x0,y0-h))/2/h
        jacobian[1][0] = (g(x0+h,y0)-g(x0-h,y0))/2/h
        jacobian[1][1] = (g(x0,y0+h)-g(x0,y0-h))/2/h
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
def main():
    # Excersise 3.2
    x,y,count,errs = lazy_newton(f1,g1,fx1,fy1,gx1,gy1,1,0,1e-10,5000)
    x2,y2,count2,errs2 = slacker_newton(f1,g1,fx1,fy1,gx1,gy1,1,0,1e-10,5000)
    print("Lazy:", x,y,count)
    print("Slacker:", x2,y2,count2)

    # Excersise 3.3
    x,y,count,errs = approximate_newton(f1,g1,fx1,fy1,gx1,gy1,1,0,1e-10,5000,1e-7)
    x2,y2,count2,errs2 = approximate_newton(f1,g1,fx1,fy1,gx1,gy1,1,0,1e-10,5000,1e-3)
    print("Low h:", x,y,count)
    print("High h:", x2,y2,count2)
    
main()
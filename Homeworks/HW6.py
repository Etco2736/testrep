import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv 
from numpy.linalg import norm

def f1(x,y):
    return x**2+y**2-4
def f1x(x,y):
    return 2*x
def f1y(x,y):
    return 2*y
def g1(x,y):
    return np.exp(x)+y-1
def g1x(x,y):
    return np.exp(x)
def g1y(x,y):
    return 1

def f2(x,y,z):
    return x+np.cos(x*y*z)-1
def f2x(x,y,z):
    return 1-y*z*np.sin(x*y*z)
def f2y(x,y,z):
    return -x*z*np.sin(x*y*z)
def f2z(x,y,z):
    return -x*y*np.sin(x*y*z)
def g2(x,y,z):
    return (1-x)**(1/4)+y+0.05*z**2-0.15*z-1
def g2x(x,y,z):
    return 1/4*(1-x)**(-3/4)
def g2y(x,y,z):
    return 1
def g2z(x,y,z):
    return 0.1*z-0.15
def r2(x,y,z):
    return -x**2-0.1*y**2+0.01*y+z-1
def r2x(x,y,z):
    return -2*x
def r2y(x,y,z):
    return -0.2*y+0.01
def r2z(x,y,z):
    return 1


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
        if count > max or x0 > 300 or y0 > 300:
            print("Did Not Converge")
            break
    return x1,y1,count,errs

def broyden(f,g,fx,fy,gx,gy,x0,y0,tol,max):
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
        s = np.array([x1-x0,y1-y0])
        y = np.array([f(x1,y1)-f(x0,y0),g(x1,y1)-g(x0,y0)])
        j_inv = j_inv + (y - j_inv @ s) @ np.transpose(s)/(np.transpose(s) @ s)
        x0 = x1
        y0 = y1
        count = count + 1
        errs.append(err)
        if count > max or x0 > 300 or y0 > 300:
            print("Did Not Converge")
            break
    return x1,y1,count,errs

def newtons3d(f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz,x0,y0,z0,tol,max_step):
    count = 0
    err = 500
    errs = []
    jacobian = np.zeros((3,3))
    j_inv = np.zeros((3,3))
    while err > tol:
        jacobian[0][0] = fx(x0,y0,z0)
        jacobian[0][1] = fy(x0,y0,z0)
        jacobian[0][2] = fz(x0,y0,z0)
        jacobian[1][0] = gx(x0,y0,z0)
        jacobian[1][1] = gy(x0,y0,z0)
        jacobian[1][2] = gz(x0,y0,z0)
        jacobian[2][0] = rx(x0,y0,z0)
        jacobian[2][1] = ry(x0,y0,z0)
        jacobian[2][2] = rz(x0,y0,z0)
        j_inv = np.linalg.inv(jacobian)
        x1 = x0 - (j_inv[0][0]*f(x0,y0,z0)+j_inv[0][1]*g(x0,y0,z0)+j_inv[0][2]*r(x0,y0,z0))
        y1 = y0 - (j_inv[1][0]*f(x0,y0,z0)+j_inv[1][1]*g(x0,y0,z0)+j_inv[1][2]*r(x0,y0,z0))
        z1 = z0 - (j_inv[2][0]*f(x0,y0,z0)+j_inv[2][1]*g(x0,y0,z0)+j_inv[2][2]*r(x0,y0,z0))
        err = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        x0 = x1
        y0 = y1
        z0 = z1
        count = count + 1
        errs.append(err)
        if count > max_step:
            print("Did Not Converge")
            break
    return x1,y1,z1,count,errs

def evalF(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz):
    F = np.zeros(3)
    F[0] = f(x[0],x[1],x[2])
    F[1] = g(x[0],x[1],x[2])
    F[2] = r(x[0],x[1],x[2])
    return F

def evalJ(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz): 
    J = np.array([[fx(x[0],x[1],x[2]),fy(x[0],x[1],x[2]),fz(x[0],x[1],x[2])],
          [gx(x[0],x[1],x[2]),gy(x[0],x[1],x[2]),gz(x[0],x[1],x[2])],
          [rx(x[0],x[1],x[2]),ry(x[0],x[1],x[2]),rz(x[0],x[1],x[2])]])
    return J

def evalg(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz):
    F = evalF(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz):
    F = evalF(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)
    J = evalJ(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)
    gradg = np.transpose(J).dot(F)
    return gradg

def SteepestDescent(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)
        z = eval_gradg(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier,its]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier,its]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier,Nmax]

def first_steepest(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz,tol,tol2,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)
        z = eval_gradg(x,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            o1,o2,o3,count,errs = newtons3d(f, g, r, fx, fy, fz, gx, gy, gz, rx, ry, rz, x[0], x[1], x[2], tol2, Nmax)
            ret = np.array([o1,o2,o3])
            return ret,g1,0,count
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec,f,g,r,fx,fy,fz,gx,gy,gz,rx,ry,rz)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            o1,o2,o3,count,errs = newtons3d(f, g, r, fx, fy, fz, gx, gy, gz, rx, ry, rz, x[0], x[1], x[2], tol2, Nmax)
            ret = np.array([o1,o2,o3])
            return ret,g1,0,count

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier,Nmax]

def main():
    # Question 1
    tol = 1e-10
    max_step = 500
    x0 = 1
    y0 = 1
    x,y,iters,err = newtons(f1, g1, f1x, f1y, g1x, g1y, x0, y0, tol, max_step)
    print("Newtons:", x, y, iters)
    x,y,iters,err = lazy_newton(f1, g1, f1x, f1y, g1x, g1y, x0, y0, tol, max_step)
    print("Lazy Newton:", x, y, iters)
    x,y,iters,err = broyden(f1, g1, f1x, f1y, g1x, g1y, x0, y0, tol, max_step)
    print("Broyden:", x, y, iters)
    x0 = 1
    y0 = -1
    x,y,iters,err = newtons(f1, g1, f1x, f1y, g1x, g1y, x0, y0, tol, max_step)
    print("Newtons:", x, y, iters)
    x,y,iters,err = lazy_newton(f1, g1, f1x, f1y, g1x, g1y, x0, y0, tol, max_step)
    print("Lazy Newton:", x, y, iters)
    x,y,iters,err = broyden(f1, g1, f1x, f1y, g1x, g1y, x0, y0, tol, max_step)
    print("Broyden:", x, y, iters)
    x0 = 0
    y0 = 0
    # x,y,iters,err = newtons(f1, g1, f1x, f1y, g1x, g1y, x0, y0, tol, max_step)
    # print("Newtons:", x, y, iters)
    # x,y,iters,err = lazy_newton(f1, g1, f1x, f1y, g1x, g1y, x0, y0, tol, max_step)
    # print("Lazy Newton:", x, y, iters)
    # x,y,iters,err = broyden(f1, g1, f1x, f1y, g1x, g1y, x0, y0, tol, max_step)
    # print("Broyden:", x, y, iters)

    # Question 2
    tol = 1e-8
    x0 = 0.5
    y0 = 0.5
    z0 = 0.5
    x,y,z,iters,err = newtons3d(f2, g2, r2, f2x, f2y, f2z, g2x, g2y, g2z, r2x, r2y, r2z, x0, y0, z0, tol, max_step)
    print("Newtons:", x, y, z, iters)
    guess = np.array([x0,y0,z0])
    output,thing,flag,iters = SteepestDescent(guess, f2, g2, r2, f2x, f2y, f2z, g2x, g2y, g2z, r2x, r2y, r2z, tol, max_step)
    print("Steepest:",output[0],output[1],output[2],iters)
    tol2 = 5e-2
    output,thing,flag,iters = first_steepest(guess, f2, g2, r2, f2x, f2y, f2z, g2x, g2y, g2z, r2x, r2y, r2z, tol2, tol, max_step)
    print("First Steepest:",output[0],output[1],output[2],iters)

main()
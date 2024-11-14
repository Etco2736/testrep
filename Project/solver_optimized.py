import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse.linalg import bicgstab, gmres, minres
from scipy.sparse import csr_matrix
from scipy.optimize import newton_krylov
import time as timer

def build_sources(num_sources,source_size,sources,c0):
    # Currently just hardcoded surfaces...
    for i in range(num_sources):
        c = c0[i]
        if i == 0:
            sources[i][0] = -0.33
            sources[i][1] = 0
            sources[i][2] = 0

            sources[i][3] = c
            sources[i][4] = 0
            sources[i][5] = 0
            sources[i][6] = c
        if i == 1:
            sources[i][0] = 0.33
            sources[i][1] = 0
            sources[i][2] = 1

            sources[i][3] = 0
            sources[i][4] = 2*c
            sources[i][5] = c
            sources[i][6] = 0

def setup_source(sources,nx,ny,dx,dy,num_source,source_size):
    source_cell = []
    mask = np.ones(nx*ny)

    for i in range(nx):
        x = -1 + i*dx
        for j in range(ny):
            y = -1 + j*dy
            for k in range(num_source):
                x_source = sources[k][0]
                y_source = sources[k][1]
                dist = np.sqrt((x_source-x)**2 + (y_source-y)**2)
                if dist < source_size:
                        source_cell.append((i,j,k))
                        mask[i+j*nx] = 0

    return source_cell,mask

def build_boundary(grid, nx, ny, sc1, sources, num_sources):
    size_grid = nx*ny
    size = 4*size_grid
    sc_set = set(sc1)
    for i in range(size):
        conc_type = i // size_grid
        indx = (i % nx) 
        indy = (i // nx) % ny
        for k in range(num_sources):
            if (indx,indy,k) in sc_set:
                grid[i] = sources[k][3+conc_type]
    return grid

def clear_sources(n,sources,num_sources,source_size,step):
    for k in range(num_sources):
        x_s = sources[k][0]
        y_s = sources[k][1]

        x_diff = n[:,0,step] - x_s
        y_diff = n[:,1,step] - y_s
        r = np.sqrt(x_diff**2 + y_diff**2)
        collided = np.where(r < source_size)[0]
        alpha = source_size**2 / r**2
        x_diff_new = x_diff[collided]*np.sqrt(alpha[collided])
        y_diff_new = y_diff[collided]*np.sqrt(alpha[collided])
        n[collided,0,step] = x_s + x_diff_new
        n[collided,1,step] = y_s + y_diff_new
    return n

def build_particles(sources,num_sources,num_particles,source_size,num_steps):
    n = np.zeros((num_particles,2,int(num_steps)))
    n[:,0,0] = np.random.uniform(-1,1,num_particles)
    n[:,1,0] = np.random.uniform(-1,1,num_particles)
    n = clear_sources(n,sources,num_sources,source_size,0)
    return n

def plot_initial(grid,nx,ny,val):
    plot_test = grid.reshape((4,nx,ny))
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    cax = ax.imshow(plot_test[0], cmap='jet',extent=[-1,1,-1,1],origin='lower')
    cax2 = ax2.imshow(plot_test[1], cmap='jet',extent=[-1,1,-1,1],origin='lower')
    cax3 = ax3.imshow(plot_test[2], cmap='jet',extent=[-1,1,-1,1],origin='lower')
    cax4 = ax4.imshow(plot_test[3], cmap='jet',extent=[-1,1,-1,1],origin='lower')
    if val == 0:
        ax.set_title(r"$[H^+]$")
        ax2.set_title(r"$[OH^-]$")
        ax3.set_title(r"$[Ca^+]$")
        ax4.set_title(r"[Benzoate]")
        name = 'C'
    elif val == 1:
        ax.set_title(r"$N_x \rightarrow H^+$")
        ax2.set_title(r"$N_x \rightarrow OH^-$")
        ax3.set_title(r"$N_x \rightarrow Ca^+$")
        ax4.set_title(r"N_x \rightarrow Benzoate")
        name = r"$N_x$"
    elif val == 2:
        ax.set_title(r"$N_y \rightarrow H^+$")
        ax2.set_title(r"$N_y \rightarrow OH^-$")
        ax3.set_title(r"$N_y \rightarrow Ca^+$")
        ax4.set_title(r"N_y \rightarrow Benzoate")
        name = r"$N_y$"

    fig.colorbar(cax, ax=ax, orientation='vertical', label=name)
    fig.colorbar(cax2, ax=ax2, orientation='vertical', label=name)
    fig.colorbar(cax3, ax=ax3, orientation='vertical', label=name)
    fig.colorbar(cax4, ax=ax4, orientation='vertical', label=name)
    plt.show()

def build_reaction(grid,k,nx,ny):
    size = 4*nx*ny
    size2 = size // 4
    r = np.zeros(size)

    c1 = np.array(grid[:size2])
    c2 = np.array(grid[size2:2*size2])
    val = -k*c1*c2
    r[:size2] = val
    r[size2:2*size2] = val

    return r

def build_derivative(c,nx,ny,dx,dy,num_sources,sc,mask,dir):
    size = 4*nx*ny
    size2 = size//4
    sc_set = set(sc)
    dy2 = 2*dy
    dx2 = 2*dx
    c_new = c.reshape(4,ny,nx)
    
    if dir == 0:
        c1 = np.zeros((4,ny))
        c2 = np.zeros((4,ny))
        dc = np.zeros((4,ny,nx))
        for i in range(nx):
            if i == 0:
                c1 = c_new[:,:,0]
                c2 = c_new[:,:,1]
                dc[:,:,i] = (c2 - c1)/dx
            elif i == nx-1:
                c1 = c_new[:,:,nx-2]
                c2 = c_new[:,:,nx-1]
                dc[:,:,i] = (c2 - c1)/dx
            else:
                c1 = c_new[:,:,i-1]
                c2 = c_new[:,:,i+1]
                dc[:,:,i] = (c2 - c1)/dx2
        dc = dc.reshape(4*nx*ny)

    if dir == 1:
        dc = np.zeros((4,ny,nx))
        c1 = np.zeros((4,nx))
        c2 = np.zeros((4,nx))
        for i in range(ny):
            if i == 0:
                c1 = c_new[:,0,:]
                c2 = c_new[:,1,:]
                dc[:,i,:] = (c2 - c1)/dy
            elif i == nx-1:
                c1 = c_new[:,ny-2,:]
                c2 = c_new[:,ny-1,:]
                dc[:,i,:] = (c2 - c1)/dy
            else:
                c1 = c_new[:,i-1,:]
                c2 = c_new[:,i+1,:]
                dc[:,i,:] = (c2 - c1)/dy2

        dc = dc.reshape(4*nx*ny)
        dc[0:size2] = dc[0:size2]*mask
        dc[size2:2*size2] = dc[size2:2*size2]*mask
        dc[2*size2:3*size2] = dc[2*size2:3*size2]*mask
        dc[3*size2:4*size2] = dc[3*size2:4*size2]*mask

    return dc

def build_flux(c,dc,D,z,nx,ny,num_sources,sc,mask,dir):
    size = nx * ny * 4
    size2 = nx * ny
    n = np.zeros(size)

    c1 = c[:size2]
    c2 = c[size2:size2*2]
    c3 = c[2*size2:3*size2]
    c4 = c[3*size2:4*size2]    
    dc1 = dc[:size2]
    dc2 = dc[size2:size2*2]
    dc3 = dc[2*size2:3*size2]
    dc4 = dc[3*size2:4*size2]

    bottom = z[0]**2*D[0]*c1 + z[1]**2*D[1]*c2 + z[2]**2*D[2]*c3 + z[3]**2*D[3]*c4
    top = z[0]*D[0]*dc1 + z[1]*D[1]*dc2 + z[2]*D[2]*dc3 + z[3]*D[3]*dc4

    n[:size2] = np.where(bottom < 1e-5, -D[0]*dc1, -D[0]*dc1 + np.divide(z[0]*D[0]*c1*top, bottom, where=bottom > 1e-5))
    n[size2:2*size2] = np.where(bottom < 1e-5, -D[1]*dc2, -D[1]*dc2 + np.divide(z[1]*D[1]*c2*top, bottom, where=bottom > 1e-5))
    n[2*size2:3*size2] = np.where(bottom < 1e-5, -D[2]*dc3, -D[2]*dc3 + np.divide(z[2]*D[2]*c3*top, bottom, where=bottom > 1e-5))
    n[3*size2:4*size2] = np.where(bottom < 1e-5, -D[3]*dc4, -D[3]*dc4 + np.divide(z[3]*D[3]*c4*top, bottom, where=bottom > 1e-5))

    edgemask = np.ones((nx,ny))
    if dir == 1:
        edgemask[0,:] = 0
        edgemask[ny-1,:] = 0
    if dir == 0:
        edgemask[:,0] = 0
        edgemask[:,nx-1] = 0

    new_edgemask = edgemask.reshape(nx*ny)
    n[:size2] = n[:size2] * new_edgemask
    n[size2:2*size2] = n[size2:2*size2] * new_edgemask
    n[2*size2:3*size2] = n[2*size2:3*size2] * new_edgemask
    n[3*size2:4*size2] = n[3*size2:4*size2] * new_edgemask

    return n     

def build_c_matrix(c,nx,ny,dx,dy,Nx,Ny,sc,mask,num_sources,z,D):
    sc_set = set(sc)
    sumx, sumy, sum_left, sum_right, sum_up, sum_down = 0, 0, 0, 0, 0, 0

    size = 4*nx*ny
    size2 = nx*ny
    size2_2 = 2*size2
    size2_3 = 3*size2

    inv_dx_sq = 1/dx**2
    inv_dy_sq = 1/dy**2
    dx2 = 2*dx
    dy2 = 2*dy

    max_elements = 20 * size2
    data = np.zeros(max_elements, dtype=np.float64)
    rows = np.zeros(max_elements, dtype=np.int32)
    cols = np.zeros(max_elements, dtype=np.int32)
    idx = 0

    def add_entry(i, j, value):
        nonlocal idx
        data[idx] = value
        rows[idx] = i
        cols[idx] = j
        idx += 1

    for i in range(size2):
        indx = i % nx 
        indy = i // nx

        check = False
        for k in range(num_sources):
            if(indx,indy,k) in sc_set:
                check = True       
        if check == True:
            continue
        
        bottom = z[0]**2*c[i] + z[1]**2*c[i+size2] + z[2]**2*c[i+size2_2] + z[3]**2*c[i+size2_3]
        if bottom < 1e-5:
            sumx = 0
            sumy = 0
        else:
            sumx = (z[0]*Nx[i]/D[0] + z[1]*Nx[i+size2]/D[1] + z[2]*Nx[i+size2_2]/D[2] + z[3]*Nx[i+size2_3]/D[3])/bottom
            sumy = (z[0]*Ny[i]/D[0] + z[1]*Ny[i+size2]/D[1] + z[2]*Ny[i+size2_2]/D[2] + z[3]*Ny[i+size2_3]/D[3])/bottom

        if indx != 0:
            bottom_left = z[0]**2*c[i-1] + z[1]**2*c[i+size2-1] + z[2]**2*c[i+size2_2-1] + z[3]**2*c[i+size2_3-1]
            if bottom_left < 1e-5:
                sum_left = 0
            else:
                sum_left = (z[0]*Nx[i-1]/D[0] + z[1]*Nx[i+size2-1]/D[1] + z[2]*Nx[i+size2_2-1]/D[2] + z[3]*Nx[i+size2_3-1]/D[3])/bottom_left

        if indx != nx-1:
            bottom_right = z[0]**2*c[i+1] + z[1]**2*c[i+size2+1] + z[2]**2*c[i+size2_2+1] + z[3]**2*c[i+size2_3+1]
            if bottom_right < 1e-5:
                sum_right = 0
            else:
                sum_right = (z[0]*Nx[i+1]/D[0] + z[1]*Nx[i+size2+1]/D[1] + z[2]*Nx[i+size2_2+1]/D[2] + z[3]*Nx[i+size2_3+1]/D[3])/bottom_right

        if indy != ny-1:
            bottom_up = z[0]**2*c[i+nx] + z[1]**2*c[i+size2+nx] + z[2]**2*c[i+size2_2+nx] + z[3]**2*c[i+size2_3+nx]
            if bottom_up < 1e-5:
                sum_up = 0
            else:
                sum_up = (z[0]*Ny[i+nx]/D[0] + z[1]*Ny[i+size2+nx]/D[1] + z[2]*Ny[i+size2_2+nx]/D[2] + z[3]*Ny[i+size2_3+nx]/D[3])/bottom_up
        
        if indy != 0:
            bottom_down = z[0]**2*c[i-nx] + z[1]**2*c[i+size2-nx] + z[2]**2*c[i+size2_2-nx] + z[3]**2*c[i+size2_3-nx]
            if bottom_down < 1e-5:
                sum_down = 0
            else:
                sum_down = (z[0]*Ny[i-nx]/D[0] + z[1]*Ny[i+size2-nx]/D[1] + z[2]*Ny[i+size2_2-nx]/D[2] + z[3]*Ny[i+size2_3-nx]/D[3])/bottom_down 

        center = 0
        if indx == 0:
            #continue
            center += (sum_right-sumx)/dx
        elif indx == nx-1:
            #continue
            center += (sumx-sum_left)/dx
        else:
            center += (sum_right-sum_left)/dx2
        if indy == 0:
            #continue
            center += (sum_up-sumy)/dy
        elif indy == ny-1:
            #continue
            center += (sumy-sum_down)/dy
        else:
            center += (sum_up-sum_down)/dy2

        if indx == 0:
            center += -sumx/dx
            right1 = (sumx/dx*z[0] - inv_dx_sq)*D[0]
            right2 = (sumx/dx*z[1] - inv_dx_sq)*D[1]
            right3 = (sumx/dx*z[2] - inv_dx_sq)*D[2]
            right4 = (sumx/dx*z[3] - inv_dx_sq)*D[3]
            add_entry(i,i+1,right1)
            add_entry(i+size2,i+size2+1,right2)
            add_entry(i+size2_2,i+size2_2+1,right3)
            add_entry(i+size2_3,i+size2_3+1,right4) 
        elif indx == nx-1:
            center += sumx/dx
            left1 = (-sumx/dx*z[0] - inv_dx_sq)*D[0]
            left2 = (-sumx/dx*z[1] - inv_dx_sq)*D[1]
            left3 = (-sumx/dx*z[2] - inv_dx_sq)*D[2]
            left4 = (-sumx/dx*z[3] - inv_dx_sq)*D[3]
            add_entry(i,i-1,left1)
            add_entry(i+size2,i+size2-1,left2)
            add_entry(i+size2_2,i+size2_2-1,left3)
            add_entry(i+size2_3,i+size2_3-1,left4) 
        else:
            left1 = (-sumx/dx2*z[0] - inv_dx_sq)*D[0]
            left2 = (-sumx/dx2*z[1] - inv_dx_sq)*D[1]
            left3 = (-sumx/dx2*z[2] - inv_dx_sq)*D[2]
            left4 = (-sumx/dx2*z[3] - inv_dx_sq)*D[3]
            right1 = (sumx/dx2*z[0] - inv_dx_sq)*D[0]
            right2 = (sumx/dx2*z[1] - inv_dx_sq)*D[1]
            right3 = (sumx/dx2*z[2] - inv_dx_sq)*D[2]
            right4 = (sumx/dx2*z[3] - inv_dx_sq)*D[3]
            add_entry(i,i-1,left1)
            add_entry(i+size2,i+size2-1,left2)
            add_entry(i+size2_2,i+size2_2-1,left3)
            add_entry(i+size2_3,i+size2_3-1,left4)
            add_entry(i,i+1,right1)
            add_entry(i+size2,i+size2+1,right2)
            add_entry(i+size2_2,i+size2_2+1,right3)
            add_entry(i+size2_3,i+size2_3+1,right4) 

        if indy == 0:
            center += -sumy/dy
            up1 = (sumy/dy*z[0] - inv_dy_sq)*D[0]
            up2 = (sumy/dy*z[1] - inv_dy_sq)*D[1]
            up3 = (sumy/dy*z[2] - inv_dy_sq)*D[2]
            up4 = (sumy/dy*z[3] - inv_dy_sq)*D[3]
            add_entry(i,i+nx,up1)
            add_entry(i+size2,i+size2+nx,up2)
            add_entry(i+size2_2,i+size2_2+nx,up3)
            add_entry(i+size2_3,i+size2_3+nx,up4) 
        elif indy == ny-1:
            center += sumy/dy
            down1 = (-sumy/dy*z[0] - inv_dy_sq)*D[0]
            down2 = (-sumy/dy*z[1] - inv_dy_sq)*D[1]
            down3 = (-sumy/dy*z[2] - inv_dy_sq)*D[2]
            down4 = (-sumy/dy*z[3] - inv_dy_sq)*D[3]
            add_entry(i,i-nx,down1)
            add_entry(i+size2,i+size2-nx,down2)
            add_entry(i+size2_2,i+size2_2-nx,down3)
            add_entry(i+size2_3,i+size2_3-nx,down4) 
        else:
            down1 = (-sumy/dy2*z[0] - inv_dy_sq)*D[0]
            down2 = (-sumy/dy2*z[1] - inv_dy_sq)*D[1]
            down3 = (-sumy/dy2*z[2] - inv_dy_sq)*D[2]
            down4 = (-sumy/dy2*z[3] - inv_dy_sq)*D[3]
            up1 = (sumy/dy2*z[0] - inv_dy_sq)*D[0]
            up2 = (sumy/dy2*z[1] - inv_dy_sq)*D[1]
            up3 = (sumy/dy2*z[2] - inv_dy_sq)*D[2]
            up4 = (sumy/dy2*z[3] - inv_dy_sq)*D[3]
            add_entry(i,i-nx,down1)
            add_entry(i+size2,i+size2-nx,down2)
            add_entry(i+size2_2,i+size2_2-nx,down3)
            add_entry(i+size2_3,i+size2_3-nx,down4)
            add_entry(i,i+nx,up1)
            add_entry(i+size2,i+size2+nx,up2)
            add_entry(i+size2_2,i+size2_2+nx,up3)
            add_entry(i+size2_3,i+size2_3+nx,up4)           

        center1 = center*D[0]*z[0] + D[0]*(2*inv_dx_sq + 2*inv_dy_sq)
        center2 = center*D[1]*z[1] + D[1]*(2*inv_dx_sq + 2*inv_dy_sq)
        center3 = center*D[2]*z[2] + D[2]*(2*inv_dx_sq + 2*inv_dy_sq)
        center4 = center*D[3]*z[3] + D[3]*(2*inv_dx_sq + 2*inv_dy_sq)
        add_entry(i,i,center1)
        add_entry(i+size2,i+size2,center2)
        add_entry(i+size2_2,i+size2_2,center3)
        add_entry(i+size2_3,i+size2_3,center4)

    matrix = csr_matrix((data[:idx], (rows[:idx], cols[:idx])), shape=(size, size))
    return matrix 

    size = nx*ny*4
    for i in range(size):
        indx = i % nx 
        indy = (i // nx) % ny
        if dir == 0:
            if indx == 0 or indx == nx-1:
                x[i] = 0
        if dir == 1:
            if indy == 0 or indy == ny-1:
                x[i] = 0
    return x       

def build_udp(c,dcdx,dcdy,psi_D,D,z,nx,ny):
    size = nx*ny
    c1 = c[0:size]
    c2 = c[size:2*size]
    c3 = c[2*size:3*size]
    c4 = c[3*size:]
    dc1dx = dcdx[0:size]
    dc2dx = dcdx[size:2*size]
    dc3dx = dcdx[2*size:3*size]
    dc4dx = dcdx[3*size:]
    dc1dy = dcdy[0:size]
    dc2dy = dcdy[size:2*size]
    dc3dy = dcdy[2*size:3*size]
    dc4dy = dcdy[3*size:]

    udpx = np.zeros(size)
    udpy = np.zeros(size)

    for i in range(size):
        indx = i % nx
        indy = i // ny

        bottom1 = D[0]*z[0]**2*c1[i] + D[1]*z[1]**2*c2[i] + D[2]*z[2]**2*c3[i] + D[3]*z[3]**2*c4[i]
        bottom2 = z[0]**2*c1[i] + z[1]**2*c2[i] + z[2]**2*c3[i] + z[3]**2*c4[i]
        top1x = D[0]*z[0]*dc1dx[i] + D[1]*z[1]*dc2dx[i] + D[2]*z[2]*dc3dx[i] + D[3]*z[3]*dc4dx[i]
        top1y = D[0]*z[0]*dc1dy[i] + D[1]*z[1]*dc2dy[i] + D[2]*z[2]*dc3dy[i] + D[3]*z[3]*dc4dy[i]
        top2x = z[0]**2*dc1dx[i] + z[1]**2*dc2dx[i] + z[2]**2*dc3dx[i] + z[3]**2*dc4dx[i]
        top2y = z[0]**2*dc1dy[i] + z[1]**2*dc2dy[i] + z[2]**2*dc3dy[i] + z[3]**2*dc4dy[i]

        if bottom1 > 1e-5 and bottom2 > 1e-5:
            udpx[i] = psi_D*top1x/bottom1 + psi_D**2/8*top2x/bottom2
            udpy[i] = psi_D*top1y/bottom1 + psi_D**2/8*top2y/bottom2
        else:
            udpx[i] = 0
            udpy[i] = 0

        if indx == 0 or indx == nx-1:
            udpx[i] = 0
        if indy == 0 or indy == ny-1:
            udpy[i] = 0

    return udpx,udpy

def update_particles(n,nx,ny,dx,dy,sources,num_sources,source_size,num_particles,Dp,dt,udpx,udpy,step):
    for k in range(num_particles):  
        i = int((n[k,0,step-1]+1) // dx)
        j = int((n[k,1,step-1]+1) // dy)
        if i == nx-1:
            i -= 1
        if j == ny-1:
            j -= 1
        x1 = i * dx - 1
        x2 = x1 + dx
        y1 =  j * dy - 1
        y2 = y1 + dy

        Q11 = udpx[i + nx * j]
        Q21 = udpx[i + 1 + nx * j]
        Q12 = udpx[i + nx * (j + 1)]
        Q22 = udpx[i + 1 + nx * (j + 1)]
        f_xy1 = (x2 - n[k,0,step-1]) / (x2 - x1) * Q11 + (n[k,0,step-1] - x1) / (x2 - x1) * Q21
        f_xy2 = (x2 - n[k,0,step-1]) / (x2 - x1) * Q12 + (n[k,0,step-1] - x1) / (x2 - x1) * Q22
        f_xy = (y2 - n[k,1,step-1]) / (y2 - y1) * f_xy1 + (n[k,1,step-1] - y1) / (y2 - y1) * f_xy2
        n[k,0,step] = n[k,0,step-1] + Dp*dt*f_xy

        Q11 = udpy[i + nx * j]
        Q21 = udpy[i + 1 + nx * j]
        Q12 = udpy[i + nx * (j + 1)]
        Q22 = udpy[i + 1 + nx * (j + 1)]
        f_xy1 = (x2 - n[k,0,step-1]) / (x2 - x1) * Q11 + (n[k,0,step-1] - x1) / (x2 - x1) * Q21
        f_xy2 = (x2 - n[k,0,step-1]) / (x2 - x1) * Q12 + (n[k,0,step-1] - x1) / (x2 - x1) * Q22
        f_xy = (y2 - n[k,1,step-1]) / (y2 - y1) * f_xy1 + (n[k,1,step-1] - y1) / (y2 - y1) * f_xy2
        n[k,1,step] = n[k,1,step-1] + Dp*dt*f_xy

    bad = np.where(n[:,0,step] < -1)[0]
    n[bad,0,step] = -1
    bad = np.where(n[:,0,step] > 1)[0]
    n[bad,0,step] = 1
    bad = np.where(n[:,1,step] < -1)[0]
    n[bad,1,step] = -1
    bad = np.where(n[:,1,step] > 1)[0]
    n[bad,1,step] = 1

    n = clear_sources(n,sources,num_sources,source_size,step)

    return n

def c_boundary(c,nx,ny):
    size = nx*ny*4
    for i in range(size):
        indx = i % nx 
        indy = (i // nx)%ny

        if indx == 0:
            c[i] = c[i+1]
        if indx == nx-1:
            c[i] = c[i-1]
        if indy == 0:
            c[i] = c[i+nx]
        if indy == ny-1:
            c[i] = c[i-nx]
        
    return c

def plot_final(c,n,nx,ny,sources,num_sources,source_size,num_particles):
    def update(frame,c):
        plt.clf()
        c_plot = c[frame].reshape(4,nx,ny)
        ax = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)
        ax.set_aspect("equal")
        ax2.set_aspect("equal")
        ax3.set_aspect("equal")
        ax4.set_aspect("equal")
        cax = ax.imshow(c_plot[0], cmap='jet',extent=[-1,1,-1,1],origin='lower')
        cax2 = ax2.imshow(c_plot[1], cmap='jet',extent=[-1,1,-1,1],origin='lower')
        cax3 = ax3.imshow(c_plot[2], cmap='jet',extent=[-1,1,-1,1],origin='lower')
        cax4 = ax4.imshow(c_plot[3], cmap='jet',extent=[-1,1,-1,1],origin='lower')
        ax.set_title(r"$[H^+]$")
        ax2.set_title(r"$[OH^-]$")
        ax3.set_title(r"$[Ca^+]$")
        ax4.set_title(r"[Benzoate]")
        name = 'C'
        fig.colorbar(cax, ax=ax, orientation='vertical', label=name)
        fig.colorbar(cax2, ax=ax2, orientation='vertical', label=name)
        fig.colorbar(cax3, ax=ax3, orientation='vertical', label=name)
        fig.colorbar(cax4, ax=ax4, orientation='vertical', label=name)

    fig = plt.figure()
    vid = FuncAnimation(fig=fig, func=update, fargs = (c,), frames=c.shape[0])
    vid.save('c_temp.mp4',fps=10)

    cmap = LinearSegmentedColormap.from_list("PurpleYellow", ["purple", "yellow"])
    norm = plt.Normalize(0, n.shape[0]-1)

    def create_gradient_line(x, y, end_idx):
        points = np.array([x[:end_idx], y[:end_idx]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(x[:end_idx])
        lc.set_linewidth(0.5)    
        return lc
    
    fig2 = plt.figure()
    def update2(frame,n):
        plt.clf()
        n_plot = n[:frame+1,:,:]
        ax = fig2.add_subplot(1,1,1)
        ax.set_facecolor('black')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        plt.title("Particle Streaklines")
        
        for i in range(num_particles):
            x = n_plot[:,i,0]
            y = n_plot[:,i,1]
            lc1 = create_gradient_line(x, y, frame)
            ax.add_collection(lc1)

        for i in range(num_sources):
            if sources[i,2] == 0:
                circle = plt.Circle((sources[i,0], sources[i,1]), source_size, color='red', fill=True)
            if sources[i,2] == 1:
                circle = plt.Circle((sources[i,0], sources[i,1]), source_size, color='blue', fill=True)
            ax.add_patch(circle)
        
    vid = FuncAnimation(fig=fig2, func=update2, fargs = (n,), frames=c.shape[0])
    vid.save('n_temp.mp4',fps=10)

def solver(c,n,Nx,Ny,nx,ny,dx,dy,sc,mask,num_sources,k,D,z,sources,source_size,num_particles,psi_D,Dp,dt,realtime,max_time,start_time):
    size = 4*nx*ny
    cm = csr_matrix((size, size))
    c_vals = []
    n_vals = []
    err_tot = 10
    iter = 0
    time = 0
    while time < max_time:
        time = realtime*dt*iter
        print("Iteration: " + str(iter))
        print(f"Time: {time} s")
        if iter != 0 and iter%10 == 0:
            print(f"Elapsed Runtime: {timer.time() - start_time} s")
        err = 1
        c_old = np.copy(c)
        while err > 1e-10:
            print("Hi")
            dcdx = build_derivative(c,nx,ny,dx,dy,num_sources,sc,mask,0)
            #plot_initial(dcdx,nx,ny,1)
            Nx = build_flux(c,dcdx,D,z,nx,ny,num_sources,sc,mask,0)
            #plot_initial(Nx,nx,ny,1)
            dcdy = build_derivative(c,nx,ny,dx,dy,num_sources,sc,mask,1)
            #plot_initial(dcdy,nx,ny,1)
            Ny = build_flux(c,dcdy,D,z,nx,ny,num_sources,sc,mask,1)
            #plot_initial(Ny,nx,ny,1)
            cm = build_c_matrix(c,nx,ny,dx,dy,Nx,Ny,sc,mask,num_sources,z,D)
            r = build_reaction(c,k,nx,ny)
            print("Hello")
            #plot_initial(c,nx,ny,0)
            #plot_initial(r,nx,ny,0)
            def c_sys(x):
                return (cm @ x)*dt - r*dt - c_old + x
            c = newton_krylov(c_sys,c,iter=1)
            print("Bruh")
            err = np.linalg.norm((cm @ c)*dt - r*dt - c_old + c)/size
            err_tot = np.linalg.norm(c - c_old)/size
            print(f"Concentration Error: {err}")
            if np.min(c) < 0:
                print(f"BAD! min: {np.min(c)}")
        dcdx = build_derivative(c,nx,ny,dx,dy,num_sources,sc,0)
        dcdy = build_derivative(c,nx,ny,dx,dy,num_sources,sc,1)
        udp_x, udp_y = build_udp(c,dcdx,dcdy,psi_D,D,z,nx,ny)
        n = update_particles(n,nx,dx,dy,sources,num_sources,source_size,num_particles,Dp,dt,udp_x,udp_y,iter+1)  
        print(f"Iteration error: {err_tot}")
        if iter % 1 == 0:
            n_vals.append(n[:,:,iter])
            c_vals.append(c)
        iter += 1
    return np.array(c_vals), np.array(n_vals)

def main():
    # Build Grid
    nx = 200
    ny = 200
    dx = 2/nx
    dy = 2/ny
    grid = np.zeros(4*nx*ny)
    dt = 0.01
    maxtime = 10

    # Diffusivities
    dh = 9.31e-9
    doh = 5.27e-9
    dca = 0.793e-9
    db = 0.863e-9
    D = [1,doh/dh,dca/dh,db/dh]
    Dp = 1e-1
    z = [1,-1,2,-1]
    psi_D = 1
    realtime = 1e-3*1e-3/dh
    num_steps = np.ceil(maxtime/realtime/dt) + 2

    # Build Sources
    print("Building Sources...")
    num_sources = 2
    source_size = 0.1
    # Row 1 holds x-values, 2 y-values, 3 type (acid vs. base), 4-7 start concentration
    # Species order... [H+], [OH-], [Ca2+], [Benzoic Base]
    c0 = np.array([1,1])
    k = 1e5
    sources = np.zeros((num_sources,7))
    build_sources(num_sources,source_size,sources,c0)
    source_cell,mask = setup_source(sources,nx,ny,dx,dy,num_sources,source_size)

    # Initial Guess
    print("Generating Initial Guess...")
    grid = build_boundary(grid,nx,ny,source_cell,sources,num_sources)
    Nx = np.zeros(4*nx*ny)
    Ny = np.zeros(4*nx*ny)
    num_particles = 1000
    n = build_particles(sources,num_sources,num_particles,source_size,num_steps)

    print("Starting Solver...")
    start_time = timer.time()
    c_vals,n_vals = solver(grid,n,Nx,Ny,nx,ny,dx,dy,source_cell,mask,num_sources,k,D,z,sources,source_size,num_particles,psi_D,Dp,dt,realtime,maxtime,start_time)
    plot_final(c_vals,n_vals,nx,ny,sources,num_sources,source_size,num_particles)

main()
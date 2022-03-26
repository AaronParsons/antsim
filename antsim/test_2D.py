'''An initial 1D test of FDTD.'''

import numpy as np
import tqdm
import time
import matplotlib.pyplot as plt
from scipy.signal.windows import hamming

# Constants
MU0 = 4 * np.pi * 1e-7 # H/m, permeability of free space
C = 2.99792458e8 # m/s
EPS0 = 1 / (MU0 * C**2) # F/m, permittivity of free space
Z0 = np.sqrt(MU0 / EPS0) # Ohms, impedance of free space
COURANT = 1 / np.sqrt(2) # Courant number

SIZE = 100 # size of 2D grid

def run(steps, size=SIZE, plot_every=1):
    '''
    Run a 1D FDTD simulation and plot the results.
    
    Arguments:
        steps: int, number of time steps to run
        size: int, size of the FDTD grid. Default 200
        plot_every: int, step between plot updates
    '''
    # Define Yee lattice, which alternates:
    # E0 H0 E1 H1 ... HN-1 EN HN
    # so that, e.g., H0 can be regarded as being at E0.5
    hx = np.zeros((size,   size-1))
    hy = np.zeros((size-1, size))
    ez = np.zeros((size,   size))

    # Add spatial impedance
    mu_r  = np.ones((size,size))
    eps_r = np.ones((size,size))
    loss_e = np.zeros((size,size)) # XXX relate to sig
    loss_m = np.zeros((size,size)) # XXX relate to sig_m

    x,y = np.indices((size,size))
    
    # inhomogeneities
    eps_r[1*size//4:3*size//4, 1*size//8:3*size//8] *= 2
    #mu_r[1*size//4:3*size//4, 5*size//8:7*size//8] *= 2
    #loss_e[1*size//4:3*size//4, 1*size//8:3*size//8] = 1
    #loss_e[(x-size//4)**2+(y-size//2)**2 < 25] = 1
    #loss_e[::6,10] = 1
    #loss_e[51:65,10] = 1
    #loss_e[36:50,10] = 1
    #loss_m[1*size//4:3*size//4, 1*size//8:3*size//8] = 0.01

    C_eze = np.ones((size,size)) * (1 - loss_e) / (1 + loss_e)
    C_ezh = np.ones((size,size)) * COURANT * Z0 / eps_r / (1 + loss_e)
    C_hxh = np.ones((size,size-1)) * (1 - loss_m[:,:-1]) / (1 + loss_m[:,:-1])
    C_hxe = np.ones((size,size-1)) * COURANT / Z0 / mu_r[:,:-1] / (1 + loss_m[:,:-1])
    C_hyh = np.ones((size-1,size)) * (1 - loss_m[:-1,:]) / (1 + loss_m[:-1,:])
    C_hye = np.ones((size-1,size)) * COURANT / Z0 / mu_r[:-1,:] / (1 + loss_m[:-1,:])

    # absorbing boundary condition
    temp1 = np.sqrt(C_ezh[0,0] * C_hye[0,0])
    temp2 = 1.0 / temp1 + 2.0 + temp1
    c0 = -(1.0 / temp1 - 2.0 + temp1) / temp2
    c1 = -2.0 * (temp1 - 1.0 / temp1) / temp2
    c2 = 4.0 * (temp1 + 1.0 / temp1) / temp2

    # Initialize plotting
    SCALE = .03
    plot_kwargs = {'vmin':-SCALE, 'vmax':SCALE, 'cmap':'RdBu'}
    alpha = 0.2
    cmap = 'gist_yarg'
    plt.ion()
    fig = plt.figure(figsize=(8,4))

    plt.subplot(131)
    Ez_plot = plt.imshow(ez, **plot_kwargs)
    plt.imshow(loss_e, cmap=cmap, alpha=alpha)

    plt.subplot(132)
    Hx_plot = plt.imshow(hx * Z0, **plot_kwargs)
    plt.imshow(loss_m, cmap=cmap, alpha=alpha)

    plt.subplot(133)
    Hy_plot = plt.imshow(hy * Z0, **plot_kwargs)
    plt.imshow(loss_m, cmap=cmap, alpha=alpha)

    src_x = SIZE // 2
    src_y = SIZE // 2

    # define edge before boundary conditions are imposed
    #L_edge, R_edge = 1, -2
    #P_out_r = 0 # holds integrated power off right edge
    #P_out_l = 0 # holds integrated power off left edge

    prev_ez_L = np.zeros((2,3,size))
    prev_ez_R = np.zeros((2,3,size))
    prev_ez_B = np.zeros((2,size,3))
    prev_ez_T = np.zeros((2,size,3))

    # Main simulation loop
    for t in tqdm.tqdm(range(steps)):

        # update H
        hx = C_hxh * hx - C_hxe * np.diff(ez,axis=1)
        hy = C_hyh * hy + C_hye * np.diff(ez,axis=0)

        # update H of additive source
        ppw = 10
        loc = 0
        arg = np.pi * ((COURANT * t - loc) / ppw - 1)
        #hy[src_x,src_y] += (1.0 - 2.0 * arg**2) * np.exp(-arg**2)
        #hy[:,-20] -= (1.0 - 2.0 * arg**2) * np.exp(-arg**2) / Z0
        

        # buffer Ez at edges
        prev_ez_L[1] = prev_ez_L[0]
        prev_ez_L[0] = ez[:3,:]
        prev_ez_R[1] = prev_ez_R[0]
        prev_ez_R[0] = ez[-3:,:]
        prev_ez_B[1] = prev_ez_B[0]
        prev_ez_B[0] = ez[:,:3]
        prev_ez_T[1] = prev_ez_T[0]
        prev_ez_T[0] = ez[:,-3:]

        # update E
        ez[1:-1,1:-1] = C_eze[1:-1,1:-1] * ez[1:-1,1:-1] + \
                        C_ezh[1:-1,1:-1] * (np.diff(hy[:,1:-1],axis=0)
                                            - np.diff(hx[1:-1,:],axis=1))

        # update E of additive source
        #ez[src_x,src_y] += (1.0 - 2.0 * arg**2) * np.exp(-arg**2)
        ez[20:-20,-20] += (1.0 - 2.0 * arg**2) * np.exp(-arg**2) * hamming(size-40)
        
        # absorbing boundary condition
        ez[0,:] = c0 * (ez[2,:] + prev_ez_L[1,0]) \
                + c1 * (prev_ez_L[0,0] + prev_ez_L[0,2] \
                        - ez[1,:] - prev_ez_L[1,1]) \
                + c2 * prev_ez_L[0,1] \
                - prev_ez_L[1,2]
        ez[-1,:] = c0 * (ez[-3,:] + prev_ez_R[1,-1]) \
                 + c1 * (prev_ez_R[0,-1] + prev_ez_R[0,-3] \
                         - ez[-2,:] - prev_ez_R[1,-2]) \
                 + c2 * prev_ez_R[0,-2] \
                 - prev_ez_R[1,-3]
        ez[:,0] = c0 * (ez[:,2] + prev_ez_B[1,:,0]) \
                + c1 * (prev_ez_B[0,:,0] + prev_ez_B[0,:,2] \
                        - ez[:,1] - prev_ez_B[1,:,1]) \
                + c2 * prev_ez_B[0,:,1] \
                - prev_ez_B[1,:,2]
        ez[:,-1] = c0 * (ez[:,-3] + prev_ez_T[1,:,-1]) \
                 + c1 * (prev_ez_T[0,:,-1] + prev_ez_T[0,:,-3] \
                         - ez[:,-2] - prev_ez_T[1,:,-2]) \
                 + c2 * prev_ez_T[0,:,-2] \
                 - prev_ez_T[1,:,-3]

        # calculate quantities we are tracking vs. time
        # fluxes off left and right edges
        #F_out_r = -0.5 * (hy[R_edge] + hy[R_edge-1]) * ez[R_edge] / MU0
        #F_out_l = 0.5 * hy[L_edge] * (ez[L_edge] + ez[L_edge+1]) / MU0
        # accumulated power off left and right edges
        #P_out_r += F_out_r
        #P_out_l += F_out_l

        # update plots
        if t % plot_every == 0:
            Ez_plot.set_data(ez)
            Hx_plot.set_data(hx * Z0)
            Hy_plot.set_data(hy * Z0)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

    # print summary report
    #print(P_out_l, P_out_r, P_out_r+P_out_l)

if __name__ == '__main__':
    run(250, plot_every=1)

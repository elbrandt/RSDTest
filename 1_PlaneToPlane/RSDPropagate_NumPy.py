import numpy as np
from tqdm import tqdm 
import time
import matplotlib.pyplot as plt

def straight_line_forward_example():
    
    lam = 500e-9 # example wavelength of light being simulated (lambda)
    dx = lam/4   # example discretization size, quarter wavelenght to avoid aliasing effects

    # locations of the planes along the optical axis (z-direction)
    zs = [0.00, 0.025, 0.05]

    # widths of the 3 planes perpendicular to the optical axis (x-direction)
    widths = [0.004, 0.2, 0.004]

    # discretize the 3 planes
    xs = [np.arange(-w/2, w/2+1e-12, dx, dtype=np.double) for w in widths]

    # initialize field at point source
    EA=np.zeros(len(xs[0]), dtype=np.cdouble)
    EA[int(len(EA)/2)]=1.0 # center element is point source

    # Test 1: Directly from point A -> plane C
    print("Propagating A->C...")
    EC1, tm = rsd_forward(zs[0], xs[0], EA, zs[2], xs[2], lam, dx)
    print(f"time to propagate A->C: {tm:.3f}s")
    plot_E_field_and_intensity(list(zip(['A', 'C'], [zs[0], zs[2]], [widths[0], widths[2]], [xs[0], xs[2]], [EA, EC1])))
    np.save('EC1.npy', EC1)

    # Test 2: point A -> plane B, then plane B -> plane C
    print("Propagating A->B...")
    EB, tm_ab = rsd_forward(zs[0], xs[0], EA, zs[1], xs[1], lam, dx)
    print(f"time to propagate A->B: {tm_ab:.3f}s")

    print("Propagating B->C...")
    EC2, tm_bc = rsd_forward(zs[1], xs[1], EB, zs[2], xs[2], lam, dx)
    print(f"time to propagate B->C: {tm_bc:.3f}s")

    np.save('EC2.npy', EC2)
    plot_E_field_and_intensity(list(zip(['A', 'B', 'C'], zs, widths, xs, [EA, EB, EC2])))

def rsd_forward(z1, x1, E1, z2, x2, lam, dx, debug=False):
    k = 2 * np.pi / lam # 2pi/lambda term in exponent
    E2 = np.zeros(len(x2), dtype=np.cdouble)
    z_dist = z2-z1
    z_dist_sqr = np.square(z_dist)

    # propagation is a nested for loop (from every point in source to every point in destination)
    # Nesting order of the two loops is arbitrary (both give same ansswer), 
    # but will be faster computationally if the smaller field is in the outer loop
    nnz1 = np.count_nonzero(E1)
    start = time.time()
    if nnz1 < len(x2): # more points in the destination than the source
        for n in tqdm(np.nonzero(E1)):
            x_cur = x1[n];    
            r = np.sqrt(z_dist_sqr + np.square(x2-x_cur))
            E2 = E2 + E1[n] * np.exp(1j*k*r) / r * (z_dist / r)* dx
            if debug:
                break
    else: # more points in the source than the destination
        for n in tqdm(range(len(x2))):
            x_cur = x2[n];    
            r = np.sqrt(z_dist_sqr + np.square(x1-x_cur))
            E2[n] = np.sum(E1 * np.exp(1j*k*r) / r * (z_dist / r) * dx)
    tm = time.time() - start
    return (E2, tm)

def plot_E_field_and_intensity(planes):
    fig, ax = plt.subplots(2,len(planes), figsize=(4.75*len(planes), 4*2))
    for p in range(len(planes)):
        nm, dist, width, x, E = planes[p]        
        En = E / np.abs(E).max() 
        
        plot_intensity(ax[0][p], x, En, f'Intensity at {nm} (z={dist*100}cm, w={width*100}cm)')
        plot_E_field(ax[1][p], x, En, f'E-Field at {nm} (z={dist*100}cm, w={width*100}cm)')
    plt.tight_layout()
    plt.show()

def plot_E_field(ax, x, E, title):
    ax.plot(x*1000,np.real(E), label='Real Part')
    ax.plot(x*1000,np.imag(E), label='Imag Part')
    ax.set_title(title); 
    ax.set_xlabel('x position [millimeters]'); 
    ax.set_ylabel('E-Field'); 
    ax.legend()

def plot_intensity(ax, x, E, title):
    ax.plot(x*1000,np.abs(E)) 
    ax.set_title(title); 
    ax.set_xlabel('x position [millimeters]'); 
    ax.set_ylabel('Intensity')

if __name__=='__main__':
    straight_line_forward_example()
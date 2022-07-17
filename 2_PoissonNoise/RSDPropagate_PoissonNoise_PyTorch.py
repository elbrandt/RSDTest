import numpy as np
from tqdm import tqdm 
import time
import matplotlib.pyplot as plt
import torch

# check if a GPU is available and set a global variable for the device we'll use
if torch.cuda.is_available():
    print("Found GPU, using cuda for calculation.")
    device = torch.device("cuda:0")
else:
    print("No GPU found, using CPU for calculation.")
    device = torch.device("cpu")

def straight_line_forward_example():
    
    lam = 500e-9 # example wavelength of light being simulated (lambda)
    dx = lam/4   # example discretization size, quarter wavelenght to avoid aliasing effects

    # locations of the planes along the optical axis (z-direction)
    zs = [0.00, 0.025, 0.05]

    # widths of the 3 planes perpendicular to the optical axis (x-direction)
    widths = [0.004, 0.2, 0.004]

    # discretize the 3 planes
    xs = [torch.arange(-w/2, w/2+1e-12, dx, dtype=torch.double).to(device) for w in widths]

    # initialize field at point source
    EA=torch.zeros(len(xs[0]), dtype=torch.cdouble).to(device)
    EA[int(len(EA)/2)]=1e9 # center element is point source, 1 billion photons.

    # Test 1: Directly from point A -> plane C
    print("Propagating A->C...")
    EC1, tm = rsd_forward(zs[0], xs[0], EA, zs[2], xs[2], lam, dx)
    print(f"time to propagate A->C: {tm:.3f}s")
    plot(list(zip(['A', 'C'], [zs[0], zs[2]], [widths[0], widths[2]], [xs[0], xs[2]], [[EA], [EC1]])))
    torch.save(EC1, 'EC1.pt')

    # Test 2: point A -> plane B, then plane B -> plane C
    print("Propagating A->B...")
    EB, tm_ab = rsd_forward(zs[0], xs[0], EA, zs[1], xs[1], lam, dx)
    print(f"time to propagate A->B: {tm_ab:.3f}s")

    # add poisson noise to B
    EBn = add_poisson_noise(EB)

    print("Propagating B->C...")
    EC2, tm_bc = rsd_forward(zs[1], xs[1], EB, zs[2], xs[2], lam, dx)
    print(f"time to propagate B->C: {tm_bc:.3f}s")
    print("Propagating Bn->C...")
    EC2n, tm_bc = rsd_forward(zs[1], xs[1], EBn, zs[2], xs[2], lam, dx)
    print(f"time to propagate Bn->C: {tm_bc:.3f}s")

    torch.save(EC2, 'EC2.pt')
    plot(list(zip(['A', 'B', 'C'], zs, widths, xs, [[EA], [EBn, EB], [EC2n, EC2]])))

def rsd_forward(z1, x1, E1, z2, x2, lam, dx, debug=False):
    k = 2 * np.pi / lam # 2pi/lambda term in exponent
    E2 = torch.zeros(len(x2), dtype=torch.cdouble).to(device)
    z_dist = z2-z1
    z_dist_sqr = np.square(z_dist)

    # propagation is a nested for loop (from every point in source to every point in destination)
    # Nesting order of the two loops is arbitrary (both give same ansswer), 
    # but will be faster computationally if the smaller field is in the outer loop
    nnz1 = torch.count_nonzero(E1)
    start = time.time()
    if nnz1 < len(x2): # more points in the destination than the source
        for n in tqdm(torch.nonzero(E1)):
            x_cur = x1[n];    
            r = (z_dist_sqr + (x2-x_cur).pow(2)).sqrt()
            E2 = E2 + E1[n] * (1j*k*r).exp() / r * (z_dist / r)* dx
            if debug:
                break
    else: # more points in the source than the destination
        for n in tqdm(range(len(x2))):
            x_cur = x2[n];    
            r = (z_dist_sqr + (x1-x_cur).pow(2)).sqrt()
            E2[n] = torch.sum(E1 * (1j*k*r).exp() / r * (z_dist / r) * dx)
    tm = time.time() - start
    return (E2, tm)

def add_poisson_noise(E):
    r = E.abs()
    theta = E.angle()
    r = torch.poisson(r)
    ret = torch.polar(r, theta)
    return ret

def plot(planes):
    fig, ax = plt.subplots(1,len(planes), figsize=(4.75*len(planes), 4*2))
    for p in range(len(planes)):
        nm, dist, width, x, E = planes[p]        
        plot_intensity(ax[p], x, E, f'Intensity at {nm} (z={dist*100}cm, w={width*100}cm)')
    plt.tight_layout()
    plt.show()

def plot_intensity(ax, x, E, title):
    if len(E) == 1:
        R = torch.abs(E[0])
        R = R / torch.amax(R)
        ax.plot(x.cpu()*1000,R.cpu())
        ax.legend(['with noise', 'without noise'])
    else:
        Rn = torch.abs(E[0])
        R = torch.abs(E[1])
        Rn = Rn / torch.amax(R) # normalize with respect to the field with no noise
        R = R / torch.amax(R)
        ax.plot(x.cpu()*1000,Rn.cpu())
        ax.plot(x.cpu()*1000,R.cpu())
        ax.legend(['with noise', 'without noise'])
    ax.set_title(title); 
    ax.set_xlabel('x position [millimeters]'); 
    ax.set_ylabel('Intensity')

if __name__=='__main__':
    straight_line_forward_example()
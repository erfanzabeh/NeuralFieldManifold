import numpy as np

def lorenz(T_steps=12000, dt=0.005, sigma=10.0, rho=28.0, beta=8/3, x0=(1.0,1.0,1.0)):
    """
    Lorenz system generator using 4th order Runge-Kutta integration.
    
    Parameters
    ----------
    T_steps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    sigma : float
        Lorenz system parameter.
    rho : float
        Lorenz system parameter.
    beta : float
        Lorenz system parameter.
    x0 : tuple of floats
        Initial conditions (x0, y0, z0).

    Returns
    -------
    tuple of np.ndarray
        Arrays containing the x, y, and z coordinates of the Lorenz system trajectory.

    Example
    -------
    >>> x, y, z = lorenz()
    """
    x, y, z = x0
    xs, ys, zs = [], [], []
    for _ in range(T_steps):
        def f(x,y,z):
            return sigma*(y-x), x*(rho - z) - y, x*y - beta*z
        k1 = f(x,y,z)
        k2 = f(x+0.5*dt*k1[0], y+0.5*dt*k1[1], z+0.5*dt*k1[2])
        k3 = f(x+0.5*dt*k2[0], y+0.5*dt*k2[1], z+0.5*dt*k2[2])
        k4 = f(x+dt*k3[0], y+dt*k3[1], z+dt*k3[2])
        x += (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        y += (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        z += (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        xs.append(x); ys.append(y); zs.append(z)
    return np.array(xs), np.array(ys), np.array(zs)
import numpy as np
from numpy import sin, cos, tan, pi
from numpy.linalg import inv
from matplotlib import pyplot as plt


# variables

m= 2.0
l = 2.0
g = 9.81

delta_t = 0.01
time = np.arange(0.0, 100.0, delta_t)

# functions
def G(y,t):
    
    theta_d, phi_d, = y[0], y[1]
    theta, phi = y[2], y[3]

    theta_dd = phi_d**2 * cos(theta) * sin(theta) - g/l * sin(theta)
    phi_dd = -2.0 * theta_d * phi_d / tan(theta)

    return np.array([theta_dd, phi_dd, theta_d, phi_d])

def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y+0.5*k1*dt, t+0.5*dt)
    k3 = G(y+0.5*k2*dt, t+0.5*dt)
    k4 = G(y+k3*dt, t+dt)
    
    #return dt * G(y, t)
    return dt * (k1 + 2*k2 + 2*k3 + k4)/6


# initial state

y = np.array([0,1.0,1.0,0])

Y1 = []
Y2 = []

# time-stepping solution
for t in time:
    y = y + RK4_step(y, t, delta_t)
    Y1.append(y[2])
    Y2.append(y[3])

# plot the result
plt.plot(time,Y1)
plt.plot(time,Y2)
plt.grid(True)
plt.legend(['\u03F4', '\u03A6'], loc='lower right')
plt.show()


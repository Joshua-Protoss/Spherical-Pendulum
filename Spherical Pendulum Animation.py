import pygame
import sys
import numpy as np
from numpy import sin, cos, tan, pi
from numpy.linalg import inv
from pygame.locals import *

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

def update(theta, phi):
    x = scale * l * sin(theta) * cos(phi) + offset[0]
    y = scale * l * cos(theta) + offset[1]
    z =  scale * l * sin(theta) * sin(phi)

    return (int(x), int(y), int(z))

def render(point):
    x, y, z = point[0], point[1], point[2]
    z_scale = (2 - z/(scale*l)) * 10.0
    


    if prev_point:
        #xp, yp = prev_point[0], prev_point[1]
        pygame.draw.line(trace, LT_BLUE, prev_point, (x,y), int(z_scale*0.4))
    
    screen.fill(WHITE)
    if is_tracing:
        screen.blit(trace, (0,0))
    
    pygame.draw.line(screen, BLACK, offset, (x,y), int(z_scale*0.3))
    pygame.draw.circle(screen, BLACK, offset, 10)
    pygame.draw.circle(screen, RED, (x,y), int(m*z_scale))
    
    return (x, y)

# setup
w, h = 1024, 768
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
LT_BLUE = (230,230,255)
offset = (800,150)
scale = 100
is_tracing = True

#screen = pygame.display.set_mode((w,h))
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
screen.fill(WHITE)
trace = screen.copy()
pygame.display.update()
clock = pygame.time.Clock()

# parameters
m = 2.0
l = 4.5
g = 9.81

prev_point = None
t = 0.0
delta_t = 0.02
y = np.array([0.0, 2.0, 1.3, 0.0])

pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 38)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_t:
                is_tracing = not(is_tracing)
            if event.key == K_c:
                trace.fill(WHITE)
            
    point = update(y[2],y[3])
    prev_point = render(point)

    time_string = 'Time: {} seconds'.format(round(t,1))
    text = myfont.render(time_string, False, (0,0,0))
    screen.blit(text,(10,10))

    t += delta_t
    y = y + RK4_step(y, t, delta_t)
    
    clock.tick(60)
    pygame.display.update()

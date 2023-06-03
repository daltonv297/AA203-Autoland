import numpy as np
from scipy.integrate import solve_ivp
from dynamics import f
import matplotlib.pyplot as plt



s0 = np.array([
    150, # u0, m/s
    0, # w0, m/s
    0, # q0, rad/s
    0, # theta0, rad
    -1500, # x0, m
    -1000, # z0, m
])

T_max = 1000e3 # N

u = np.array([0.3 * T_max, -20])

t_span = (0, 50)
dt = 0.01
t = np.arange(t_span[0], t_span[1], dt)


soln = solve_ivp(f, t_span, s0, method='RK45', t_eval=t, args=[u])

t_out = soln.t
s_out = soln.y

u, w, q, theta, x, z = s_out
h = -z
V = np.sqrt(u*u + w*w)
alpha = np.degrees(np.arctan(w/u))

fig, axs = plt.subplots(4, 1)
axs[0].plot(x, h)
axs[1].plot(x, V)
axs[2].plot(x, alpha)
axs[3].plot(x, theta)
plt.show()

import numpy as np
from scipy.integrate import solve_ivp
from dynamics import f
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial

# @partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    A, B = jax.jacrev(f, (0,1))(s, u)
    c = f(s, u) - A @ s - B @ u
    return A, B, c

def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator


dt = 0.1

f_no_time = lambda s, u : f(None, s, u)
# Initialize the discrete-time dynamics
fd = discretize(f_no_time, dt)

s0 = np.array([
    150, # u0, m/s
    5, # w0, m/s
    0, # q0, rad/s
    0.05, # theta0, rad
    -1500, # x0, m
    -1000, # z0, m
])

Thr_max = 1000e3 # N
n = 6
m = 2
T = 10
t = np.arange(0., T + dt, dt)
N = t.size - 1

u = np.full((N, m), [0.0*Thr_max, 0.0])
s = np.zeros((N + 1, n))
s[0] = s0
for k in range(N):
    s[k+1] = fd(s[k], u[k])

A, B, c = affinize(fd, s[:-1], u)
A, B, c = np.array(A), np.array(B), np.array(c)

s_linear = np.zeros_like(s)
u_linear = np.full((N, m), [0.0*Thr_max, -0.5])
s_linear[0] = s0
for k in range(N):
    s_linear[k+1] = A[k] @ s_linear[k] + B[k] @ u_linear[k] + c[k]

# u = np.array([0.0 * T_max, -50])

# t_span = (0, 10)
# t = np.arange(t_span[0], t_span[1], dt)


# soln = solve_ivp(f, t_span, s0, method='RK45', t_eval=t, args=[u])

# t_out = soln.t
# s_out = soln.y

# u, w, q, theta, x, z = s_out
deviation = np.linalg.norm(s - s_linear, ord=np.inf, axis=1)
s = s_linear
u = s[:, 0]
w = s[:, 1]
q = s[:, 2]
theta = s[:, 3]
x = s[:, 4]
z = s[:, 5]
h = -z
V = np.sqrt(u*u + w*w)
alpha = np.degrees(np.arctan(w/u))
theta_deg = np.degrees(theta)
horiz_var = t


fig, axs = plt.subplots(4, 2, figsize=(12,12))
axs[0,0].plot(horiz_var, h)
axs[0,0].set_ylabel('h')
axs[1,0].plot(horiz_var, V)
axs[1,0].set_ylabel('V')
axs[2,0].plot(horiz_var, alpha)
axs[2,0].set_ylabel(r'$\alpha$')
axs[3,0].plot(horiz_var, theta_deg)
axs[3,0].set_ylabel(r'$\theta$')
axs[0,1].plot(horiz_var, u)
axs[0,1].set_ylabel('u')
axs[1,1].plot(horiz_var, w)
axs[1,1].set_ylabel('w')
axs[2,1].plot(horiz_var, q)
axs[2,1].set_ylabel('q')
axs[3,1].plot(horiz_var, deviation)
axs[3,1].set_ylabel(r'$\rho$')
plt.show()

import numpy as np
from scipy.integrate import solve_ivp
from dynamics import f
from dynamics_simplified import f_simplified
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import cvxpy as cvx
from tqdm import tqdm
from functools import partial

@partial(jax.jit, static_argnums=(0,))
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

def solve_scp(f, s0, s_goal, N, P, Q, R, T_max, delta_e_max, ρ, eps, max_iters):
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize dynamically feasible nominal trajectories
    u = np.zeros((N, m))
    s = np.zeros((N + 1, n))
    s[0] = s0
    for k in range(N):
        s[k+1] = fd(s[k], u[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, J[i + 1] = scp_iteration(f, s0, s_goal, s, u, N,
                                       P, Q, R, T_max, delta_e_max, ρ)
        dJ = np.abs(J[i + 1] - J[i])
        prog_bar.set_postfix({'objective': '{:.5f}'.format(J[i+1]),'objective change': '{:.5f}'.format(dJ)})
        if dJ < eps:
            converged = True
            print('SCP converged after {} iterations.'.format(i))
            break
    if not converged:
        raise RuntimeError('SCP did not converge!')
    J = J[1:i+1]
    return s, u, J


def scp_iteration(f, s0, s_goal, s_prev, u_prev, N, P, Q, R, T_max, delta_e_max, ρ):
    A, B, c = affinize(f, s_prev[:-1], u_prev)
    A, B, c = np.array(A), np.array(B), np.array(c)
    n = Q.shape[0]
    m = R.shape[0]
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    eps = 1e-1

    constraints = []
    cost_terms = []
    for k in range(N):
        cost_terms.append(cvx.quad_form(s_cvx[k] - s_goal, Q))
        cost_terms.append(cvx.quad_form(u_cvx[k], R))

        if k == 0:
            constraints.append(s_cvx[k] == s0)

        if k > 0:
            constraints.append(A[k-1] @ s_cvx[k-1] + B[k-1] @ u_cvx[k-1] + c[k-1] == s_cvx[k])

        # Trust region constraints
        constraints.append(cvx.norm((s_cvx[k] - s_prev[k]) / (s_prev[k] + eps)) <= ρ)
        # constraints.append(cvx.norm((u_cvx[k] - u_prev[k]) / (u_prev[k] + eps)) <= ρ)
        
        # thrust constraint
        constraints.append(u_cvx[k, 0] >= 0)
        constraints.append(u_cvx[k, 0] <= T_max)
        # elevator constraint
        constraints.append(cvx.abs(u_cvx[k, 1]) <= delta_e_max)


    constraints.append(A[N-1] @ s_cvx[N-1] + B[N-1] @ u_cvx[N-1] + c[N-1] == s_cvx[N])
    cost_terms.append(cvx.quad_form(s_cvx[N] - s_goal, P))

    objective = cvx.sum(cost_terms)

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve(verbose=False)
    if prob.status == 'optimal_inaccurate':
        print('inaccurate solution')
    elif prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)
    s = s_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    return s, u, J

s0 = np.array([
    150, # u0, m/s
    5, # w0, m/s
    0.01, # q0, rad/s
    0.02, # theta0, rad
    -1500, # x0, m
    -1000, # z0, m
])

T_max = 1
delta_e_max = 1

s_goal = np.array([150, 0, 0, 0, 0, -1000])  # desired state
Q = np.diag([10, 1e-3, 1e-1, 1e-1, 1e-5, 1])
P = Q
R = np.diag([5.0, 5.0])                 # control cost matrix

n = 6                                # state dimension
m = 2                                # control dimension
dt = 0.1                             # discrete time resolution
T = 15.                              # total simulation time
ρ = 0.5                               # trust region parameter
eps = 5e-2                           # convergence tolerance
max_iters = 100                      # maximum number of SCP iterations

f_no_time = lambda s, u : f(None, s, u)
# Initialize the discrete-time dynamics
fd = jax.jit(discretize(f_no_time, dt))

# Solve the problem with SCP
t = np.arange(0., T + dt, dt)
N = t.size - 1
s, u, J = solve_scp(fd, s0, s_goal, N, P, Q, R, T_max, delta_e_max, ρ,
                            eps, max_iters)

# Simulate open-loop control
for k in range(N):
    s[k+1] = fd(s[k], u[k])

Thr = u[:, 0]
delta_e = u[:, 1]
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
color = 'tab:blue'
axs[3,1].plot(horiz_var[:-1], Thr, color=color)
axs[3,1].set_ylabel('Thrust', color=color)
ax_twin = axs[3,1].twinx()
color = 'tab:orange'
ax_twin.plot(horiz_var[:-1], delta_e, color=color)
ax_twin.set_ylabel(r'$\delta_e$', color=color)
# color = 'tab:blue'
# axs[3,1].plot(horiz_var[:-1], X, color=color)
# axs[3,1].set_ylabel('X', color=color)
# ax_twin = axs[3,1].twinx()
# color = 'tab:orange'
# ax_twin.plot(horiz_var[:-1], Z, color=color)
# ax_twin.set_ylabel('Z', color=color)
# color = 'tab:green'
# ax_twin.plot(horiz_var[:-1], M, color=color)
# ax_twin.set_ylabel('M', color=color)
plt.show()
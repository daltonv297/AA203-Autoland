import numpy as np
from dynamics import f
from dynamics_simplified import f_simplified
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import cvxpy as cvx
from tqdm.auto import tqdm
from functools import partial
from time import time

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

def scp_iteration(f, s0, s_goal, s_prev, u_prev, P, Q, R, ρ):
    """Solve a single SCP sub-problem for the obstacle avoidance problem."""
    n = s_prev.shape[-1]    # state dimension
    m = u_prev.shape[-1]    # control dimension
    N = u_prev.shape[0]     # number of steps

    A, B, c = affinize(f, s_prev[:-1], u_prev)
    A, B, c = np.array(A), np.array(B), np.array(c)

    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    constraints = []
    cost_terms = []
    for k in range(N):
        cost_terms.append(cvx.quad_form(s_cvx[k] - s_goal, Q))
        cost_terms.append(cvx.quad_form(u_cvx[k], R))

        if k == 0:
            constraints.append(s_cvx[k] == s0)

        if k > 0:
            constraints.append(A[k-1] @ s_cvx[k-1] + B[k-1] @ u_cvx[k-1] + c[k-1] == s_cvx[k])

        # trust region constraint
        # constraints.append(cvx.norm((s_cvx[k] - s_prev[k]) / (s_prev[k] + eps)) <= ρ)
        
        # thrust constraint
        constraints.append(u_cvx[k, 0] >= 0)
        constraints.append(u_cvx[k, 0] <= 1)
        # # elevator constraint
        # constraints.append(cvx.abs(u_cvx[k, 1]) <= 1)

    constraints.append(A[N-1] @ s_cvx[N-1] + B[N-1] @ u_cvx[N-1] + c[N-1] == s_cvx[N])
    cost_terms.append(cvx.quad_form(s_cvx[N] - s_goal, P))

    objective = cvx.sum(cost_terms)

    # END PART (e) ############################################################

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


def solve_scp(f, s0, s_goal, N, P, Q, R, eps, max_iters, ρ,
                                 s_init=None, u_init=None,
                                 convergence_error=False):
    """Solve the obstacle avoidance problem via SCP."""
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize trajectory
    if s_init is None or u_init is None:
        s = np.zeros((N + 1, n))
        u = np.zeros((N, m))
        s[0] = s0
        for k in range(N):
            s[k+1] = f(s[k], u[k])
    else:
        s = np.copy(s_init)
        u = np.copy(u_init)

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    for i in range(max_iters):
        s, u, J[i + 1] = scp_iteration(f, s0, s_goal, s, u, P, Q, R, ρ)
        # for k in range(N):
        #     s[k+1] = f(s[k], u[k])
        dJ = np.abs(J[i + 1] - J[i])
        if dJ < eps:
            converged = True
            print('converged in ' + str(i) + ' iterations')
            break
    if not converged and convergence_error:
        raise RuntimeError('SCP did not converge!')
    return s, u

scaling_factor = np.array([
    1/200.,
    1/35.,
    10.,
    10.,
    1/1500.,
    1/100.
])

s0 = np.array([
    150, # u0, m/s
    5, # w0, m/s
    -0.01, # q0, rad/s
    -0.02, # theta0, rad
    -1500, # x0, m
    -100, # z0, m
])
s0 = s0 * scaling_factor

T_max = 1
delta_e_max = 1

s_goal = np.array([150, 20, 0, np.deg2rad(5), 0, 0])     # goal state
s_goal = s_goal * scaling_factor

n = 6                                   # state dimension
m = 2                                   # control dimension
# Q = np.diag([10, 1e-3, 1e-1, 1e-1, 1e-5, 1])
# P = Q
# R = np.diag([5.0, 5.0])               
# Q = np.diag(np.array([10, 1e-3, 1e6, 1e5, 1e-5, 1]))   # state cost matrix
# R = np.diag([100.0, 80.0])                    # control cost matrix
# P = np.diag(np.array([10, 1e-3, 1e-1, 1e-1, 1e-5, 1000]))                         # terminal state cost matrix
Q = np.diag([10, 0.1, 0.1, 0.1, 0.1, 10])
P = np.diag([10, 0.1, 0.1, 0.1, 0.1, 1000])
R = np.diag([0.01, 0.01])

# Define constants
T = 100                                  # total simulation time
dt = 0.1
eps = 1e-3                              # SCP convergence tolerance
ρ = 1.0

N = 10       # MPC horizon
N_scp = 10  # maximum number of SCP iterations

f_no_time = lambda s, u : f(None, s, u, scaling_factor)
# Initialize the discrete-time dynamics
fd = jax.jit(discretize(f_no_time, dt))

s_mpc = np.zeros((T, N + 1, n))
u_mpc = np.zeros((T, N, m))
s = np.copy(s0)
total_time = time()
total_control_cost = 0.
s_init = None
u_init = None
for t in tqdm(range(T)):
    s_mpc[t], u_mpc[t] = solve_scp(fd, s, s_goal, N, P, Q, R, eps, N_scp, ρ, s_init, u_init)

    s = fd(s, u_mpc[t, 0, :])

    # Accumulate the actual control cost
    total_control_cost += u_mpc[t, 0].T @ R @ u_mpc[t, 0]

    # Use this solution to warm-start the next iteration
    u_init = np.concatenate([u_mpc[t, 1:], u_mpc[t, -1:]])
    s_init = np.concatenate([
        s_mpc[t, 1:],
        fd(s_mpc[t, -1], u_mpc[t, -1]).reshape([1, -1])
    ])
total_time = time() - total_time
print('Total elapsed time:', total_time, 'seconds')
print('Total control cost:', total_control_cost)

s_mpc = s_mpc / scaling_factor

Thr = u_mpc[:, :, 0]
delta_e = u_mpc[:, :, 1]
u = s_mpc[:, :, 0]
w = s_mpc[:, :, 1]
q = s_mpc[:, :, 2]
theta = s_mpc[:, :, 3]
x = s_mpc[:, :, 4]
z = s_mpc[:, :, 5]

h = -z
V = np.sqrt(u*u + w*w)
alpha = np.degrees(np.arctan(w/u))
theta_deg = np.degrees(theta)
horiz_var = x

state_dict = {0: h, 1: V, 2: alpha, 3: theta_deg, 4: u, 5: w, 6: q}
label_dict = {0: 'h', 1: 'V', 2: r'$\alpha$', 3: r'$\theta$', 4: 'u', 5: 'w', 6: 'q'}

fig, axs = plt.subplots(4, 2, figsize=(12,12))
fig.suptitle('$N = {}$, '.format(N)
             + r'$N_\mathrm{SCP} = ' + '{}$'.format(N_scp))

for i, ax in enumerate(axs.T.reshape(-1)[:-1]):
    for t in range(T):
        ax.plot(horiz_var[t, :], state_dict[i][t, :], '--*', color='k')
    ax.plot(horiz_var[:, 0], state_dict[i][:, 0], '-o')
    ax.set_ylabel(label_dict[i])
    ax.axis('equal')

ax = axs.reshape(-1)[-1]

color = 'tab:blue'
ax.plot(Thr[:, 0], '-o', color=color)
ax.set_ylabel('Thrust', color=color)
ax_twin = ax.twinx()
color = 'tab:orange'
ax_twin.plot(delta_e[:, 0], color=color)
ax_twin.set_ylabel(r'$\delta_e$', color=color)
plt.show()
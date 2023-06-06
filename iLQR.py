import numpy as np
from scipy.integrate import solve_ivp
# from dynamics import f
# from dynamics_simplified import f_simplified
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import cvxpy as cvx
from tqdm import tqdm
from functools import partial
import time as time
from scipy.integrate import odeint



import numpy as np
import jax
import jax.numpy as jnp

def s_dot(t, s, u_in):
    '''Computes s_dot = f(s, u_in).

    Coordinate system (body-fixed):
    X_b: points toward aircraft nose
    Y_b: points toward aircraft right wing
    Z_b: points down perpendicular to x_b and y_b

    Coordinate system (earth-fixed):
    X_e, Y_e: horizontal perpendicular axes
    Z_e: down

    Parameters
    ----------
    s : jax.numpy.ndarray
        Current state
        s = [u, w, q, theta, x_e, z_e]
            u: velocity component parallel to X_b (m/s)
            w: velocity component parallel to Z_b (m/s)
            q: pitch rate (positive nose up) (rad/s)
            theta: pitch angle (positive nose up) (rad)
            x_e: earth-fixed x position (m)
            z_e: earth-fixed z position (positive down) (m)
            
    u_in : jax.numpy.ndarray
        Control input
        u_in = [T, delta_e]
            T: thrust command (0 <= T <= 1) (N)
            delta_e: elevator deflection command (-1 <= delta_e <= 1) (positive down) (degrees)

    Returns
    -------
    s_dot : jax.numpy.ndarray
        Time derivative of state
    '''

    # Constants:
    g = 9.81 # m/s^2
    rho = 1.225 # kg/m^3

    # https://www.boeing.com/commercial/airports/3_view.page
    m = 272000 # mass, kg
    I_y = 5.07e9 # Moment of inertia about the y-axis, kg*m^2 
    S = 511 # planform wing area, m^2
    c = 9.14 # mean aerodynamic chord, m
    AR = 8 # aspect ratio
    e = 0.85 # Oswald efficiency factor
    # m = 10
    # I_y = 1000
    # S = 1
    # c = 0.1
    # stability derivatives:
    CL_alpha = 4.758 
    CL_q = 9.911
    CL_delta_e = 0.00729
    CM_alpha = -1.177
    CM_q = -26.684
    CM_delta_e = -0.05
    CL0 = 0.1720 # AVL
    CD0 = 0.0184 # https://www.sesarju.eu/sites/default/files/documents/sid/2018/papers/SIDs_2018_paper_75.pdf
    CM0 = -0.0969 # AVL
    
    T_max = 1000e3 # N
    delta_e_max = 30 # deg

    u, w, q, theta, x_e, z_e = s
    T_c, delta_e_c = u_in

    T = T_c * T_max
    delta_e = delta_e_c * delta_e_max
    
    V = jnp.sqrt(u*u + w*w)
    alpha = jnp.arctan(w/u)
    q_hat = q * c / (2 * V)

    # Lift, drag, and moment coefficients
    CL = CL0 + CL_alpha*alpha + CL_q*q_hat + CL_delta_e*delta_e
    CM = CM0 + CM_alpha*alpha + CM_q*q_hat + CM_delta_e*delta_e
    CD = CD0 + CL*CL / (np.pi * AR * e)

    # Lift, drag, and moment forces (wind frame)
    L = 0.5 * rho * V*V * S * CL
    D = 0.5 * rho * V*V * S * CD
    M = 0.5 * rho * V*V * S * c * CM

    # X and Z forces (body frame)
    X = T - D * jnp.cos(alpha) + L * jnp.sin(alpha)
    Z = -L * jnp.cos(alpha) - D * jnp.sin(alpha)

    u_dot = X/m - g*jnp.sin(theta) - q*w
    w_dot = Z/m + g*jnp.cos(theta) + q*u
    q_dot = M/I_y
    theta_dot = q
    x_e_dot = u*jnp.cos(theta) + w*jnp.sin(theta)
    z_e_dot = w*jnp.cos(theta) - u*jnp.sin(theta)

    return jnp.array([u_dot, w_dot, q_dot, theta_dot, x_e_dot, z_e_dot])

def linearize(f, s, u):
    """Linearize the function `f(s, u)` around `(s, u)`.

    Arguments
    ---------
    f : callable
        A nonlinear function with call signature `f(s, u)`.
    s : numpy.ndarray
        The state (1-D).
    u : numpy.ndarray
        The control input (1-D).

    Returns
    -------
    A : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `s`.
    B : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `u`.
    """
    # WRITE YOUR CODE BELOW ###################################################
    # INSTRUCTIONS: Use JAX to compute `A` and `B` in one line.
    # print(u)
    A, B = jax.jacfwd(f,[0,1])(s,u)
    ###########################################################################
    return A, B

def ilqr(f, s0, s_goal, N, Q, R, QN, eps=1e-3, max_iters=1000):
    """Compute the iLQR set-point tracking solution.

    Arguments
    ---------
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    Q : numpy.ndarray
        The state cost matrix (2-D).
    R : numpy.ndarray
        The control cost matrix (2-D).
    QN : numpy.ndarray
        The terminal state cost matrix (2-D).
    eps : float, optional
        Termination threshold for iLQR.
    max_iters : int, optional
        Maximum number of iLQR iterations.

    Returns
    -------
    s_bar : numpy.ndarray
        A 2-D array where `s_bar[k]` is the nominal state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u_bar : numpy.ndarray
        A 2-D array where `u_bar[k]` is the nominal control at time step `k`,
        for `k = 0, 1, ..., N-1`
    Y : numpy.ndarray
        A 3-D array where `Y[k]` is the matrix gain term of the iLQR control
        law at time step `k`, for `k = 0, 1, ..., N-1`
    y : numpy.ndarray
        A 2-D array where `y[k]` is the offset term of the iLQR control law
        at time step `k`, for `k = 0, 1, ..., N-1`
    """
    if max_iters <= 1:
        raise ValueError('Argument `max_iters` must be at least 1.')
    n = Q.shape[0]        # state dimension
    m = R.shape[0]        # control dimension

    # Initialize gains `Y` and offsets `y` for the policy
    Y = np.zeros((N, m, n))
    y = np.zeros((N, m))
    print(y.shape)

    # Initialize the nominal trajectory `(s_bar, u_bar`), and the
    # deviations `(ds, du)`
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = f(s_bar[k], u_bar[k])
    ds = np.zeros((N + 1, n))
    du = np.zeros((N, m))
    P_init = np.zeros((N-1, n, n))
    p_init = np.zeros((N-1, n, 1))
    beta_init = np.zeros((N-1, 1))

    # iLQR loop
    converged = False
    for _ in range(max_iters):
        # Linearize the dynamics at each step `k` of `(s_bar, u_bar)`
        A, B = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
        A, B = np.array(A), np.array(B)

        # PART (c) ############################################################
        # INSTRUCTIONS: Update `Y`, `y`, `ds`, `du`, `s_bar`, and `u_bar`.
        P_tp1 = P_init
        p_tp1 = p_init
        beta_tp1 = beta_init

        P_tp1[-1] = QN
        sbar_m_sgoal = np.matrix(s_bar[-1,:] - s_goal).T

        p_tp1[-1] = (sbar_m_sgoal.T*QN).T
        # print(u_bar[-1].T@R@u_bar[-1])
        beta_tp1[-1] = 1/2*sbar_m_sgoal.T@Q@sbar_m_sgoal + 1/2*u_bar[-1].T@R@u_bar[-1]
        # print(beta_tp1[-1])
        # p_tp1[-1] = 
        # print(beta_tp1[-1] )

        # Backwards Pass
        for i in reversed(range(1, N-1)):
            A_mat = A[i]
            B_mat = np.matrix(B[i])
            P_tp1_mat = np.matrix(P_tp1[i])
            p_tp1_mat = np.matrix(p_tp1[i])
            # print(p_tp1_mat.shape)

            # Compute Hessian
            H_xx = Q + A_mat.T*P_tp1_mat*A_mat
            H_uu = R + B_mat.T*P_tp1_mat*B_mat
            H_xu = A_mat.T*P_tp1_mat*B_mat # + S, but S is zero

            # Copute Params
            sbar_m_sgoal = np.matrix(s_bar[i,:] - s_goal).T
            alf = 1/2*sbar_m_sgoal.T@Q@sbar_m_sgoal + 1/2*u_bar[i].T@R@u_bar[i]
            q = (sbar_m_sgoal.T*Q).T
            r = u_bar[i].T@R
            eta = alf + beta_tp1[i] # No other terms -- Ct = 0
            h_x = q + A_mat.T@p_tp1_mat
            h_u = np.transpose(np.matrix(r) + (B_mat.T@p_tp1_mat).T)
            # print(r[:, 1])
            
            Y[i] = -np.linalg.inv(H_uu)@H_xu.T
            y[i,:] = -np.matmul(np.linalg.inv(H_uu), h_u).T

            # print(alf)

            P_tp1[i-1] = H_xx + H_xu*Y[i]
            p_tp1[i-1] = h_x + H_xu*(y[i,:].reshape((-1,1)))
            beta_tp1[i-1] = eta + 1/2*h_u.T*y[i,:].reshape((-1,1))
        
        ds[0] = s0 - s_bar[0]
        # Forwards Pass
        for i in range(N):
            du[i] = np.dot(Y[i],ds[i]) + y[i]
            ds[i+1] = f(s_bar[i]+ds[i], u_bar[i] + du[i]) - s_bar[i+1]

        s_bar += ds
        u_bar += du

        print(np.max(np.abs(du)))
        #######################################################################

        if np.max(np.abs(du)) < eps:
            converged = True
            break
    if not converged:
        raise RuntimeError('iLQR did not converge!')
    return s_bar, u_bar, Y, y

# Define constants
s0 = np.array([
    150, # u0, m/s
    5, # w0, m/s
    0.01, # q0, rad/s
    0.02, # theta0, rad
    -1500, # x0, m
    -100, # z0, m
])
n = 6                                       # state dimension
m = 2                                       # control dimension
Q = np.diag(np.array([10, 1e-3, 1e6, 1e5, 1e-5, 1]))   # state cost matrix
R = np.diag([100.0, 80.0])                    # control cost matrix
QN = np.diag(np.array([10, 1e-3, 1e-1, 1e-1, 1e-5, 1000]))                         # terminal state cost matrix
s_goal = np.array([150, 20, 0, np.deg2rad(5), 0, 0])     # goal state
T = 15.                                     # simulation time
dt = 0.1                                    # sampling time
animate = False                             # flag for animation
closed_loop = False                         # flag for closed-loop control

# Initialize continuous-time and discretized dynamics
f = jax.jit(lambda s, u: s_dot(None, s, u))
fd = jax.jit(lambda s, u, dt=dt: s + dt*f(s, u))

# Compute the iLQR solution with the discretized dynamics
print('Computing iLQR solution ... ', end='', flush=True)
start = time.time()
t = np.arange(0., T, dt)
N = t.size - 1
s_bar, u_bar, Y, y = ilqr(fd, s0, s_goal, N, Q, R, QN)
print('done! ({:.2f} s)'.format(time.time() - start), flush=True)

# Simulate on the true continuous-time system
print('Simulating ... ', end='', flush=True)
start = time.time()
s = np.zeros((N + 1, n))
u = np.zeros((N, m))
s[0] = s0
for k in range(N):
    # PART (d) ################################################################
    # INSTRUCTIONS: Compute either the closed-loop or open-loop value of
    # `u[k]`, depending on the Boolean flag `closed_loop`.
    if closed_loop:
        u[k] = u_bar[k] + np.dot(Y[k],(s[k]-s_bar[k])) + y[k]
    else:  # do open-loop control
        u[k] = u_bar[k]
    ###########################################################################
    s[k+1] = odeint(lambda s, t: f(s, u[k]), s[k], t[k:k+2])[1]
print('done! ({:.2f} s)'.format(time.time() - start), flush=True)

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
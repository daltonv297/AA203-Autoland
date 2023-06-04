import numpy as np
import jax
import jax.numpy as jnp

def f(t, s, u_in):
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
            T: thrust (N)
            delta_e: elevator deflection angle (positive down) (degrees)

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
    
    u, w, q, theta, x_e, z_e = s
    T, delta_e = u_in
    
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
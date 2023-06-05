import numpy as np
import jax
import jax.numpy as jnp

def f_simplified(t, s, u_in):
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
        u_in = [X, Z, M]
            X: X force (N)
            Z: Z force (N)
            M: Pitching moment (N-m)

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
    
    u, w, q, theta, x_e, z_e = s
    X, Z, M = u_in

    u_dot = X/m - g*jnp.sin(theta) - q*w
    w_dot = Z/m + g*jnp.cos(theta) + q*u
    q_dot = M/I_y
    theta_dot = q
    x_e_dot = u*jnp.cos(theta) + w*jnp.sin(theta)
    z_e_dot = w*jnp.cos(theta) - u*jnp.sin(theta)

    return jnp.array([u_dot, w_dot, q_dot, theta_dot, x_e_dot, z_e_dot])
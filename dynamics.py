import numpy as np

def f(s, u_in):
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
    s : numpy.ndarray
        Current state
        s = [u, w, q, theta, x_e, z_e]
            u: velocity component parallel to X_b
            w: velocity component parallel to Z_b
            q: pitch rate (positive nose up)
            theta: pitch angle (positive nose up)
            x_e: earth-fixed x position
            z_e: earth-fixed z position (positive down)
            
    u_in : numpy.ndarray
        Control input
        u_in = [T, delta_e]
            T: thrust
            delta_e: elevator deflection angle (positive down)

    Returns
    -------
    s_dot : numpy.ndarray
        Time derivative of state
    '''

    # Constants:
    g = 9.81 # m/s^2
    rho = 1.225 # kg/m^3

    # TODO: find real values for these
    m = 1 # mass, kg
    I_y = 1 # Moment of inertia about the y-axis, kg*m^2
    S = 1 # planform wing area, m^2
    c = 1 # mean aerodynamic chord, m
    AR = 1 # aspect ratio
    e = 1 # Oswald efficiency factor
    # stability derivatives:
    CL_alpha = 1 
    CL_q = 1
    CL_delta_e = 1
    CM_alpha = 1
    CM_q = 1
    CM_delta_e = 1
    CL0 = 1
    CD0 = 1
    CM0 = 1
    
    u, w, q, theta, x_e, z_e = s
    T, delta_e = u_in
    
    V = np.sqrt(u*u + w*w)
    alpha = np.arctan(w/u)
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
    X = T - D * np.cos(alpha) + L * np.sin(alpha)
    Z = -L * np.cos(alpha) - D * np.sin(alpha)

    u_dot = X/m - g*np.sin(theta) - q*w
    w_dot = Z/m + g*np.cos(theta) + q*u
    q_dot = M/I_y
    theta_dot = q
    x_e_dot = u*np.cos(theta) + w*np.sin(theta)
    z_e_dot = w*np.cos(theta) - u*np.sin(theta)

    return np.array([u_dot, w_dot, q_dot, theta_dot, x_e_dot, z_e_dot])
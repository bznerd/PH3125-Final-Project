# Apply the rk4 method for as many steps as necessary to reach t_final
def rk4(x, u, t, dxdt, dt, t_final):
    x_next = x
    t_next = t
    # Iterate over the number of steps it will take to reach t_final rounded down to the nearest integer
    for _ in range(int((t_final-t)/dt)):
        # Apply step
        x_next, t_next = rk4_step(x_next, u, t_next, dxdt, dt)

    # If a non-integer number of steps was requried take one more shorter step to reach the target final time
    if (t_next < t_final):
        # Apply step
        x_next, t_next = rk4_step(x_next, u, t_next, dxdt, t_final-t_next) 
    
    # Return final values
    return x_next, t_next

# Runge-Kutta 4th order numerical integration method
# x is the state vector, u is the input vector, t the current time, dxdt is the time derivative of the state vector as a function of (x, u, t) and dt is the time step
# implemented using formulas from https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
def rk4_step(x, u, t, dxdt, dt):
    k1 = dxdt(x, u, t)
    k2 = dxdt(x + dt * k1/2, u, t + dt/2)
    k3 = dxdt(x + dt * k2/2, u, t + dt/2)
    k4 = dxdt(x + dt * k3, u, t + dt)

    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4), t + dt

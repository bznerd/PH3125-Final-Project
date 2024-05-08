# Apply the rk4 method for as many steps as necessary to reach t_final
def rk4(x, t, dxdt, dt, t_final):
    x_next = x
    t_next = t
    # Iterate over the number of steps it will take to reach t_final rounded down to the nearest integer
    for _ in range(int((t_final-t)/dt)):
        # Apply step
        x_next, t_next = rk4_step(x, t, dxdt, dt)

    # If a non-integer number of steps was requried take one more shorter step to reach the target final time
    if (t_next < t_final):
        # Apply step
        x_next, t_next = rk4_step(x, t, dxdt, t_final-t_next) 
    
    # Return final values
    return x_next, t_next

# Runge-Kutta 4th order numerical integration method
# x is the state vector, t the current time, dxdt is the time derivative of the state vector as a function of (x, t) and dt is the time step
def rk4_step(x, t, dxdt, dt):
    k1 = dxdt(x, t)
    k2 = dxdt(x + dt * k1/2, t + dt/2)
    k3 = dxdt(x + dt * k2/2, t + dt/2)
    k4 = dxdt(x + dt * k3, t + dt)

    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4), t + dt

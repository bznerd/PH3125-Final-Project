import numpy as np
# Drag force
# takes velocity vector, fluid density rho, drag coefficient and frontal surface area
def drag_force(velocity, rho, drag_coeff, area):
    # unit vector in the direction of velocity
    v_hat = velocity / np.linalg.norm(velocity)
    # F_drag = 1/2 C_d*p*A*v^2 directed opposite the direction of travel
    return -(1/2 * drag_coeff * rho * area * velocity**2) * v_hat


# Lift force
# takes relative air velcocity in the direction of the wing, fluid density rho, lift coefficient, and wing area
# assumes no vertical velocity
def lift_force(velocity, rho, coeff_lift, area):
    direction = np.array([[0,0,1]])
    # F_lift = 1/2 C_l*p*A*V^2 directed perpendicular to velocity
    return direction * (1/2 * coeff_lift * rho * area * velocity**2)
    

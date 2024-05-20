import numpy as np

# Constants
air_density = 1
transition_coefficient = 0.01 # unitless
transition_coefficient = 0.006 # unitless
g = 9.81 # m/s^2

# Drag force
# takes velocity vector, fluid density rho, drag coefficient and frontal surface area
def drag_force(velocity, rho, drag_coeff, area):
    # unit vector in the direction of velocity
    v_mag = np.linalg.norm(velocity)
    v_hat = velocity / v_mag if v_mag > 0 else np.zeros(len(velocity))
    # F_drag = 1/2 C_d*p*A*v^2 directed opposite the direction of travel
    return -(1/2 * drag_coeff * rho * area * v_mag**2) * v_hat


# Lift force
# takes relative air velcocity in the direction of the wing, fluid density rho, lift coefficient, and wing area
# assumes no vertical velocity
def lift_force(velocity, rho, coeff_lift, area):
    direction = np.array([0,0,1])
    v_mag = np.linalg.norm(velocity)
    # F_lift = 1/2 C_l*p*A*V^2 directed perpendicular to velocity
    return direction * (1/2 * coeff_lift * rho * area * v_mag**2)


# I'm fucking quitting
def static_to_kinetic_transition(F_static, F_kinetic, F_limit, transition_coefficient):
    force_mag = np.linalg.norm(F_static)
    transition_ratio = (1 + np.e ** (1/transition_coefficient - force_mag/(transition_coefficient * F_limit)))**(-1)
    return (1 - transition_ratio) * F_static + transition_ratio * F_kinetic, transition_ratio
    

# Torque from a dc motor
# takes motor angular speed w, applied voltage V_app, free speed w_free, max voltage V_max, stall torque T_stall, and stall current I_stall
def dc_motor_torque(w, V_app, w_free, V_max, T_stall, I_stall):
    K_s = T_stall/I_stall
    R_m = V_max/I_stall
    V_emf = w/w_free * V_max

    I_stator = (V_app - V_emf)/R_m
    return I_stator * K_s

class DC_Motor:
    def __init__(self, w_free, V_max, T_stall, I_stall):
        self.w_free = w_free
        self.V_max = V_max
        self.T_stall = T_stall
        self.I_stall = I_stall

    def calculate_torque(self, w, V_app):
        return dc_motor_torque(w, V_app, self.w_free, self.V_max, self.T_stall, self.I_stall)

# Collection of all the parameters, state, and dynamics of the simulated race car 
# State vector is x,y position, x,y velocity, and the car heading
class RaceCar:
    def __init__(self, mass, drag_coeff, drag_area, coeff_lift, wing_area, wheel_base, wheel_radius, coeff_friction, dc_motor):
        self.mass = mass # kg
        self.drag_coeff = drag_coeff # 
        self.drag_area = drag_area
        self.coeff_lift = coeff_lift
        self.wing_area = wing_area
        self.wheel_base = wheel_base # m
        self.wheel_radius = wheel_radius # m
        self.coeff_friction = coeff_friction
        self.dc_motor = dc_motor
        self.state = np.zeros(5)

    def steering_angle_to_turn_radius(self, steering_angle):
        return np.sqrt((self.wheel_base/2)**2 + self.wheel_base**2 * np.tan(steering_angle)**(-2))

    def dxdt(self, x, u, t):
        velocity = x[2:4] 
        heading = x[4]

        heading_vector = np.array([np.cos(heading), np.sin(heading)])
        if u[2] != 0:
            inward_normal = np.array([-1 * np.sin(heading), np.cos(heading)]) if u[2] > 0 else np.array([np.sin(heading), -1 * np.cos(heading)])
        else: inward_normal = np.zeros(2)
        forward_motion = velocity.dot(heading_vector) * heading_vector

        wheel_speed = np.linalg.norm(forward_motion)/self.wheel_radius
        motor_force = self.dc_motor.calculate_torque(wheel_speed, u[0])/self.wheel_radius * heading_vector
        braking_force = -1 * u[1] * heading_vector 

        turn_radius = self.steering_angle_to_turn_radius(u[2])
        centripetal_acceleration = np.linalg.norm(forward_motion)**2 / turn_radius
        turning_force = centripetal_acceleration * self.mass * inward_normal

        static_friction = motor_force + braking_force + turning_force
        kinetic_friction = -1 * velocity/np.linalg.norm(velocity) * g * self.mass if np.linalg.norm(velocity) != 0 else np.zeros(2)
        normal_force = g * self.mass + 1 * lift_force(forward_motion, air_density, self.coeff_lift, self.wing_area)[2]
        limit_of_friction = self.coeff_friction * normal_force 
        frictional_force, transition_factor = static_to_kinetic_transition(static_friction, kinetic_friction, limit_of_friction, transition_coefficient)

        drag = drag_force(velocity, air_density, self.drag_coeff, self.drag_area)
        f_net = drag + frictional_force

        acceleration = f_net / self.mass
        angular_velocity = np.linalg.norm(forward_motion)/turn_radius * np.sign(u[2])

        return np.array([velocity[0], velocity[1], acceleration[0], acceleration[1], angular_velocity])
        

    def update_state(self, new_state):
        self.state = new_state

    def get_state(self):
        return self.state


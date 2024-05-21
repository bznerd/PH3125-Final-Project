import numpy as np

# Constants
air_density = 1.225 # kg/m^3
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

# Static to kinetic friction transition
# takes static friction force, kinetic friction force, the limit of static friction, and a transition coefficient and calculates a smooth transition from static to kinetic friction as static friction approaches the limit
def static_to_kinetic_transition(F_static, F_kinetic, F_limit, transition_coefficient):
    force_mag = np.linalg.norm(F_static) # calculate the magnitude of static friction
    transition_ratio = (1 + np.e ** (1/transition_coefficient - force_mag/(transition_coefficient * F_limit)))**(-1) # calculate the transition ratio from static to kinetic friction using a signmoid curve
    return (1 - transition_ratio) * F_static + transition_ratio * F_kinetic, transition_ratio # apply a weighted average of the static and kinetic friction based on the transition ratio
    

# Torque from a dc motor
# takes motor angular speed w, applied voltage V_app, free speed w_free, max voltage V_max, stall torque T_stall, and stall current I_stall
# units are rad/s volts, Nm, and amps
def dc_motor_torque(w, V_app, w_free, V_max, T_stall, I_stall):
    K_s = T_stall/I_stall # Torque constant
    R_m = V_max/I_stall # Winding resistance in ohms
    V_emf = w/w_free * V_max # emf in volts

    I_stator = (V_app - V_emf)/R_m # winding current in amps
    return I_stator * K_s # output torque in Nm

# Stores the constants for a dc motor and allows calculating the output torque from applied voltage and angular velocity
class DC_Motor:
    # Constructor requires constants to define the motor dynamics
    def __init__(self, w_free, V_max, T_stall, I_stall):
        self.w_free = w_free # rads/s
        self.V_max = V_max # volts
        self.T_stall = T_stall # Nm
        self.I_stall = I_stall # amps

    # calculate the torque from angular velocity and applied voltage
    def calculate_torque(self, w, V_app):
        return dc_motor_torque(w, V_app, self.w_free, self.V_max, self.T_stall, self.I_stall) # Nm

# Collection of all the parameters, state, and dynamics of the simulated race car 
# State vector is x,y position, x,y velocity, and the car heading
class RaceCar:
    # Constructor
    def __init__(self, mass, drag_coeff, drag_area, coeff_lift, wing_area, wheel_base, wheel_radius, coeff_friction, dc_motor):
        self.mass = mass # kg
        self.drag_coeff = drag_coeff # unitless
        self.drag_area = drag_area # m^2
        self.coeff_lift = coeff_lift # unitless
        self.wing_area = wing_area # m^2
        self.wheel_base = wheel_base # m
        self.wheel_radius = wheel_radius # m
        self.coeff_friction = coeff_friction # unitless
        self.dc_motor = dc_motor
        self.state = np.zeros(5)

    # calculate the turn radius for a given steering angle using the kinematics of the chassis
    # steering angle in radians, radius in meters
    def steering_angle_to_turn_radius(self, steering_angle):
        return np.sqrt((self.wheel_base/2)**2 + self.wheel_base**2 * np.tan(steering_angle)**(-2)) # m

    # time derivative of the state vector
    def dxdt(self, x, u, t):
        # slice the velocity from the state vector
        velocity = x[2:4] 
        # car heading
        heading = x[4]

        # make a unit vector in the direciton of the car heading
        heading_vector = np.array([np.cos(heading), np.sin(heading)])
        # calculate the inward normal vector (points into the turn)
        if u[2] != 0:
            # 90 degrees to heading if the turn is non zero
            inward_normal = np.array([-1 * np.sin(heading), np.cos(heading)]) if u[2] > 0 else np.array([np.sin(heading), -1 * np.cos(heading)])
        # zero vector if the turn is zero
        else: inward_normal = np.zeros(2)
        # vector representing velocity in the direction of the car heading
        forward_motion = velocity.dot(heading_vector) * heading_vector

        # dervive the wheel speed from the forward velocity of the car and wheel radius
        wheel_speed = np.linalg.norm(forward_motion)/self.wheel_radius # rad/s
        # calculate the force applied by the motor using the motor torque multplied by the wheel radius applied in the direction of the car heading
        motor_force = self.dc_motor.calculate_torque(wheel_speed, u[0])/self.wheel_radius * heading_vector # N
        # Braking force is taken from the input vector and applied opposite the car heading
        braking_force = -1 * u[1] * heading_vector # N

        # Turn radius from steering angle
        turn_radius = self.steering_angle_to_turn_radius(u[2]) # m
        # centripetal acceleartion from turn radius and forward velocity
        centripetal_acceleration = np.linalg.norm(forward_motion)**2 / turn_radius # m/s^2
        # force required to turn the car in the direction of the inward normal
        turning_force = centripetal_acceleration * self.mass * inward_normal # N

        # Static friction force required to accelerate/brake/turn
        static_friction = motor_force + braking_force + turning_force # N
        # Force of kinetic friction if the car were sliding directed opposite the direction of travel
        kinetic_friction = -1 * velocity/np.linalg.norm(velocity) * g * self.mass if np.linalg.norm(velocity) != 0 else np.zeros(2) # N
        # Normal force is the sum of the force of gravity and force generated by downforce producing wings
        normal_force = g * self.mass + 1 * lift_force(forward_motion, air_density, self.coeff_lift, self.wing_area)[2] # N
        # Limit of static friction from normal force and coefficient of friction 
        limit_of_friction = self.coeff_friction * normal_force # N
        # Frictional force and the transition factor between static and kinetic caclulated using the smooth transition function
        frictional_force, transition_factor = static_to_kinetic_transition(static_friction, kinetic_friction, limit_of_friction, transition_coefficient) # N

        # Calculate the force of drag on the car acting opposite the direction of travel
        drag = drag_force(velocity, air_density, self.drag_coeff, self.drag_area) # N
        # Net force acting on the car is the sum of drag and friction
        f_net = drag + frictional_force # N

        # acceleration of the car a = F/m
        acceleration = f_net / self.mass # m/s^2
        # rate of change of the cars heading calculated from the turn radius of the car
        angular_velocity = np.linalg.norm(forward_motion)/turn_radius * np.sign(u[2]) # rad/s

        # return the state derivative vector
        return np.array([velocity[0], velocity[1], acceleration[0], acceleration[1], angular_velocity])
        

    # store a new state for the car
    def update_state(self, new_state):
        self.state = new_state

    # return the state stored for the car
    def get_state(self):
        return self.state


from modules import physics
import numpy as np
# Car parameters

# Basic chassis parameters
mass = 200 # kg
wheel_base = 1.7 # m
wheel_radius = 0.26 # m 
coeff_friction = 1.9 # unitless

# DC Motor parameters
w_free = 150 # rad/s
V_max = 200 # Volts
T_stall = 700 # Nm
I_stall = 180 # A

# No downforce car
drag_coeff = 0.63 # unitless
drag_area = 2.5 # m^2

# Downforce car
downforce_drag_coeff = 0.98 # unitless
downforce_drag_area = 3.5 # m^2
lift_coeff = 2.58 # unitless
wing_area = 3.5 # m^2

# shared dc motor object 
dc_motor = physics.DC_Motor(w_free, V_max, T_stall, I_stall)

# race car objects with different parameters
no_drag_car = physics.RaceCar(mass, 0, 0, 0, 0, wheel_base, wheel_radius, coeff_friction, dc_motor)
no_downforce_car = physics.RaceCar(mass, drag_coeff, drag_area, 0, 0, wheel_base, wheel_radius, coeff_friction, dc_motor)
downforce_car = physics.RaceCar(mass, downforce_drag_coeff, downforce_drag_area, lift_coeff, wing_area, wheel_base, wheel_radius, coeff_friction, dc_motor)

# control inputs
straight_line = np.array([150, 0, 0])
constant_turn = np.array([150, 0, 0.15])
wide_turn = np.array([150, 0, 0.1])

# scenarios
scenarios = {
    "straight_line_no_drag": {"car": no_drag_car, "u": straight_line},
    "straight_line_drag": {"car": no_downforce_car, "u": straight_line},
    "straight_line_high_drag": {"car": downforce_car, "u": straight_line},
    "constant_turn_no_downforce": {"car": no_downforce_car, "u": constant_turn},
    "constant_turn_downforce": {"car": downforce_car, "u": constant_turn}}

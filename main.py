from modules import sim, physics
from time import sleep
from matplotlib import pyplot as plt
import numpy as np

mass = 200 # kg
drag_coeff = 0
drag_area = 0
lift_coeff = 0
wing_area = 0
wheel_base = 1.7 # m
wheel_radius = 0.26 # m 
coeff_friction = 1.3 # unitless
w_free = 120 # rad/s
V_max = 200 # Volts
T_stall = 200 # Nm
I_stall = 180 # A

dc_motor = physics.DC_Motor(w_free, V_max, T_stall, I_stall)
racecar = physics.RaceCar(mass, drag_coeff, drag_area, lift_coeff, wing_area, wheel_base, wheel_radius, coeff_friction, dc_motor)

t = 0
step_size = 0.02
sim_interval = 0.02
control = np.array([200, 0, 0])
position_x = [0]
position_y = [0]
velocity = [0]
lateral_velocity = [0]
time = [t]

for i in range(3000):
    x_current = racecar.get_state()
    t_current = t

    control = np.array([200, 0, 0.0005 * t])

    x_next, t_next = sim.rk4(x_current, control, t_current, racecar.dxdt, step_size, t_current + sim_interval)
    #print(x_next)
    racecar.update_state(x_next)
    t = t_next
    position_x.append(x_next[0])
    position_y.append(x_next[1])
    velocity.append(np.linalg.norm(x_next[2:4]))
    heading = np.array([-np.sin(x_next[4]), np.cos(x_next[4])])
    lateral_velocity.append(np.linalg.norm(heading.dot(x_next[2:4]) * heading))
    time.append(t)

#plt.plot(position_x, position_y, label="Position (m)")
plt.plot(time, velocity, label="Velocity (m/s)")
plt.plot(time, lateral_velocity, label="Lateral velocity (m/s)")
plt.legend()
plt.show()

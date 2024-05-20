from modules import sim
from matplotlib import pyplot as plt
from matplotlib import animation, patches
import numpy as np
import scenarios

scenario = "constant_turn_downforce" 
car = scenarios.scenarios[scenario]["car"]
u = scenarios.scenarios[scenario]["u"]

car_width = 1.5
car_length = 3

def offset_car(position, heading):
    return position - np.matmul(np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]), np.array([car_length, car_width])/2)

step_size = 0.02
control = np.array([200, 0, 0.15])
sim_length = 30
fps = 25

def simulate(racecar, control, sim_length, step_size, fps):
    racecar.update_state(np.zeros(len(racecar.get_state())))
    t = 0
    state = np.array([racecar.get_state()])
    forward_velocity = [0]    
    lateral_velocity = [0]
    time = [t]

    while t < sim_length:
        x_current = racecar.get_state()
        t_current = t

        x_next, t_next = sim.rk4(x_current, control, t_current, racecar.dxdt, step_size, t_current + 1/fps)
        racecar.update_state(x_next)
        t = t_next
        state = np.append(state, [x_next], axis=0) 
        forward_velocity.append(np.linalg.norm(x_next[2:4]))
        heading = np.array([-np.sin(x_next[4]), np.cos(x_next[4])])
        lateral_velocity.append(np.linalg.norm(heading.dot(x_next[2:4]) * heading))
        time.append(t)

    return state, forward_velocity, lateral_velocity, time

def xy_position_plot(name, fps, state, forward_velocity, lateral_velocity, time, window):
    fig, axes = plt.subplots(1,2, figsize=(15,8))
    position_plot, velocity_plot = axes
    fig.suptitle(f"Scenario: {name}")
    position_plot.set(title="XY Position of Car", xlabel="X position (m)", ylabel="Y position (m)")
    velocity_plot.set(title="Velocity of Car Over Time", xlabel="Time (s)", ylabel="Velocity (m/s)")

    car = patches.Rectangle(offset_car(state[0,:2], state[0,4]), car_length, car_width, angle=(180/np.pi) * state[0,4])
    position_plot.set(xlim=window[0], ylim=window[1])
    position_plot.set_aspect("equal", adjustable="box")
    velocity_plot.set(xlim=[0,time[-1]+5], ylim=[0, 20])
    position_line = position_plot.plot(state[0,0], state[0,1], label="Position (m)")[0]
    position_plot.add_patch(car)
    position_plot.grid()
    velocity_line = velocity_plot.plot(time[0], forward_velocity[0], label="Forward velocity (m/s)")[0]
    lateral_velocity_line = velocity_plot.plot(time[0], lateral_velocity[0], label="Lateral velocity (m/s)")[0]
    velocity_plot.legend()

    def update(frame):
        position_line.set_xdata(state[:frame, 0]) 
        position_line.set_ydata(state[:frame, 1])
        car.set_xy(offset_car(state[frame,:2], state[frame,4]))
        car.set_angle((180/np.pi) * state[frame,4])

        velocity_line.set_xdata(time[:frame]) 
        velocity_line.set_ydata(forward_velocity[:frame])

        lateral_velocity_line.set_xdata(time[:frame]) 
        lateral_velocity_line.set_ydata(lateral_velocity[:frame])
        return (position_line, velocity_line, lateral_velocity_line)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(time), interval=int(1000/fps))
    ani.save(f"plots/{name}.gif")
    plt.show()

def position_v_time_plot(name, state, forward_velocity, time):
    fig, axes = plt.subplots(1,2, figsize=(15,8))
    position_plot, velocity_plot = axes
    fig.suptitle(f"Scenario: {name}")
    position_plot.set(title="Position of Car Over Time", xlabel="Time (s)", ylabel="X position (m)")
    velocity_plot.set(title="Velocity of Car Over Time", xlabel="Time (s)", ylabel="Velocity (m/s)")

    position_plot.plot(time, state[:,0], label="Position (m)")[0]
    velocity_plot.plot(time, forward_velocity, label="Forward velocity (m/s)")[0]

    plt.savefig(f"plots/{name}.png")
    plt.show()

state, forward_velocity, lateral_velocity, time = simulate(car, u, sim_length, step_size, fps)
print("Simulation done, generating plots")

if "straight_line" in scenario:
    position_v_time_plot(scenario, state, forward_velocity, time)
else:
    window = ([-20, 20], [-5, 35])
    xy_position_plot(scenario, fps, state, forward_velocity, lateral_velocity, time, window=window)

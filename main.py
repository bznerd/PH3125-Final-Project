# Import libraries and modules
from modules import sim
from matplotlib import pyplot as plt
from matplotlib import animation, patches
import numpy as np
import scenarios

# Set the scenario to be run and load the car and control vector objects
scenario = "constant_turn_downforce" 
car = scenarios.scenarios[scenario]["car"]
u = scenarios.scenarios[scenario]["u"]

# Car width and length for plotting
car_width = 1.5 # m
car_length = 3 # m

# Method to offset the car so that the center of the rectangle lies at the passed position
def offset_car(position, heading):
    # apply a rotation matrix on the translation between the corner and center of the rectangle and subtract this from the passed position
    return position - np.matmul(np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]), np.array([car_length, car_width])/2)

# simulation parameters
step_size = 0.02 # step size (s)
sim_length = 30 # length of simulation (s)
fps = 25 # frames per second for the animation (Hz)

# Simulate the race car for the specificed period of time
def simulate(racecar, control, sim_length, step_size, fps):
    # Reset the state of the car
    racecar.update_state(np.zeros(len(racecar.get_state())))
    # Set variables to record state over the simulation length
    t = 0
    state = np.array([racecar.get_state()])
    forward_velocity = [0] # forward velocity m/s
    lateral_velocity = [0] # sideways velocity m/s
    time = [t]

    # Run simulation steps until the simulation length is reached
    while t < sim_length:
        # Get the current state and time
        x_current = racecar.get_state()
        t_current = t

        # integrate one step forward
        x_next, t_next = sim.rk4(x_current, control, t_current, racecar.dxdt, step_size, t_current + 1/fps)
        # update the car state with the new state
        racecar.update_state(x_next)
        t = t_next
        # Record the state, velocities, and time
        state = np.append(state, [x_next], axis=0) 
        forward_velocity.append(np.linalg.norm(x_next[2:4]))
        heading = np.array([-np.sin(x_next[4]), np.cos(x_next[4])])
        lateral_velocity.append(np.linalg.norm(heading.dot(x_next[2:4]) * heading))
        time.append(t)

    # Return the recorded states, velocites, and time
    return state, forward_velocity, lateral_velocity, time

# Figure with two plots, one for the xy position animated over time, another with the forward and lateral velocity of the car
def xy_position_plot(name, fps, state, forward_velocity, lateral_velocity, time, window):
    # create the plot object
    fig, axes = plt.subplots(1,2, figsize=(15,8))
    # unpack subplots
    position_plot, velocity_plot = axes
    # Set the plot and axis titles
    fig.suptitle(f"Scenario: {name}")
    position_plot.set(title="XY Position of Car", xlabel="X position (m)", ylabel="Y position (m)")
    velocity_plot.set(title="Velocity of Car Over Time", xlabel="Time (s)", ylabel="Velocity (m/s)")

    # Set the sizes of the plots
    position_plot.set(xlim=window[0], ylim=window[1])
    position_plot.set_aspect("equal", adjustable="box")
    position_plot.grid()
    velocity_plot.set(xlim=[0,time[-1]+5], ylim=[0, 20])

    # Create the objects for the plot lines and the rectangle representing the car
    car = patches.Rectangle(offset_car(state[0,:2], state[0,4]), car_length, car_width, angle=(180/np.pi) * state[0,4])
    position_line = position_plot.plot(state[0,0], state[0,1], label="Position (m)")[0]
    position_plot.add_patch(car)
    velocity_line = velocity_plot.plot(time[0], forward_velocity[0], label="Forward velocity (m/s)")[0]
    lateral_velocity_line = velocity_plot.plot(time[0], lateral_velocity[0], label="Lateral velocity (m/s)")[0]

    # Display a legend for the velocity plot
    velocity_plot.legend()

    # Update method to draw the lines over time
    def update(frame):
        # Display the position line up until this frame number
        position_line.set_xdata(state[:frame, 0]) 
        position_line.set_ydata(state[:frame, 1])
        # Update the position of the rectangle to the latest position of the car
        car.set_xy(offset_car(state[frame,:2], state[frame,4]))
        car.set_angle((180/np.pi) * state[frame,4])

        # Display the velocity lines up until this frame number
        velocity_line.set_xdata(time[:frame]) 
        velocity_line.set_ydata(forward_velocity[:frame])

        lateral_velocity_line.set_xdata(time[:frame]) 
        lateral_velocity_line.set_ydata(lateral_velocity[:frame])
        return (position_line, velocity_line, lateral_velocity_line)

    # Animate the plots over the simulation length at the specified fps
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(time), interval=int(1000/fps))
    # Save the plot in the plots folder
    ani.save(f"plots/{name}.gif")
    # Display the plot
    plt.show()

# figure with two plots, position versus time and velocity versus time
def position_v_time_plot(name, state, forward_velocity, time):
    # two subplots
    fig, axes = plt.subplots(1,2, figsize=(15,8))
    # unpack plots
    position_plot, velocity_plot = axes
    # Set plot and axis titles
    fig.suptitle(f"Scenario: {name}")
    position_plot.set(title="Position of Car Over Time", xlabel="Time (s)", ylabel="X position (m)")
    velocity_plot.set(title="Velocity of Car Over Time", xlabel="Time (s)", ylabel="Velocity (m/s)")

    # Display the x position of the car versus time
    position_plot.plot(time, state[:,0], label="Position (m)")[0]
    # Dispaly the forward velocity of the car versus time
    velocity_plot.plot(time, forward_velocity, label="Forward velocity (m/s)")[0]

    # Save the plot in the plots folder
    plt.savefig(f"plots/{name}.png")
    # Display the plot
    plt.show()

# Run the simulation for the specified scenario
state, forward_velocity, lateral_velocity, time = simulate(car, u, sim_length, step_size, fps)
print(f"Maximum forward velocity: {forward_velocity[-1]} m/s, Maximum lateral velocity: {lateral_velocity[-1]} m/s")
print("Simulation done, generating plots")

# If it is a straight line scenario display the position versus time plots
if "straight_line" in scenario:
    position_v_time_plot(scenario, state, forward_velocity, time)
# If it is a turning scenario display the xy position animated plots
else:
    window = ([-20, 20], [-5, 35])
    xy_position_plot(scenario, fps, state, forward_velocity, lateral_velocity, time, window=window)

#!/usr/bin/python3
import numpy as np
from gymnasium.spaces import Box
from mr_exploration.dynamics.double_integrator import DoubleIntegrator
from mr_exploration.dynamics.dublin_car import DublinCarModel
from mr_exploration.controllers.ergodic_controller import RTErgodicController
from mr_exploration.util.target_dist import TargetDist
from mr_exploration.util.utils import *
from mr_exploration.agents.agent import Agent

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    num_agents = 1         # Number of robots
    max_speed = 0.1        # Maximum allowable speed

    # Define the observation, action, and exploration spaces
    observation_space = Box(np.array([0., 0., -np.inf, -np.inf]),
                            np.array([1.0, 1.0, np.inf, np.inf]),
                            dtype=np.float64)

    action_space = Box(np.array([-1., -1.]),
                       np.array([1.0, 1.0]),
                       dtype=np.float64)

    explr_space = Box(np.array([0., 0.]),
                      np.array([1.0, 1.0]),
                      dtype=np.float64)

    # Initialize the dynamics. Uncomment and switch models if needed.
    di = DoubleIntegrator(0.1, 0.1,
                          observation_space, 
                          action_space,
                          explr_space)
    # Alternatively, you might use:
    # di = DublinCarModel(observation_space, action_space, explr_space)

    # Create target distribution and controller
    target_dist = TargetDist(num_nodes=3)
    controller = RTErgodicController(dynamics=di, horizon=15,
                                     num_basis=5, batch_size=200, capacity=500)
    # Optionally set Fourier coefficients for the target distribution:
    # controller.phik = convert_phi2phik(controller.basis, target_dist.grid_vals, target_dist.grid)

    # Create the agent (robot) and assign the target distribution to it
    robot = Agent(initial_state=np.array([0, 0, 0, 0]),
                  dynamics=di,
                  controller=controller,
                  agent_idx=0,
                  total_agents=num_agents,
                  max_speed=max_speed)
    robot.t_dist = target_dist

    robots = [robot]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    xy, vals = target_dist.get_grid_spec()  # static target distribution background
    ax.contourf(*xy, vals, levels=10, alpha=0.6)  # Display as a contour plot

    # For each robot, create a line object to display its trajectory.
    robot_lines = []
    for _ in robots:
        line, = ax.plot([], [], marker='o', markersize=6, lw=2)
        robot_lines.append(line)

    ax.set_title("Real-Time Robot Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Animation update function: advances the simulation and updates the plot
    def update(frame):
        for i, r in enumerate(robots):
            r.run(steps=1)  # Advance simulation by one step for each robot
            trajectory = np.array(r._trajectory)  # Assuming _trajectory stores positions [x, y, ...]
            if trajectory.size > 0:
                # Update the line object with the current trajectory data
                robot_lines[i].set_data(trajectory[:, 0], trajectory[:, 1])
        return robot_lines

    # Create the animation with 500 frames and update every 50ms
    ani = FuncAnimation(fig, update, frames=500, interval=50, blit=True, repeat=False)
    
    plt.show()

if __name__ == "__main__":
    main()

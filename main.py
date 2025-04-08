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
    num_agents = 2         # Number of robots
    max_speed = 1.0        # Maximum allowable speed

    # Define the observation, action, and exploration spaces
    observation_space = Box(np.array([0., 0., -np.inf, -np.inf]),
                            np.array([1.0, 1.0, np.inf, np.inf]),
                            dtype=np.float64)

    action_space = Box(np.array([-max_speed, -max_speed]),
                       np.array([max_speed, max_speed]),
                       dtype=np.float64)

    explr_space = Box(np.array([0., 0.]),
                      np.array([1.0, 1.0]),
                      dtype=np.float64)

    # Initialize the dynamics. Uncomment and switch models if needed.
    di = DoubleIntegrator(1.0, 1.0,
                          observation_space,
                          action_space,
                          explr_space)

    # Create target distribution and controller
    target_dist = TargetDist(num_nodes=3)
    controller = RTErgodicController(dynamics=di, horizon=15,
                                     num_basis=5, batch_size=200, capacity=500)

    # Create multiple agents with different initial states.
    robots = []
    
    # Specify initial states for some agents.
    initial_states = [
        np.array([0.5, 0.1, 0.1, 0]),
        np.array([0.1, 0.9, 0.0, 0.1]),
        np.array([0.8, 0.8, 0.1, 0]),
        np.array([0.2, 0.8, 0.1, 0])
    ]
    
    for i in range(num_agents):
        robot = Agent(initial_state=initial_states[i],
                    dynamics=di,
                    controller=controller,
                    agent_idx=i,
                    total_agents=num_agents,
                    max_speed=max_speed)
        robot.t_dist = target_dist
        robots.append(robot)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    xy, vals = target_dist.get_grid_spec()  # static target distribution background

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

    def update(frame):
        # Clear the previous frame.
        ax.cla()

        # Re-draw static elements (e.g., target distribution background).
        ax.contourf(*xy, vals, levels=10)
        ax.set_title("Real-Time Robot Movement")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Advance the simulation and plot the robot as a dot.
        r: Agent
        other_robot: Agent
        for r in robots:
            for other_robot in robots:
                if r.idx != other_robot.idx:
                    r.update_ck(other_robot.idx, other_robot._controller.ck)
            r.run(steps=1)  # Advance simulation by one step
            if len(r._trajectory) > 0:
                pos = r._trajectory[-1]  # Get the current position
                # Plot as a red dot
                ax.plot(pos[0], pos[1], 'ro', markersize=10)

        # Return all drawn artists.
        return ax.lines

    # Create the animation with 500 frames and update every 50ms
    ani = FuncAnimation(fig, update, frames=1000,
                        interval=60, blit=True, repeat=False)

    plt.show()


if __name__ == "__main__":
    main()

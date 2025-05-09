#!/usr/bin/python3
import numpy as np
from gymnasium.spaces import Box
from mr_exploration.dynamics.double_integrator import DoubleIntegrator
from mr_exploration.controllers.ergodic_controller import RTErgodicController
from mr_exploration.fourier_metric.distribution import Distribution
from mr_exploration.fourier_metric.utils import *
from mr_exploration.agents.agent import Agent
from mr_exploration.sensor.simple_sensor import SimpleSensor
from mr_exploration.belief.belief import Belief

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from scipy.stats import multivariate_normal


def compute_discrete_entropy(prob_map: np.ndarray, base=np.e) -> float:
    """Compute entropy of a discrete probability distribution over a grid."""
    p = prob_map.flatten()
    p = p[p > 0]  # remove zero entries
    return -np.sum(p * np.log(p) / np.log(base))


def main():
    num_agents = 1         # Number of robots
    max_speed = 0.4        # Maximum allowable speed
    max_accel = 10.0

    sensing_range = 0.1

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
    di = DoubleIntegrator(max_speed,
                          max_accel,
                          observation_space,
                          action_space,
                          explr_space)
    
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Create target distribution and controller
    # target_dist = TargetDist(num_nodes=3)
    target_dist = Distribution(num_pts=30)
    controller = RTErgodicController(dynamics=di, horizon=20,
                                     num_basis=5, batch_size=200, capacity=500)

    # Create multiple agents with different initial states.
    robots = []

    # Specify initial states for some agents.
    initial_states = [
        np.array([0.1, 0.1, 0.0, 0]),
        np.array([0.1, 0.1, 0.0, 0]),
        np.array([0.8, 0.8, 0.1, 0]),
        np.array([0.2, 0.8, 0.1, 0])
    ]

    ground_truth_animal_state = [
        np.array([0.55, 0.8]),
        np.array([0.2, 0.3])
    ]

    belief_animal = [Belief(state=np.array(
        [0.5, 0.5]), variance=np.array([0.5, 0.5])**2),
        Belief(state=np.array(
            [0.5, 0.5]), variance=np.array([0.5, 0.5])**2),
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

    # Animals
    sensor = SimpleSensor(range=sensing_range, noise=0.01)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))

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
        # Clear previous contour collections (static elements) from the axes.
        for coll in ax.collections[:]:
            coll.remove()

        # Re-draw static elements (e.g., target distribution background).
        xy, vals = target_dist.get_grid_spec()  # static target distribution background
        ax.contourf(*xy, vals, levels=10)
        ax.set_title("Real-Time Robot Movement")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        drawn_artists = []
        for state in ground_truth_animal_state:
            # Plot the animal's estimated position
            animal_plot = ax.plot(state[0], state[1], 'ro', markersize=5)
            drawn_artists.append(animal_plot)

        # Advance the simulation and plot the robot as a dot.
        r: Agent
        other_robot: Agent
        for r in robots:
            # Update the robot's measurement
            measurement, valid = sensor.step(
                sensor_state=r.state[:2],
                ground_truth_state=ground_truth_animal_state
            )

            # Plot the animal's estimated position
            target_dist.means.clear()
            target_dist.vars.clear()
            means = []
            vars = []
            for idx, m in enumerate(measurement):
                if valid[idx]:
                    belief_animal[idx].step(m[0], m[1])
                means.append(belief_animal[idx].state)
                vars.append(belief_animal[idx].variance)

                animal_pos = ax.plot(
                    belief_animal[idx].state[0], belief_animal[idx].state[1], 'kx', markersize=7)
                drawn_artists.append(animal_pos)

            target_dist.update(means, vars)
            r.t_dist = target_dist
            rv = multivariate_normal(
                mean=target_dist.means[1], cov=np.diag(target_dist.vars[1]))
            probs = rv.pdf(grid_points)
            probs /= np.sum(probs)
            prob_map = probs.reshape(xx.shape)
            H = compute_discrete_entropy(prob_map)

            for other_robot in robots:
                if r.idx != other_robot.idx:
                    r.update_ck(other_robot.idx, other_robot._controller.ck)
            r.run(steps=1)  # Advance simulation by one step

            # Plotting
            if len(r._trajectory) > 0:
                pos = r._trajectory[-1]  # Get the current position
                # Plot as a red dot
                robot_plot = ax.plot(pos[0], pos[1], 'go', markersize=5)
                drawn_artists.append(robot_plot)
                robot_sensing_range = Circle(
                    (pos[0], pos[1]), sensing_range,
                    fill=False, linestyle='--', color='green')
                ax.add_patch(robot_sensing_range)
                drawn_artists.append(robot_sensing_range)

        # Return all drawn artists.
        # return ax.lines
        return drawn_artists

    # Create the animation with 500 frames and update every 50ms
    ani = FuncAnimation(fig, update, frames=500,
                        interval=80, blit=False, repeat=False)

    plt.show()


if __name__ == "__main__":
    main()

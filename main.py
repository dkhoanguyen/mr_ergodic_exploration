# !/usr/bin/python3
import numpy as np
from gymnasium.spaces import Box

from mr_exploration.dynamics.double_integrator import DoubleIntegrator
from mr_exploration.controllers.ergodic_controller import RTErgodicController
from mr_exploration.util.target_dist import TargetDist
from mr_exploration.util.utils import *

from mr_exploration.agents.agent import Agent


def main():
    num_agents = 1  # Number of robots
    max_speed = 1.0  # Set the maximum allowable speed

    observation_space = Box(np.array([0., 0., -np.inf, -np.inf]),
                            np.array([1.0, 1.0, np.inf, np.inf]),
                            dtype=np.float64)

    action_space = Box(np.array([-1., -1.]),
                       np.array([1.0, 1.0]),
                       dtype=np.float64)

    explr_space = Box(np.array([0., 0.]),
                      np.array([1.0, 1.0]),
                      dtype=np.float64)

    di = DoubleIntegrator(observation_space, action_space,
                          explr_space)
    target_dist = TargetDist(num_nodes=3)
    controller = RTErgodicController(dynamics=di, horizon=15,
                                     num_basis=5, batch_size=200, capacity=500)
    # Set the Fourier coefficients for the target distribution
    controller.phik = convert_phi2phik(
        controller.basis, target_dist.grid_vals, target_dist.grid
    )

    robot = Agent(initial_state=np.array([0, 0, 0, 0]),
                  dynamics=di,
                  controller=controller,
                  agent_idx=0,
                  total_agents=num_agents,
                  max_speed=max_speed)
    robot.t_dist = target_dist

    robots = []
    robots.append(robot)
    # Run the simulation
    steps = 255
    all_ergodic_metrics = []
    for step in range(steps):
        for robot in robots:
            # for other_robots in robots:
            #     if other_robots.idx != robot.idx:
            #         robot.update_ck(other_robots.idx, other_robots._controller.ck)
            robot.run(steps=1)

    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.animation as animation
    # Prepare data for animation
    trajectories = [np.array(robot._trajectory) for robot in robots]
    xy, vals = target_dist.get_grid_spec()

    import matplotlib.pyplot as plt
    # Plot trajectories for all agents
    plt.figure(figsize=(8, 8))
    plt.contourf(*xy, vals, levels=10)
    for robot in robots:
        trajectory = np.array(robot._trajectory)
        print(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Agent")
    plt.title("Trajectories of All Agents")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.grid()
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

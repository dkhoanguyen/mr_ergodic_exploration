import numpy as np
from double_integrator import DoubleIntegrator
from ergodic_control import RTErgodicControl
from target_dist import TargetDist
from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi
import matplotlib.pyplot as plt

# class Agent(DoubleIntegrator):
#     def __init__(self, agent_num=0, tot_agents=1):
#         DoubleIntegrator.__init__(self)

#         self.agent_num = agent_num
#         self.tot_agents = tot_agents
#         self.agent_name = f'agent{agent_num}'
#         self.model = DoubleIntegrator()

#         # Initialize the target distribution
#         self.t_dist = TargetDist(num_nodes=3)

#         # Initialize the ergodic controller
#         self.controller = RTErgodicControl(
#             self.model, self.t_dist, horizon=25, num_basis=10, batch_size=200, capacity=500
#         )

#         # Set the Fourier coefficients for the target distribution
#         self.controller.phik = convert_phi2phik(
#             self.controller.basis, self.t_dist.grid_vals, self.t_dist.grid
#         )

#         # Initialize state and trajectory
#         self.reset()
#         self.trajectory = []
#         self.ck_list = [None] * tot_agents

#     def update_ck(self, agent_idx, ck):
#         self.ck_list[agent_idx] = ck

#     def update_tdist(self, new_grid_vals):
#         print("Updating target distribution")
#         self.t_dist.grid_vals = np.array(new_grid_vals)
#         self.t_dist.has_update = True

#     def run(self, steps=10):
#         for step in range(steps):
#             # Update target distribution if needed
#             if self.t_dist.has_update:
#                 self.controller.phik = convert_phi2phik(
#                     self.controller.basis, self.t_dist.grid_vals, self.t_dist.grid
#                 )
#                 self.t_dist.has_update = False

#             # # Check communication status for all agents
#             # comm_link = all(_ck is not None for _ck in self.ck_list)

#             # Update ck in the controller and compute control input
#             # if comm_link:
#             #     ctrl = self.controller(self.state, self.ck_list, self.agent_num)
#             # else:
#             ctrl = self.controller(self.state)

#             # Store the Fourier coefficients (ck) for sharing
#             current_ck = self.controller.ck.copy()
#             self.update_ck(self.agent_num, current_ck)

#             # Update the state using the control input
#             self.state = self.step(ctrl)
#             self.trajectory.append(self.state.copy())

#             # # Print current state and Fourier coefficients for debugging
#             # print(f"Step {step}: State = {self.state}, Ck = {current_ck}")

#         # Return the trajectory for analysis
#         return np.array(self.trajectory)

# if __name__ == "__main__":
#     # Initialize the agent
#     agent = Agent(agent_num=0, tot_agents=1)

#     # Run the simulation
#     trajectory = agent.run(steps=1000)

#     # Visualize the trajectory
#     trajectory = np.array(trajectory)
#     plt.figure(figsize=(8, 8))
#     plt.scatter(trajectory[:, 0], trajectory[:, 1], label="Trajectory")
#     plt.title("Ergodic Exploration")
#     plt.xlabel("X Position")
#     plt.ylabel("Y Position")
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     # plt.grid()
#     plt.legend()
#     plt.show()

#     # # Plot trajectory as a heatmap
#     # bins = [np.linspace(0, 1, 20), np.linspace(0, 1, 20)]
#     # hist, x_edges, y_edges = np.histogram2d(trajectory[:, 0], trajectory[:, 1], bins=bins, density=True)

#     # plt.figure(figsize=(6, 6))
#     # plt.imshow(hist.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
#     # plt.colorbar(label="Density")
#     # plt.title("Empirical Distribution from Trajectory")
#     # plt.xlabel("X Position")
#     # plt.ylabel("Y Position")
#     # plt.show()

class Agent(DoubleIntegrator):
    def __init__(self, agent_num=0, tot_agents=1, max_speed=1.0):
        DoubleIntegrator.__init__(self)

        self.agent_num = agent_num
        self.tot_agents = tot_agents
        self.agent_name = f'agent{agent_num}'
        self.model = DoubleIntegrator()

        # Initialize the target distribution
        self.t_dist = TargetDist(num_nodes=3)

        # Initialize the ergodic controller
        self.controller = RTErgodicControl(
            self.model, self.t_dist, horizon=15, num_basis=5, batch_size=200, capacity=500
        )

        # Set the Fourier coefficients for the target distribution
        self.controller.phik = convert_phi2phik(
            self.controller.basis, self.t_dist.grid_vals, self.t_dist.grid
        )

        # Initialize state and trajectory
        self.reset()
        self.trajectory = []
        self.ergodic_metrics = []
        self.ck_list = [None] * tot_agents
        self.max_speed = max_speed

    def update_ck(self, agent_idx, ck):
        self.ck_list[agent_idx] = ck

    def update_tdist(self, new_grid_vals):
        print("Updating target distribution")
        self.t_dist.grid_vals = np.array(new_grid_vals)
        self.t_dist.has_update = True

    def run(self, steps=200):
        for step in range(steps):
            # Update target distribution if needed
            if self.t_dist.has_update:
                self.controller.phik = convert_phi2phik(
                    self.controller.basis, self.t_dist.grid_vals, self.t_dist.grid
                )
                self.t_dist.has_update = False

            # Check communication status for all agents
            comm_link = all(_ck is not None for _ck in self.ck_list)

            # Update ck in the controller and compute control input
            if comm_link:
                ctrl = self.controller(self.state, self.ck_list, self.agent_num)
            else:
                ctrl = self.controller(self.state)

            # Enforce maximum speed constraint during optimization
            ctrl_speed = np.linalg.norm(ctrl)
            if ctrl_speed > self.max_speed:
                ctrl = ctrl / ctrl_speed * self.max_speed

            # Store the Fourier coefficients (ck) for sharing
            current_ck = self.controller.ck.copy()
            self.update_ck(self.agent_num, current_ck)

            # Update the state using the control input
            self.state = self.step(ctrl)
            self.trajectory.append(self.state.copy())

            # Calculate and store the ergodic metric
            ergodic_metric = np.sum(self.controller.lamk * (current_ck - self.controller.phik)**2)
            self.ergodic_metrics.append(ergodic_metric)
            # print(f"Agent {self.agent_num} - Step {step}: Ergodic Metric = {ergodic_metric}")

        # Return the ergodic metrics for analysis
        return self.ergodic_metrics

if __name__ == "__main__":
    num_agents = 3  # Number of robots
    max_speed = 1.0  # Set the maximum allowable speed

    # Initialize agents
    agents = [Agent(agent_num=i, tot_agents=num_agents, max_speed=max_speed) for i in range(num_agents)]

    # Run the simulation
    steps = 250
    all_ergodic_metrics = []
    for step in range(steps):
        for agent in agents:
            # Update ck from other agents
            for other_agent in agents:
                if other_agent.agent_num != agent.agent_num:
                    agent.update_ck(other_agent.agent_num, other_agent.controller.ck)

            # Perform one step of the simulation for the agent
            agent.run(steps=1)

        # Collect metrics for plotting
        step_metrics = [agent.ergodic_metrics[-1] for agent in agents]
        all_ergodic_metrics.append(step_metrics)

    # Plot the ergodic metric over time for all agents
    # plt.figure(figsize=(8, 6))
    # for i in range(num_agents):
    #     plt.plot(
    #         [metrics[i] for metrics in all_ergodic_metrics],
    #         label=f"Agent {i}"
    #     )
    # plt.title("Ergodic Metric Over Time (Multi-Robot)")
    # plt.xlabel("Time Step")
    # plt.ylabel("Ergodic Metric")
    # plt.grid()
    # plt.legend()
    # plt.show()
    # Plot trajectories for all agents
    plt.figure(figsize=(8, 8))
    xy, vals = agents[0].t_dist.get_grid_spec()
    plt.contourf(*xy, vals, levels=10)
    for agent in agents:
        trajectory = np.array(agent.trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Agent {agent.agent_num}")
    plt.title("Trajectories of All Agents")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.show()

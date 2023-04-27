import time
from typing import Union

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions

from flatland_railway_extension.FlatlandEnvironmentHelper import FlatlandEnvironmentHelper
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from flatland_railway_extension.RailroadSwitchCluster import RailroadSwitchCluster
from flatland_railway_extension.environments.DynamicAgent import DynamicAgent
from flatland_railway_extension.environments.FlatlandDynamics import FlatlandDynamics
from flatland_railway_extension.environments.FlatlandResourceAllocator import FlatlandResourceAllocator
from flatland_railway_extension.environments.InfrastructureData import InfrastructureData
from flatland_railway_extension.environments.MultiResourcesAllocationRailEnv import MultiResourcesAllocationRailEnv
from flatland_railway_extension.utils.FlatlandDynamicsRenderer import FlatlandDynamicsRenderer
from flatland_railway_extension.utils.FlatlandRenderer import FlatlandRenderer

from example.flatland_dynamics.flatland_dynamics_simple_renderer import FlatlandDynamicsSimpleRenderer
from policy.policy import Policy
from solver.flatland_solver import FlatlandSolver


class FlatlandDynamicsSolver(FlatlandSolver):
    def __init__(self, env: FlatlandDynamics):
        super(FlatlandDynamicsSolver, self).__init__(env)
        self.env = env
        self.policy: Union[Policy, None] = None
        self.railroad_switch_analyser: Union[RailroadSwitchAnalyser, None] = None

    def get_name(self) -> str:
        return self.__class__.__name__

    def _create_infrastructure_data(self) -> InfrastructureData:
        infrastructure_data = InfrastructureData()
        cell_length_grid = np.ones((self.env.height, self.env.width)) * 400
        gradient_grid = np.zeros((self.env.height, self.env.width))
        velocity_grid = np.ones((self.env.height, self.env.width)) * 100
        for key in self.railroad_switch_analyser.railroad_switch_neighbours.keys():
            velocity_grid[key] = 80
        for key in self.railroad_switch_analyser.railroad_switches.keys():
            velocity_grid[key] = 60
        infrastructure_data.set_infrastructure_max_velocity_grid(velocity_grid / 3.6)
        infrastructure_data.set_infrastructure_cell_length_grid(cell_length_grid)
        infrastructure_data.set_infrastructure_gradient_grid(gradient_grid)
        return infrastructure_data

    def _map_infrastructure_data(self):
        # Create a test infrastructure
        # ---------------------------------------------------------------------------------------------------------------
        # share the infrastructure with the agents ( train runs)
        for agent in self.env.agents:
            if isinstance(agent, DynamicAgent):
                agent.set_infrastructure_data(
                    self._create_infrastructure_data()
                )
                agent.rolling_stock.set_max_braking_acceleration(-0.15)
                agent.set_mass(500)
                agent.set_tractive_effort_rendering(False)
        if isinstance(self.env, FlatlandDynamics):
            self.env.set_infrastructure_data(self._create_infrastructure_data())

    def before_episode_starts(self):
        self.railroad_switch_analyser = RailroadSwitchAnalyser(env=self.env)
        self._map_infrastructure_data()

    def before_step_starts(self):
        if self.renderer is not None and isinstance(self.renderer, FlatlandDynamicsSimpleRenderer):
            self.renderer.renderer.set_flatland_resource_allocator(self.env.get_active_flatland_resource_allocator())
            if self.renderer.renderer.is_closed():
                return True
        return False

    def render_flatland_dynamics_details(self):
        if self.renderer is not None and self.rendering_enabled:
            for i_agent, agent in enumerate(self.env.agents):
                n_agents = self.env.get_num_agents()
                agent.do_debug_plot(i_agent + 1, n_agents, i_agent + 1 == n_agents, i_agent == 0)

    def run_episode(self, episode, env, policy, eps, training_mode):
        total_reward = super(FlatlandDynamicsSolver, self).run_episode(episode, env, policy, eps, training_mode)
        self.render_flatland_dynamics_details()
        return total_reward

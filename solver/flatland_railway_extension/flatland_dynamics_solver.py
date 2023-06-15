from typing import Union

import numpy as np
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from flatland_railway_extension.environments.DynamicAgent import DynamicAgent
from flatland_railway_extension.environments.FlatlandDynamics import FlatlandDynamics
from flatland_railway_extension.environments.InfrastructureData import InfrastructureData

from environment.environment import Environment
from policy.policy import Policy
from rendering.base_renderer import BaseRenderer
from rendering.flatland_railway_extension.flatland_dynamics_simple_renderer import FlatlandDynamicsSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver


class FlatlandDynamicsSolver(FlatlandSolver):
    def __init__(self,
                 env: Environment,
                 policy: Policy,
                 renderer: Union[BaseRenderer, None] = None,
                 max_steps: int = 1000):
        super(FlatlandDynamicsSolver, self).__init__(env, policy, renderer)
        self.raw_env = self.env.get_raw_env()
        self.railroad_switch_analyser: Union[RailroadSwitchAnalyser, None] = None
        self.set_max_steps(max_steps)

    def get_name(self) -> str:
        return self.__class__.__name__

    def _create_infrastructure_data(self) -> InfrastructureData:
        infrastructure_data = InfrastructureData()
        cell_length_grid = np.ones((self.raw_env.height, self.raw_env.width)) * 400
        gradient_grid = np.zeros((self.raw_env.height, self.raw_env.width))
        velocity_grid = np.ones((self.raw_env.height, self.raw_env.width)) * 100
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
        for agent in self.raw_env.agents:
            if isinstance(agent, DynamicAgent):
                agent.set_infrastructure_data(
                    self._create_infrastructure_data()
                )
                agent.rolling_stock.set_max_braking_acceleration(-0.15)
                agent.set_mass(500)
                agent.set_tractive_effort_rendering(False)
        if isinstance(self.raw_env, FlatlandDynamics):
            self.raw_env.set_infrastructure_data(self._create_infrastructure_data())

    def before_episode_starts(self):
        self.railroad_switch_analyser = RailroadSwitchAnalyser(env=self.raw_env)
        self._map_infrastructure_data()

    def before_step_starts(self):
        if self.renderer is not None and isinstance(self.renderer, FlatlandDynamicsSimpleRenderer):
            self.renderer.renderer.set_flatland_resource_allocator(
                self.raw_env.get_active_flatland_resource_allocator())
            if self.renderer.renderer.is_closed():
                return True
        return False

    def render_flatland_dynamics_details(self):
        if self.renderer is not None and self.rendering_enabled:
            for i_agent, agent in enumerate(self.raw_env.agents):
                n_agents = self.raw_env.get_num_agents()
                agent.do_debug_plot(i_agent + 1, n_agents, i_agent + 1 == n_agents, i_agent == 0)

    def run_episode(self, episode, env, policy, eps, training_mode):
        total_reward, tot_terminate, tot_steps = super(FlatlandDynamicsSolver, self).run_episode(episode,
                                                                                                 env,
                                                                                                 policy,
                                                                                                 eps,
                                                                                                 training_mode)
        self.render_flatland_dynamics_details()
        return total_reward, tot_terminate, tot_steps

from flatland.core.env_observation_builder import ObservationBuilder
from flatland_railway_extension.FlatlandEnvironmentHelper import FlatlandEnvironmentHelper
from flatland_railway_extension.environments.FlatlandDynamics import FlatlandDynamics

from solver.environment import Environment


class FlatlandDynamicsEnvironment(Environment):

    def __init__(self,
                obs_builder_object: ObservationBuilder,
                number_of_agents=10):
        environment, action_space, observation_space = FlatlandDynamicsEnvironment._make_environment_flatland_dynamics(
            obs_builder_object,
            number_of_agents)
        super(FlatlandDynamicsEnvironment, self).__init__(environment, action_space, observation_space)

    def get_name(self) -> str:
        return "Environment:FlatlandRailwayExtension:FlatlandDynamics"

    def reset(self):
        state, info = self.raw_env.reset()
        return state, info

    def step(self, actions):
        state_next, reward, terminal, info = self.raw_env.step(actions)
        return state_next, reward, terminal, info

    @staticmethod
    def _make_environment_flatland_dynamics(obs_builder_object, number_of_agents):
        flatland_environment_helper = FlatlandEnvironmentHelper(rail_env=FlatlandDynamics,
                                                                number_of_agents=number_of_agents,
                                                                random_seed=2341,
                                                                obs_builder_object=obs_builder_object)

        action_space = flatland_environment_helper.get_rail_env().action_space[0]

        obs_states, _ = flatland_environment_helper.get_rail_env().reset()
        observation_space = len(obs_states[0])

        return flatland_environment_helper.get_rail_env(), action_space, observation_space

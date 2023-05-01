from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

from solver.environment import Environment


class RailEnvironment(Environment):

    def __init__(self,
                obs_builder_object: ObservationBuilder,
                max_rails_between_cities=2,
                max_rails_in_city=4,
                malfunction_rate=1 / 1000,
                n_cities=5,
                number_of_agents=10,
                grid_width=30,
                grid_height=40,
                random_seed=0):
        environment = RailEnvironment._create_flatland_env(
            obs_builder_object,
            max_rails_between_cities,
            max_rails_in_city,
            malfunction_rate,
            n_cities,
            number_of_agents,
            grid_width,
            grid_height,
            random_seed)

        action_space = environment.action_space[0]
        obs_states, _ = environment.reset()
        observation_space = len(obs_states[0])

        super(RailEnvironment, self).__init__(environment, action_space, observation_space)


    def get_name(self) -> str:
        return "Environment:Flatland:RailEnv"

    def reset(self):
        state, info = self.raw_env.reset()
        return state, info

    def step(self, actions):
        state_next, reward, terminal, info = self.raw_env.step(actions)
        return state_next, reward, terminal, info

    @staticmethod
    def _create_flatland_env(
            obs_builder_object: ObservationBuilder,
            max_rails_between_cities=2,
            max_rails_in_city=4,
            malfunction_rate=1 / 1000,
            n_cities=5,
            number_of_agents=10,
            grid_width=30,
            grid_height=40,
            random_seed=0) -> RailEnv:
        return RailEnv(
            width=grid_width,
            height=grid_height,
            rail_generator=sparse_rail_generator(
                max_num_cities=n_cities,
                seed=random_seed,
                grid_mode=True,
                max_rails_between_cities=max_rails_between_cities,
                max_rail_pairs_in_city=max_rails_in_city
            ),
            malfunction_generator=ParamMalfunctionGen(
                MalfunctionParameters(
                    malfunction_rate=malfunction_rate, min_duration=10, max_duration=50
                )
            ),
            random_seed=random_seed,
            number_of_agents=number_of_agents,
            obs_builder_object=obs_builder_object
        )


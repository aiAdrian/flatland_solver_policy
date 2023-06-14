import os
from functools import lru_cache
from typing import Union, Callable

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv

from environment.flatland.rail_env import RailEnvironment
from utils.progress_bar import ProgressBar


class RailEnvironmentPersistable(RailEnvironment):
    def __init__(self,
                 obs_builder_object_creator: Callable[[], ObservationBuilder],
                 max_rails_between_cities=2,
                 max_rails_in_city=4,
                 malfunction_rate=1 / 1000,
                 n_cities=5,
                 number_of_agents=10,
                 grid_width=30,
                 grid_height=40,
                 grid_mode=True,
                 random_seed=25041978):
        super(RailEnvironmentPersistable, self).__init__(
            obs_builder_object=obs_builder_object_creator(),
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city,
            malfunction_rate=malfunction_rate,
            n_cities=n_cities,
            number_of_agents=number_of_agents,
            grid_width=grid_width,
            grid_height=grid_height,
            grid_mode=grid_mode,
            random_seed=random_seed
        )
        self._random_seed = random_seed
        self._loaded_env = []
        self._loaded_env_itr = 0

        self._obs_builder_object_creator = obs_builder_object_creator
        self._max_rails_between_cities = max_rails_between_cities
        self._max_rails_in_city = max_rails_in_city
        self._malfunction_rate = malfunction_rate
        self._n_cities = n_cities
        self._number_of_agents = number_of_agents
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._grid_mode = grid_mode
        self._random_seed = random_seed

    def clone(self):
        print('\r>> clone: ', end='')
        cloned = RailEnvironmentPersistable(
            obs_builder_object_creator=self._obs_builder_object_creator,
            max_rails_between_cities=self._max_rails_between_cities,
            max_rails_in_city=self._max_rails_in_city,
            malfunction_rate=self._malfunction_rate,
            n_cities=self._n_cities,
            number_of_agents=self._number_of_agents,
            grid_width=self._grid_width,
            grid_height=self._grid_height,
            grid_mode=self._grid_mode,
            random_seed=self._random_seed)
        return cloned

    def generate_and_persist_environments(self,
                                          generate_nbr_env: Union[int, None] = None,
                                          path: Union[str, None] = 'generated_envs'):
        self._generate_environment(generate_nbr_env, path)

    def load_environments_from_path(self, path: Union[str, None] = 'generated_envs'):
        self._load_generated(path)

    def _generate_environment(self, generate_nbr_env, path):
        if generate_nbr_env is not None and path is not None:
            for i in range(generate_nbr_env):
                self._random_seed += i
                self.raw_env.reset(regenerate_rail=True,
                                   regenerate_schedule=True,
                                   random_seed=self._random_seed)

                if not os.path.exists(path):
                    os.makedirs(path)
                fn = self._save_raw_env(path)
                ProgressBar.console_print(i, generate_nbr_env, info=fn)
            ProgressBar.console_print(100, 100, info='done.')

    def _load_generated(self, path):
        if not os.path.exists(path):
            return
        if path is not None:
            for file in os.listdir(path):
                if file.endswith(".pkl"):
                    self._loaded_env.append(os.path.join(path, file))
            self._loaded_env_itr = 0
            print('Load environments from disk. # ', len(self._loaded_env), ' loaded.')

    def reset(self):
        if len(self._loaded_env) > 0:
            filename = self._loaded_env[self._loaded_env_itr]
            state, info, loaded_env = self._cached_reset(filename)
            self._reset_cached_rail_env(loaded_env.raw_env)
            self._copy_attribute_from_env(loaded_env.raw_env)

            self._loaded_env_itr += 1
            if self._loaded_env_itr >= len(self._loaded_env):
                self._loaded_env_itr = 0

            return state, info
        state, info = self.raw_env.reset()
        return state, info

    def _reset_cached_rail_env(self, raw_env: RailEnv):
        # manual reset (loaded from cache)
        raw_env.reset_agents()
        raw_env._elapsed_steps = 0
        raw_env.dones["__all__"] = False

    def _copy_attribute_from_env(self, rail_env: RailEnv):
        '''
        Copy all class attribute and it's value from RailEnv to self.raw_env
        :param rail_env: The original agent created in the RailEnv
        '''
        for attribute, value in rail_env.__dict__.items():
            setattr(self.raw_env, attribute, value)

    @lru_cache(maxsize=1000)
    def _cached_reset(self, filename):
        env = self.clone()
        RailEnvPersister.load(env.raw_env, filename)
        state, info = env.raw_env.reset(False, False)
        return state, info, env

    def _save_raw_env(self, path):
        '''
        Save the given RailEnv environment as pickle
        '''

        filename = os.path.join(
            path, f"{self.raw_env.width}x{self.raw_env.height}x{self.raw_env.get_num_agents()}_{self._random_seed}.pkl"
        )
        RailEnvPersister.save(self.raw_env, filename)
        return filename

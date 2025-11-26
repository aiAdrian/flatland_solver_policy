import os
from functools import lru_cache
from typing import Union, Callable, List

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv

from environment.flatland.rail_env import RailEnvironment
from utils.progress_bar import ProgressBar
from flatland.envs import malfunction_generators as mal_gen

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
                 random_seed=25041978,
                 silent=False,
                 disable_mal_functions=False):
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
            random_seed=random_seed,
            silent=silent
        )
        self._disable_mal_functions = disable_mal_functions
        self._random_seed = random_seed
        self._loaded_env = []
        self._loaded_env_itr = 0
        self._silent = silent
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

    def _clone(self, info_str: Union[str, None],
               number_of_agents: Union[int, None] = None,
               random_seed: Union[int, None] = None):
        #if info_str is not None:
        #    print(' -> cache:', info_str, end=' ') 
        return RailEnvironmentPersistable(
            obs_builder_object_creator=self._obs_builder_object_creator,
            max_rails_between_cities=self._max_rails_between_cities,
            max_rails_in_city=self._max_rails_in_city,
            malfunction_rate=self._malfunction_rate,
            n_cities=self._n_cities,
            number_of_agents=number_of_agents if number_of_agents is not None else self._number_of_agents,
            grid_width=self._grid_width,
            grid_height=self._grid_height,
            grid_mode=self._grid_mode,
            random_seed=random_seed if random_seed is not None else self._random_seed,
            silent=True)

    def generate_and_persist_environments(self,
                                          generate_nbr_env: Union[int, None] = None,
                                          generate_agents_per_env: Union[List[int]] = [10],
                                          path: Union[str, None] = 'generated_envs',
                                          overwrite_existing=True):
        self._generate_environment(generate_nbr_env, generate_agents_per_env, path, overwrite_existing)

    def load_environments_from_path(self, path: Union[str, None] = 'generated_envs'):
        self._load_generated(path)

    def _generate_environment(self, generate_nbr_env: int,
                              generate_agents_per_env: List[int],
                              path: str,
                              overwrite_existing: bool):
        if generate_nbr_env is not None and path is not None:
            for i_nbr_agent, nbr_agent in enumerate(generate_agents_per_env):
                for itr_env in range(generate_nbr_env):
                    rnd_seed = self._random_seed + itr_env + i_nbr_agent * generate_nbr_env
                    fn = self._generate_file_name(path,
                                                  self.raw_env.width,
                                                  self.raw_env.height,
                                                  nbr_agent,
                                                  rnd_seed
                                                  )
                    if os.path.exists(fn) and not overwrite_existing:
                        continue
                    env = self._clone(info_str=None,
                                      number_of_agents=nbr_agent,
                                      random_seed=rnd_seed)
                    env.raw_env.reset(regenerate_rail=True,
                                      regenerate_schedule=True,
                                      random_seed=rnd_seed)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    fn = env._save_raw_env(path)
                    ProgressBar.console_print(itr_env +
                                              i_nbr_agent * generate_nbr_env,
                                              generate_nbr_env * len(generate_agents_per_env), info=fn)
            ProgressBar.console_print(100, 100, info='done.')

    def _load_generated(self, path):
        if not os.path.exists(path):
            return
        if path is not None:
            file_found = []
            for file in os.listdir(path):
                if file.endswith(".pkl"):
                    file_path = os.path.join(path, file)
                    mod_time = os.path.getmtime(file_path)
                    file_found.append((file_path, mod_time))
            file_found.sort(key=lambda x: x[1])

            for filename, mod_time in file_found:
                self._loaded_env.append(filename)

            self._loaded_env_itr = 0
            print('Load environments from disk. # ', len(self._loaded_env), ' loaded.')

    def get_nbr_loaded_envs(self):
        return len(self._loaded_env)

    def reset(self):
        if len(self._loaded_env) > 0:
            filename = self._loaded_env[self._loaded_env_itr]
            state, info, loaded_env = self._cached_reset(filename)
            self._reset_cached_rail_env(loaded_env.raw_env)
            self._copy_attribute_from_env(loaded_env.raw_env)
 
            loaded_env.raw_env.obs_builder.set_env(self.raw_env)
            loaded_env.raw_env.obs_builder.reset()

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
        env = self._clone(filename)
        RailEnvPersister.load(env.raw_env, filename)

        if not self._silent:
            print(filename)
        if self._disable_mal_functions:
            env.raw_env.malfunction_generator =  mal_gen.NoMalfunctionGen()
            env.raw_env.malfunction_process_data = env.raw_env.malfunction_generator.get_process_data()

        state, info = env.raw_env.reset(False, False)
        return state, info, env

    def _generate_file_name(self, path: str,
                            width: int,
                            height: int,
                            nbr_agents: int,
                            rnd_seed: int) -> str:
        return os.path.join(path, '{:04d}x{:04d}x{:04d}_{:09d}.pkl'.format(width, height, nbr_agents, rnd_seed))

    def _save_raw_env(self, path):
        '''
        Save the given RailEnv environment as pickle
        '''
        filename = self._generate_file_name(path,
                                            self.raw_env.width,
                                            self.raw_env.height,
                                            self.raw_env.get_num_agents(),
                                            self._random_seed)
        RailEnvPersister.save(self.raw_env, filename=filename)
        return filename

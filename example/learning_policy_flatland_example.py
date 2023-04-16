from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.dead_lock_avoidance_agent import \
    DeadlockAvoidanceObservation, DeadLockAvoidanceAgent
from policy.policy import Policy
from solver.flatland_solver import FlatlandSolver


def create_flatland_env(max_rails_between_cities=2,
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
        obs_builder_object=DeadlockAvoidanceObservation()
    )

def create_environment():
    environment = create_flatland_env()
    observation_space = None
    action_space = environment.action_space
    return environment, observation_space, action_space


def create_deadlock_avoidance_policy(rail_env: RailEnv, action_space: int) -> Policy:
    return DeadLockAvoidanceAgent(rail_env, action_space, enable_eps=False, show_debug_plot=False)


if __name__ == "__main__":
    env, obs_space, act_space = create_environment()

    solver = FlatlandSolver(env)
    solver.make_renderer(None)
    # solver.activate_rendering()
    solver.set_policy(create_deadlock_avoidance_policy(env, act_space))
    solver.do_training(max_episodes=1000)

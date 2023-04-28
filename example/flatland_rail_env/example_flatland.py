from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

from example.flatland_rail_env.flatland_simple_renderer import FlatlandSimpleRenderer
from observation.flatland.flatten_tree_observation_for_rail_env.flatten_tree_observation_for_rail_env import \
    FlattenTreeObsForRailEnv
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQN_Param, DDDQNPolicy
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy
from policy.policy import Policy
from solver.flatland.flatland_solver import FlatlandSolver


def create_flatland_env(obs_builder_object: ObservationBuilder,
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


def create_environment(obs_builder_object: ObservationBuilder, number_of_agents: int):
    environment = create_flatland_env(obs_builder_object, number_of_agents=number_of_agents)
    action_space = environment.action_space[0]

    obs_states, _ = environment.reset()
    observation_space = len(obs_states[0])

    return environment, observation_space, action_space


def create_deadlock_avoidance_policy(rail_env: RailEnv, action_space: int, show_debug_plot=False) -> Policy:
    return DeadLockAvoidancePolicy(rail_env, action_space, enable_eps=False, show_debug_plot=show_debug_plot)


def create_dddqn_policy(observation_space: int, action_space: int) -> Policy:
    param = DDDQN_Param(hidden_size=128,
                        buffer_size=10_000,
                        batch_size=1024,
                        update_every=10,
                        learning_rate=0.5e-3,
                        tau=1.e-3,
                        gamma=0.95,
                        buffer_min_size=0,
                        use_gpu=False)

    return DDDQNPolicy(observation_space, action_space, param)


def create_ppo_policy(observation_space: int, action_space: int) -> Policy:
    return PPOPolicy(observation_space, action_space, True)


if __name__ == "__main__":
    observation_builder = FlattenTreeObsForRailEnv(
        max_depth=3,
        predictor=ShortestPathPredictorForRailEnv(max_depth=50)
    )

    env, obs_space, act_space = create_environment(obs_builder_object=observation_builder, number_of_agents=10)
    print('{} : agents: {:3} actions: {:3} obs_space: {:4}'.format(env.__class__.__name__,
                                                                   env.get_num_agents(), act_space, obs_space))

    solver = FlatlandSolver(env, create_deadlock_avoidance_policy(env, act_space), FlatlandSimpleRenderer(env))
    solver.do_training(max_episodes=10)

    solver = FlatlandSolver(env, create_dddqn_policy(obs_space, act_space))
    solver.do_training(max_episodes=10)

    solver = FlatlandSolver(env, create_ppo_policy(obs_space, act_space))
    solver.do_training(max_episodes=10)

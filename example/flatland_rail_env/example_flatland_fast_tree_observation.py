from typing import List, Callable

import numpy as np
from flatland.envs.step_utils.states import TrainState

from environment.environment import Environment
from example.flatland_rail_env.flatland_rail_env_persister import RailEnvironmentPersistable
from observation.flatland.flatland_fast_tree_observation.flatland_fast_tree_observation import \
    FlatlandFastTreeObservation
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.policy import Policy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver, RewardList, TerminalList, InfoDict
from utils.training_evaluation_pipeline import create_td3_policy, create_ppo_policy, create_a2c_policy, \
    create_dddqn_policy, crate_random_policy

# Idead based on https://discourse.aicrowd.com/t/accelerate-the-learning-increase-agents-behavior-at-higher-speed/3838

def create_deadlock_avoidance_policy(environment: Environment,
                                     action_space: int,
                                     show_debug_plot=False) -> DeadLockAvoidancePolicy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(),
                                   action_space,
                                   enable_eps=False,
                                   show_debug_plot=show_debug_plot)


def flatland_reward_shaper(reward: RewardList, terminal: TerminalList, info: InfoDict, env: Environment) -> List[float]:
    distance_map = env.raw_env.distance_map.get()

    for i, agent in enumerate(env.raw_env.agents):
        reward[i] = 0.0
        if agent.state == TrainState.DONE:
            reward[i] = 0.0
        if terminal[i] and agent.state == TrainState.DONE and agent.arrival_time == env.raw_env._elapsed_steps:
            reward[i] = 1.0

        if agent.position is not None:
            r = distance_map[i, agent.position[0], agent.position[1], agent.direction]
            if np.isinf(r):
                r = 10000000000
            if np.isnan(r):
                r = 10000000000
            r = max(1, r)
            r = 1 / r
            r = r / 10000
            reward[i] += r
        reward[i] /= env.get_num_agents()
    return reward


if __name__ == "__main__":
    environment = RailEnvironmentPersistable(
        obs_builder_object_creator=FlatlandFastTreeObservation,
        grid_width=30,
        grid_height=40,
        grid_mode=True,
        number_of_agents=10)
    environment.generate_and_persist_environments(generate_nbr_env=5,
                                                  generate_agents_per_env=[1, 2, 3, 5],  # [1, 2, 3, 5, 10, 20, 30, 50],
                                                  overwrite_existing=False)
    environment.load_environments_from_path()

    do_rendering = False
    policy_creator_list: List[Callable[[int, int], Policy]] = [create_td3_policy,
                                                               create_a2c_policy,
                                                               create_ppo_policy,
                                                               create_dddqn_policy,
                                                               crate_random_policy]
    for pcl in [policy_creator_list[0]]:
        solver = FlatlandSolver(environment,
                                pcl(environment.get_observation_space(), environment.get_action_space()),
                                FlatlandSimpleRenderer(environment) if do_rendering else None)
        solver.set_reward_shaper(flatland_reward_shaper)
        # solver.load_policy()
        solver.perform_training(max_episodes=5000)

    solver_deadlock = FlatlandSolver(environment,
                                     create_deadlock_avoidance_policy(environment, environment.get_action_space()),
                                     FlatlandSimpleRenderer(environment) if do_rendering else None)
    solver_deadlock.perform_training(max_episodes=1000)

from typing import List, Union

import numpy as np
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser

from environment.environment import Environment
from example.flatland_rail_env.flatland_rail_env_persister import RailEnvironmentPersistable
from observation.flatland.flatland_tree_observation.flatland_tree_observation import FlatlandTreeObservation, \
    TreeObservationSearchStrategy, TreeObservationReturnType
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.learning_policy.learning_policy import LearningPolicy
from policy.learning_policy.ppo_policy.ppo_agent import PPO_Param, PPOPolicy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver
from solver.multi_agent_base_solver import RewardList, TerminalList, InfoDict


def create_obs_builder_object():
    return FlatlandTreeObservation(
        search_strategy=TreeObservationSearchStrategy.BreadthFirstSearch,
        observation_return_type=TreeObservationReturnType.Flatten,
        depth_limit=15,
        observation_depth_limit=3,
        observation_depth_limit_discount=0.95,
        activate_simplified=False,
        render_debug_tree=False)


class DecisionPointPPOPolicy(PPOPolicy):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 in_parameters: Union[PPO_Param, None] = None):
        super(DecisionPointPPOPolicy, self).__init__(state_size, action_size, in_parameters)
        self._env: Union[Environment, None] = None
        self.switchAnalyser: Union[RailroadSwitchAnalyser, None] = None

    def get_name(self):
        return self.__class__.__name__

    def reset(self, e: Environment):
        self._env = e
        self.switchAnalyser = RailroadSwitchAnalyser(e.raw_env)
        super(DecisionPointPPOPolicy, self).reset(e)

    @staticmethod
    def _get_agent_position_and_direction(agent: EnvAgent):
        position = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        return position, direction

    def act(self, handle: int, state, eps=0.):
        agent: EnvAgent = self._env.raw_env.agents[handle]
        position, direction = self._get_agent_position_and_direction(agent)
        agent_at_railroad_switch, agent_near_to_railroad_switch, \
            agent_at_railroad_switch_cell, agent_near_to_railroad_switch_cell = \
            self.switchAnalyser.check_agent_decision(position=position, direction=direction)

        # only if the agent is moving
        if not agent.state == TrainState.MOVING:
            # when the agent is moving and the agent is not at a decision point - best option is just move forward
            # near to all switches are important:
            # (1) fork
            #
            #                |   /  |[---][---][
            #                |  /   |
            #      [---][-A-]| /    |[---][---][--
            #      ->->->->-> switch >->->->->->
            #
            # (2) fusion
            #
            #      -][---][-B-]| \    |
            #                  |  \   |
            #      -][---][-C-]|   \  |[---][---
            #      ->->->->->-> switch >->->->->->
            #
            # A, B, C are cases where the agent should not just walk forward (due deadlock)
            # switch as well
            if not agent_at_railroad_switch and not agent_near_to_railroad_switch_cell:
                return RailEnvActions.MOVE_FORWARD

        return super(DecisionPointPPOPolicy, self).act(handle, state, eps)


def create_deadlock_avoidance_policy(environment: Environment,
                                     action_space: int,
                                     show_debug_plot=False) -> DeadLockAvoidancePolicy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(),
                                   action_space,
                                   enable_eps=False,
                                   show_debug_plot=show_debug_plot)


def create_ppo_policy(observation_space: int, action_space: int) -> LearningPolicy:
    print('>> create_ppo_policy')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    ppo_param = PPO_Param(hidden_size=64,
                          buffer_size=16_000,
                          buffer_min_size=0,
                          batch_size=512,
                          learning_rate=0.5e-4,
                          discount=0.95,
                          use_replay_buffer=True,
                          use_gpu=True)
    return DecisionPointPPOPolicy(observation_space, action_space, ppo_param)


global reward_signal_updated
reward_signal_updated = None


def flatland_reward_shaper(reward: RewardList, terminal: TerminalList, info: InfoDict, env: Environment) -> List[float]:
    global reward_signal_updated
    distance_map = env.raw_env.distance_map.get()
    if env.raw_env._elapsed_steps < 5:
        reward_signal_updated = None
    if reward_signal_updated is None:
        reward_signal_updated = np.zeros(len(env.raw_env.agents))
    for i, agent in enumerate(env.raw_env.agents):
        reward[i] = 0.0
        if terminal[i] and \
                reward_signal_updated[i] == 0 and \
                agent.state == TrainState.DONE and \
                env.raw_env._elapsed_steps < (env.raw_env._max_episode_steps - 5):
            reward[i] = 1.0
            reward_signal_updated[i] = 1.0
        elif reward_signal_updated[i] == 0.0 and terminal[i]:
            reward[i] = -1.0
            reward_signal_updated[i] = 1.0
        else:
            reward[i] -= 0.001

    return reward


if __name__ == "__main__":
    environment = RailEnvironmentPersistable(
        obs_builder_object_creator=create_obs_builder_object,
        grid_width=30,
        grid_height=40,
        grid_mode=True,
        number_of_agents=10)
    environment.generate_and_persist_environments(generate_nbr_env=10,
                                                  generate_agents_per_env=[1, 2, 3, 5],  # [1, 2, 3, 5, 10, 20, 30, 50],
                                                  overwrite_existing=False)
    environment.load_environments_from_path()

    do_rendering = False

    solver = FlatlandSolver(environment,
                            create_ppo_policy(environment.get_observation_space(), environment.get_action_space()),
                            FlatlandSimpleRenderer(environment) if do_rendering else None)
    solver.set_reward_shaper(flatland_reward_shaper)
    # solver.load_policy()
    solver.perform_training(max_episodes=5000, checkpoint_interval=environment.get_nbr_loaded_envs())

    solver_deadlock = FlatlandSolver(environment,
                                     create_deadlock_avoidance_policy(environment, environment.get_action_space()),
                                     FlatlandSimpleRenderer(environment) if do_rendering else None)
    solver_deadlock.perform_training(max_episodes=2 * environment.get_nbr_loaded_envs())  # ~11min

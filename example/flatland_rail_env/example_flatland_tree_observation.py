from typing import Callable, Type, List, Union

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
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver
from solver.multi_agent_base_solver import RewardList, TerminalList, InfoDict
from policy.policy import Policy
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy, PPO_Param
from utils.training_evaluation_pipeline import policy_creator_list
from utils.training_evaluation_pipeline import create_td3_policy, create_a2c_policy, create_ppo_policy, create_dddqn_policy, create_random_policy
from utils.training_evaluation_pipeline import policy_creator_list
from utils.training_evaluation_pipeline import create_td3_policy, create_a2c_policy, create_ppo_policy, create_dddqn_policy, create_random_policy

# Enforce disable GPU
import torch
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_obs_builder_object():
    return FlatlandTreeObservation(
        search_strategy=TreeObservationSearchStrategy.BreadthFirstSearch,
        observation_return_type=TreeObservationReturnType.Flatten,
        depth_limit=50,
        observation_depth_limit=2,
        observation_depth_limit_discount=0.25,
        activate_simplified=False,
        render_debug_tree=False)

def create_deadlock_avoidance_policy(environment: Environment,
                                     action_space: int,
                                     show_debug_plot=False) -> DeadLockAvoidancePolicy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(),
                                   action_space,
                                   enable_eps=False,
                                   show_debug_plot=show_debug_plot)

class DecisionPointPPOPolicy(PPOPolicy):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 in_parameters: Union[PPO_Param, None] = None,
                 use_deadlock_avoidance_policy = False):
        self.deadlock_avoidance_policy = None
        self.use_deadlock_avoidance_policy = use_deadlock_avoidance_policy
        super(DecisionPointPPOPolicy, self).__init__(state_size, action_size, in_parameters)
        self._env: Union[Environment, None] = None
        self.switchAnalyser: Union[RailroadSwitchAnalyser, None] = None

    def get_name(self):
        if self.use_deadlock_avoidance_policy:
            return self.__class__.__name__ + "_DLA"
        return self.__class__.__name__


    def step(self, handle, state, action, reward, next_state, done):
        super(DecisionPointPPOPolicy, self).step(handle, state, action, reward, next_state, done)
        if self.use_deadlock_avoidance_policy:
            if self.deadlock_avoidance_policy is not None:
                self.deadlock_avoidance_policy.step(handle, state, action, reward, next_state, done)

    def start_step(self, train):
        super(DecisionPointPPOPolicy, self).start_step(train)
        if self.use_deadlock_avoidance_policy:
            self.deadlock_avoidance_policy.start_step(train)

    def reset(self, env: Environment):
        self._env = env
        self.switchAnalyser = RailroadSwitchAnalyser(env.raw_env)
        super(DecisionPointPPOPolicy, self).reset(env)
        if self.use_deadlock_avoidance_policy:
            if self.deadlock_avoidance_policy is None:
                self.deadlock_avoidance_policy = create_deadlock_avoidance_policy(env, self.action_size)
            self.deadlock_avoidance_policy.reset(env)


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
        if agent.state == TrainState.MOVING:
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

        action = super(DecisionPointPPOPolicy, self).act(handle, state, eps)
        if self.use_deadlock_avoidance_policy:
            if agent.state.is_on_map_state():
                if action == RailEnvActions.MOVE_FORWARD:
                    dla_action = self.deadlock_avoidance_policy.act(handle, state, eps)
                    return dla_action
                    # if RailEnvActions.STOP_MOVING == dla_action:
                    #     return RailEnvActions.STOP_MOVING

        return action

def create_dp_ppo_policy(observation_space: int, action_space: int) -> LearningPolicy:
    print('>> create_ppo_policy')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    ppo_param = PPO_Param(hidden_size=128,
                          buffer_size=16_000,
                          buffer_min_size=0,
                          batch_size=512,
                          learning_rate=0.5e-4,
                          discount=0.75,
                          use_replay_buffer=True,
                          use_gpu=True)
    return DecisionPointPPOPolicy(observation_space, action_space, ppo_param, False)


def create_dp_ppo_policy_dla(observation_space: int, action_space: int) -> LearningPolicy:
    print('>> create_ppo_policy_dla')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    ppo_param = PPO_Param(hidden_size=128,
                          buffer_size=16_000,
                          buffer_min_size=0,
                          batch_size=512,
                          learning_rate=0.5e-4,
                          discount=0.75,
                          use_replay_buffer=True,
                          use_gpu=True)
    return DecisionPointPPOPolicy(observation_space, action_space, ppo_param, True)


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




policy_creator_list: List[Callable[[int, int], Policy]] = [
                                                           create_td3_policy,   #0
                                                           create_a2c_policy,   #1
                                                           create_ppo_policy,   #2
                                                           create_dddqn_policy, #3
                                                           create_dp_ppo_policy, #4
                                                           create_dp_ppo_policy_dla, #5
                                                           create_random_policy #6
                                                           ]




if __name__ == "__main__":
    do_rendering = False
    do_training = True

    environment = RailEnvironmentPersistable(
        obs_builder_object_creator=create_obs_builder_object,
        grid_width=30,
        grid_height=40,
        grid_mode=True,
        number_of_agents=10)
    environment.generate_and_persist_environments(generate_nbr_env=10,
                                                  generate_agents_per_env=[1, 1, 2, 2, 5],  # [1, 2, 3, 5, 10, 20, 30, 50],
                                                  overwrite_existing=False)
    environment.load_environments_from_path()

    solver_deadlock = FlatlandSolver(environment,
                                     create_deadlock_avoidance_policy(environment, environment.get_action_space()),
                                     FlatlandSimpleRenderer(environment) if do_rendering else None)
    solver_deadlock.perform_training(max_episodes=200)


    if do_training:
        for pcl in policy_creator_list:
            solver = FlatlandSolver(environment,
                                    pcl(environment.get_observation_space(), environment.get_action_space()),
                                    FlatlandSimpleRenderer(environment) if do_rendering else None)
            solver.set_reward_shaper(flatland_reward_shaper)
            # solver.load_policy()
            solver.perform_training(max_episodes=2000)


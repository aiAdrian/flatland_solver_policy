from typing import List, Union

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from environment.environment import Environment
from example.flatland_rail_env.flatland_rail_env_persister import RailEnvironmentPersistable
from observation.flatland.flatland_tree_observation.flatland_tree_observation import FlatlandTreeObservation, \
    TreeObservationReturnType, TreeObservationSearchStrategy
from policy.heuristic_policy.heuristic_policy import HeuristicPolicy
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.learning_policy.learning_policy import LearningPolicy
from policy.learning_policy.ppo_policy.ppo_agent import PPO_Param, PPOPolicy
from policy.learning_policy.reinforce_heuristic_policy.reinforce_heuristic_policy import ReinforceHeuristicPolicy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver, RewardList, TerminalList, InfoDict


def create_deadlock_avoidance_policy(environment: Environment,
                                     action_space: int,
                                     show_debug_plot=False) -> DeadLockAvoidancePolicy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(),
                                   action_space,
                                   enable_eps=False,
                                   show_debug_plot=show_debug_plot)


def flatland_reward_shaper(reward: RewardList, terminal: TerminalList, info: InfoDict, env: Environment) -> List[float]:
    for i, agent in enumerate(env.raw_env.agents):
        reward[i] = 0.0
        if agent.state == TrainState.DONE:
            reward[i] = 0.0
        if terminal[i] and agent.state == TrainState.DONE and agent.arrival_time == env.raw_env._elapsed_steps:
            reward[i] = 1.0
    return reward


class MyReinforceHeuristicPolicy(ReinforceHeuristicPolicy):

    def __init__(self,
                 learning_policy: LearningPolicy,
                 heuristic_policy: HeuristicPolicy,
                 heuristic_policy_epsilon=0.1):
        super(MyReinforceHeuristicPolicy, self).__init__(learning_policy=learning_policy,
                                                         heuristic_policy=heuristic_policy,
                                                         heuristic_policy_epsilon=heuristic_policy_epsilon)
        self._env: Union[Environment, None] = None

    def reset(self, e: Environment):
        self._env = e

    def act(self, handle: int, state, eps=0.):
        tree_obs: FlatlandTreeObservation = self._env.raw_env.obs_builder
        agent: EnvAgent = self._env.raw_env.agents[handle]
        p, d = tree_obs.get_agent_position_and_direction(agent)
        agent_at_railroad_switch, agent_near_to_railroad_switch, _, _ = \
            tree_obs.switchAnalyser.check_agent_decision(position=p, direction=d)
        if not agent_at_railroad_switch and not agent_near_to_railroad_switch and agent.position is not None:
            return RailEnvActions.MOVE_FORWARD
        return super(MyReinforceHeuristicPolicy, self).act(handle, state, eps)


class DecisionPointPPOPolicy(PPOPolicy):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 in_parameters: Union[PPO_Param, None] = None):
        super(DecisionPointPPOPolicy, self).__init__(state_size, action_size, in_parameters)
        self._env: Union[Environment, None] = None

    def get_name(self):
        return self.__class__.__name__

    def reset(self, e: Environment):
        self._env = e
        super(DecisionPointPPOPolicy, self).reset(env)

    def act(self, handle: int, state, eps=0.):
        tree_obs: FlatlandTreeObservation = self._env.raw_env.obs_builder
        agent: EnvAgent = self._env.raw_env.agents[handle]
        p, d = tree_obs.get_agent_position_and_direction(agent)
        agent_at_railroad_switch, agent_near_to_railroad_switch, _, _ = \
            tree_obs.switchAnalyser.check_agent_decision(position=p, direction=d)
        if not agent_at_railroad_switch and not agent_near_to_railroad_switch and agent.position is not None:
            return RailEnvActions.MOVE_FORWARD
        return super(DecisionPointPPOPolicy, self).act(handle, state, eps)


def create_decision_point_ppo_policy(observation_space: int, action_space: int) -> LearningPolicy:
    ppo_param = PPO_Param(hidden_size=256,
                          buffer_size=16_000,
                          buffer_min_size=0,
                          batch_size=128,
                          learning_rate=0.5e-3,
                          discount=0.95,
                          use_replay_buffer=True,
                          use_gpu=False)
    return DecisionPointPPOPolicy(observation_space, action_space, ppo_param)


def flatland_reward_shaper(reward: RewardList, terminal: TerminalList, info: InfoDict, env: Environment) -> List[float]:
    distance_map = env.distance_map.get()
    for i, agent in enumerate(env.raw_env.agents):
        reward[i] = 0.0
        if agent.state == TrainState.DONE:
            reward[i] = 0.0
        if terminal[i] and agent.state == TrainState.DONE and agent.arrival_time == env.raw_env._elapsed_steps:
            reward[i] = 1.0

        if agent.position is not None:
            r = distance_map[i, agent.position[0], agent.position[1], agent.direction]
            r = max(1, r)
            r = 1 / r
            r = r / 1000
            reward[i] += r

    return reward


if __name__ == "__main__":
    do_rendering = True
    do_render_debug_tree = True
    activate_simplified = True
    use_reinforced_heuristic_policy = True


    def create_obs_builder_object():
        return FlatlandTreeObservation(
            search_strategy=TreeObservationSearchStrategy.BreadthFirstSearch,
            observation_return_type=TreeObservationReturnType.Flatten,
            depth_limit=20,
            activate_simplified=activate_simplified,
            render_debug_tree=do_render_debug_tree and do_rendering)


    env = RailEnvironmentPersistable(
        obs_builder_object_creator=create_obs_builder_object,
        grid_width=30,
        grid_height=40,
        grid_mode=True,
        number_of_agents=10)
    env.generate_and_persist_environments(generate_nbr_env=5,
                                          generate_agents_per_env=[1, 2, 3, 5],  # [1, 2, 3, 5, 10, 20, 30, 50],
                                          overwrite_existing=False)
    env.load_environments_from_path()

    if use_reinforced_heuristic_policy:
        # policy = MyReinforceHeuristicPolicy(
        #     learning_policy=create_dddqn_policy(env.get_observation_space(), env.get_action_space()),
        #     heuristic_policy=create_deadlock_avoidance_policy(env, env.get_action_space()),
        #     heuristic_policy_epsilon=0.5
        # )
        policy = create_decision_point_ppo_policy(env.get_observation_space(), env.get_action_space())
        max_episodes = 5000
        max_episodes_eval = 120
    else:
        policy = create_deadlock_avoidance_policy(env, env.get_action_space())
        max_episodes = 120
        max_episodes_eval = 120

    solver = FlatlandSolver(env,
                            policy,
                            FlatlandSimpleRenderer(env) if do_rendering else None)
    solver.load_policy()
    solver.set_reward_shaper(flatland_reward_shaper)
    solver.perform_training(max_episodes=max_episodes)
    solver.perform_evaluation(max_episodes=max_episodes_eval)

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
from policy.learning_policy.reinforce_heuristic_policy.reinforce_heuristic_policy import ReinforceHeuristicPolicy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver, RewardList, TerminalList, InfoDict
from utils.training_evaluation_pipeline import create_ppo_policy


def create_deadlock_avoidance_policy(environment: Environment,
                                     action_space: int,
                                     show_debug_plot=False) -> DeadLockAvoidancePolicy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(), action_space, enable_eps=False,
                                   show_debug_plot=show_debug_plot)


def flatland_reward_shaper(reward: RewardList, terminal: TerminalList, info: InfoDict, env: Environment) -> List[float]:
    for i, agent in enumerate(env.raw_env.agents):
        reward[i] = -0.000001
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


if __name__ == "__main__":
    do_rendering = False
    do_render_debug_tree = False
    use_reinforced_heuristic_policy = False


    def create_obs_builder_object():
        return FlatlandTreeObservation(
            search_strategy=TreeObservationSearchStrategy.BreadthFirstSearch,
            observation_return_type=TreeObservationReturnType.Flatten,
            depth_limit=3,
            render_debug_tree=do_render_debug_tree and do_rendering)


    env = RailEnvironmentPersistable(
        obs_builder_object_creator=create_obs_builder_object,
        grid_width=30,
        grid_height=40,
        grid_mode=True,
        number_of_agents=10)
    env.generate_and_persist_environments(generate_nbr_env=5,
                                          generate_agents_per_env=[1, 2, 3, 5, 10, 20, 30, 50])
    env.load_environments_from_path()

    if use_reinforced_heuristic_policy:
        policy = MyReinforceHeuristicPolicy(
            learning_policy=create_ppo_policy(env.get_observation_space(), env.get_action_space()),
            heuristic_policy=create_deadlock_avoidance_policy(env, env.get_action_space()),
            heuristic_policy_epsilon=1.0
        )
    else:
        policy = create_deadlock_avoidance_policy(env, env.get_action_space())
    solver = FlatlandSolver(env,
                            policy,
                            FlatlandSimpleRenderer(env) if do_rendering else None)
    solver.set_reward_shaper(flatland_reward_shaper)
    solver.perform_training(max_episodes=120)
    solver.perform_evaluation(max_episodes=120)

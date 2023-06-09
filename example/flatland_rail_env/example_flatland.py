from environment.environment import Environment
from environment.flatland.rail_env import RailEnvironment
from observation.flatland.dummy_observation import FlatlandDummyObservation
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.policy import Policy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver


def create_deadlock_avoidance_policy(environment: Environment, action_space: int, show_debug_plot=False) -> Policy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(), action_space, enable_eps=False,
                                   show_debug_plot=show_debug_plot)


if __name__ == "__main__":
    do_rendering = True

    observation_builder = FlatlandDummyObservation()

    env = RailEnvironment(obs_builder_object=observation_builder, number_of_agents=25)
    solver_deadlock = FlatlandSolver(env,
                                     create_deadlock_avoidance_policy(env, env.get_action_space()),
                                     FlatlandSimpleRenderer(env) if do_rendering else None)
    solver_deadlock.perform_training(max_episodes=10)

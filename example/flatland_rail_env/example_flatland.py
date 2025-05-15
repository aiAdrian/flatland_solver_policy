from example.flatland_rail_env.flatland_rail_env_persister import RailEnvironmentPersistable

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
    do_rendering = False

    # observation_builder = FlatlandDummyObservation()
    # environment = RailEnvironment(obs_builder_object=observation_builder, number_of_agents=25)

    environment = RailEnvironmentPersistable(
        obs_builder_object_creator=FlatlandDummyObservation,
        grid_width=30,
        grid_height=40,
        grid_mode=True,
        silent=True,
        number_of_agents=10,
        disable_mal_functions=False)
    environment.generate_and_persist_environments(generate_nbr_env=5,
                                                  generate_agents_per_env=[1, 2, 3, 5, 10, 20, 30, 50],
                                                  overwrite_existing=False)
    environment.load_environments_from_path()


    solver_deadlock = FlatlandSolver(environment,
                                     create_deadlock_avoidance_policy(environment, environment.get_action_space()),
                                     FlatlandSimpleRenderer(environment) if do_rendering else None)
    solver_deadlock.perform_training(max_episodes=40)

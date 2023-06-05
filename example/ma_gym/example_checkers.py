from environment.ma_gym.multi_agent_gym_environment import MultiAgentGymEnvironment
from rendering.ma_gym.multi_agent_gym_renderer import MultiAgentGymRenderer
from solver.ma_gym.multi_agent_gym_solver import MultiAgentGymSolver
from utils.training_evaluation_pipeline import execute_policy_comparison, create_dddqn_policy

environment = MultiAgentGymEnvironment(env_to_load='ma_gym:Checkers-v0')
execute_policy_comparison(environment, MultiAgentGymSolver)

renderer = MultiAgentGymRenderer(environment)
renderer.set_sleep_time(0.1)

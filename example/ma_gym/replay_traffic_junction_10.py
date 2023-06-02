from environment.ma_gym.multi_agent_gym_environment import MultiAgentGymEnvironment
from rendering.ma_gym.multi_agent_gym_renderer import MultiAgentGymRenderer
from solver.ma_gym.multi_agent_gym_solver import MultiAgentGymSolver
from utils.training_evaluation_pipeline import execute_replay_policy_comparison

environment = MultiAgentGymEnvironment(env_to_load='ma_gym:TrafficJunction10-v0')
execute_replay_policy_comparison(environment, MultiAgentGymSolver, MultiAgentGymRenderer(environment))

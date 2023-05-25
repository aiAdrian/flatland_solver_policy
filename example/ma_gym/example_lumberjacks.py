from example.ma_gym.integration import MultiAgentGymEnvironment
from example.ma_gym.training_evaluation_pipeline import experimental_training_evaluation_pipeline

environment = MultiAgentGymEnvironment(env_to_load='ma_gym:Lumberjacks-v0')
experimental_training_evaluation_pipeline(environment)

import sys
import os

BASE_DIR = "/home/u216993/workspace/ai4realnet/aiAdrian_flatland"
sys.path.append(os.path.join(BASE_DIR, "flatland-rl"))
sys.path.append(os.path.join(BASE_DIR, "flatland_railway_extension"))
sys.path.append(BASE_DIR)

from flatland_solver_policy.example.flatland_rail_env.marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator

def test_tree_structure():
    n_agents = 1
    env = RailEnv(
        width=30, height=30,
        rail_generator=sparse_rail_generator(max_num_cities=2, grid_mode=False, max_rails_between_cities=2),
        line_generator=sparse_line_generator(),
        number_of_agents=n_agents
    )
    env.reset()
    
    observation_builder = DecisionPointObservation(search_depth=5)
    observation_builder.set_env(env)
    
    agent_handle = 0
    agent = env.agents[agent_handle]
    pos = agent.position if agent.position is not None else agent.initial_position
    dir = agent.direction if agent.direction is not None else agent.initial_direction
    
    tree = observation_builder._local_search(
        handle=agent_handle,
        start_pos=pos,
        start_dir=dir,
        depth_limit=5
    )
    
    print(f"Type of tree: {type(tree)}")
    if tree and len(tree) > 0:
        print(f"First element type: {type(tree[0])}")
        print(f"First element: {tree[0]}")
        
    if isinstance(tree, dict):
        # Maybe it returns a dict of nodes?
        nodes = list(tree.values())
        print(f"Total nodes: {len(nodes)}")
        node = nodes[0]
        print(f"Node keys: {dir(node)}")
        print(f"type: {node.type}, type_name: {node.type_name}, decision_depth: {node.decision_depth}")

if __name__ == "__main__":
    test_tree_structure()

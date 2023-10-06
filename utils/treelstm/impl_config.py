"""
https://github.com/RoboEden/flatland-marl
"""


class FeatureParserConfig:
    # Fixed
    action_sz: int = 5
    state_sz: int = 7
    road_type_sz: int = 11
    transitions_sz: int = 4 * 4

    node_sz: int = 6

    agent_attr: int = 2


class NetworkConfig:
    hidden_sz = 128
    tree_embedding_sz = 128

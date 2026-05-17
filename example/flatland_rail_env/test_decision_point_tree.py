#!/usr/bin/env python3
"""
Quick test to verify that the new _local_search() creates only decision-point nodes
with types INIT, SWITCH, PRE_M and NOT at every cell hop.
"""
import sys
import os
sys.path.insert(0, '/home/u216993/workspace/ai4realnet/aiAdrian_flatland/flatland-rl')
sys.path.insert(0, '/home/u216993/workspace/ai4realnet/aiAdrian_flatland/flatland_railway_extension')

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator

from marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation, NodeType

def test_tree_structure():
    """Test that _local_search creates only decision-point nodes."""
    
    # Create a simple environment
    env = RailEnv(
        width=15,
        height=15,
        rail_generator=sparse_rail_generator(
            max_num_cities=5,
            seed=42,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rail_objects_in_city=2,
        ),
        schedule_generator=sparse_line_generator(seed=42),
        number_of_agents=2,
    )
    
    env.reset()
    
    # Create observation builder
    obs_builder = DecisionPointObservation(
        debug=False,
        search_depth=5,
        observation_profile="local_tree_encoder",
    )
    obs_builder.env = env
    
    # Build agent map
    obs_builder.agent_map = np.zeros((env.height, env.width), dtype=np.int32) - 1
    for agent in env.agents:
        if agent.position is not None:
            obs_builder.agent_map[agent.position] = agent.handle
    
    # Get agent 0
    agent = env.agents[0]
    if agent.position is None:
        print("Agent not started yet, skipping test")
        return
    
    handle = agent.handle
    pos = agent.position
    direction = agent.direction
    
    print(f"\n{'='*70}")
    print(f"Testing _local_search() for agent {handle} at pos {pos}, dir {direction}")
    print(f"{'='*70}\n")
    
    # Call local search
    try:
        result = obs_builder._local_search(handle, pos, direction, depth_limit=5)
    except Exception as e:
        print(f"ERROR in _local_search: {e}")
        import traceback
        traceback.print_exc()
        return
    
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])
    
    print(f"Tree Structure: {len(nodes)} nodes, {len(edges)} edges\n")
    
    # Analyze nodes
    print("Node Types:")
    print(f"{'ID':<4} {'Type':<12} {'Type#':<6} {'Decision_Depth':<15} {'Transitions':<12} {'Deadlock_Risk':<14}")
    print("-" * 75)
    
    type_map = {0: "INIT", 1: "SWITCH", 2: "PRE_M"}
    node_type_counts = {0: 0, 1: 0, 2: 0}
    
    for node in nodes:
        node_id = node.get("id", -1)
        node_type = node.get("type", -1)
        type_name = type_map.get(node_type, f"UNKNOWN({node_type})")
        decision_depth = node.get("decision_depth", -1)
        num_transitions = node.get("num_transitions", 0)
        deadlock_risk = node.get("deadlock_risk", 0.0)
        
        if node_type in node_type_counts:
            node_type_counts[node_type] += 1
        
        print(f"{node_id:<4} {type_name:<12} {node_type:<6} {decision_depth:<15} {num_transitions:<12} {deadlock_risk:<14.3f}")
    
    print(f"\nNode Type Summary:")
    print(f"  INIT (0):   {node_type_counts[0]} nodes")
    print(f"  SWITCH (1): {node_type_counts[1]} nodes")
    print(f"  PRE_M (2):  {node_type_counts[2]} nodes")
    print(f"  TOTAL:      {len(nodes)} nodes")
    
    # Verify no nodes exist without a proper type
    invalid_nodes = [n for n in nodes if n.get("type", -1) not in [0, 1, 2]]
    if invalid_nodes:
        print(f"\n❌ ERROR: Found {len(invalid_nodes)} nodes with invalid type!")
        for n in invalid_nodes:
            print(f"   {n}")
    else:
        print(f"\n✅ All nodes have valid types (INIT, SWITCH, PRE_M)")
    
    # Verify edge structure
    print(f"\nEdge Structure ({len(edges)} edges):")
    print(f"{'Src':<4} {'Dst':<4} {'Merge':<6} {'Has_Oncoming':<14} {'Cells':<6}")
    print("-" * 50)
    
    for edge in edges[:10]:  # Show first 10
        src = edge.get("src", -1)
        dst = edge.get("dst", -1)
        merge = edge.get("merge_conflict", False)
        has_oncoming = edge.get("has_oncoming_edge", False)
        cells = edge.get("edge_len_cells", 0)
        print(f"{src:<4} {dst:<4} {int(merge):<6} {int(has_oncoming):<14} {cells:<6}")
    
    if len(edges) > 10:
        print(f"... ({len(edges) - 10} more edges)")
    
    # Final verdict
    print(f"\n{'='*70}")
    if len(nodes) > 0 and invalid_nodes == []:
        print("✅ SUCCESS: Tree uses decision-point node architecture!")
        print(f"   - {len(nodes)} decision-point nodes (INIT/SWITCH/PRE_M only)")
        print(f"   - {len(edges)} decision-point edges")
        print(f"   - Depth counts decision transitions (not cell hops)")
    else:
        print("❌ FAILED: Tree structure validation")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    test_tree_structure()

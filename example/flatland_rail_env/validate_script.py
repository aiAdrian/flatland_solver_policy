import sys
import os

base_path = "/home/u216993/workspace/ai4realnet/aiAdrian_flatland"
flatland_rl_path = os.path.join(base_path, "flatland-rl")
extension_path = os.path.join(base_path, "flatland_railway_extension")
solver_policy_path = os.path.join(base_path, "flatland_solver_policy")
obs_path = os.path.join(base_path, "flatland_solver_policy/example/flatland_rail_env")

sys.path.insert(0, flatland_rl_path)
sys.path.insert(0, extension_path)
sys.path.insert(0, solver_policy_path)
sys.path.insert(0, obs_path)

try:
    from marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation
    from marl_attention_temporal_mappo import TreePayloadEncoder
    
    print(f"DecisionPointObservation.BASE_OBS_SIZE: {DecisionPointObservation.BASE_OBS_SIZE}")
    print(f"TreePayloadEncoder.NODE_DIM: {TreePayloadEncoder.NODE_DIM}")
    print(f"TreePayloadEncoder.EDGE_DIM: {TreePayloadEncoder.EDGE_DIM}")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

import sys
import os

# Set up paths
base_path = "/home/u216993/workspace/ai4realnet/aiAdrian_flatland"
flatland_rl_path = os.path.join(base_path, "flatland-rl")
extension_path = os.path.join(base_path, "flatland_railway_extension")
obs_path = os.path.join(base_path, "flatland_solver_policy/example/flatland_rail_env/marl_attention_temporal_observation")

sys.path.insert(0, flatland_rl_path)
sys.path.insert(0, extension_path)
sys.path.insert(0, os.path.dirname(obs_path))

try:
    from marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation
    print("Successfully imported DecisionPointObservation")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Inspect the file content for keys directly to verify the structure as requested
file_path = "/home/u216993/workspace/ai4realnet/aiAdrian_flatland/flatland_solver_policy/example/flatland_rail_env/marl_attention_temporal_observation/decision_point_observation.py"

print("\nVerifying keys in source file:")
keys_to_check = [
    "deadlock_risk", 
    "deadlock_ahead", 
    "deadlock_hard_block", 
    "dst_deadlock_risk", 
    "dst_deadlock_hard_block"
]

with open(file_path, 'r') as f:
    content = f.read()
    for key in keys_to_check:
        if key in content:
            print(f"  [OK] Found '{key}'")
        else:
            print(f"  [MISSING] '{key}'")


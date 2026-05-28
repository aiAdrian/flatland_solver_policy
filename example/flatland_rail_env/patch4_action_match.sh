#!/bin/bash
set -e

OBS_FILE=$(find . -name "decision_point_observation.py" 2>/dev/null | head -1)
echo "📁 Patching: $OBS_FILE"

python3 << 'PYEOF'
import os, re

obs_path = os.popen("find . -name 'decision_point_observation.py' 2>/dev/null | head -1").read().strip()
with open(obs_path, 'r') as f:
    src = f.read()

# Find with regex (whitespace-tolerant)
pattern = re.compile(
    r'(\s*)prev_sp\s*=\s*self\._prev_sp_hint\.get\(handle\)\s*\n'
    r'(\s*)if\s+last_action_id\s*==\s*4:\s*\n'
    r'.*?'
    r'raw_features\[21\]\s*=\s*1\.0\s+if\s+float\(prev_sp\[sp_idx\]\)\s*>\s*0\.5\s+else\s+0\.0\s*\n'
    r'(\s*#[^\n]*\n)?',
    re.DOTALL
)

match = pattern.search(src)
if not match:
    print("❌ Could not locate action_matches_sp block via regex")
    print("   Showing surrounding context for manual fix:")
    idx = src.find("action_matches_sp")
    if idx > 0:
        # Find feature [21] block
        idx2 = src.find("raw_features[21]", idx)
        if idx2 > 0:
            print("─" * 60)
            print(src[idx2-200:idx2+400])
            print("─" * 60)
    exit(1)

indent = match.group(1)
new_block = (
    f"{indent}# FIX: Tri-state with neutral default for ambiguous cases.\n"
    f"{indent}prev_sp = self._prev_sp_hint.get(handle)\n"
    f"{indent}if last_action_id == 4:\n"
    f"{indent}    raw_features[21] = 0.5  # STOP — defensible yield\n"
    f"{indent}elif prev_sp is not None and last_action_id in (1, 2, 3):\n"
    f"{indent}    sp_idx = last_action_id - 1\n"
    f"{indent}    raw_features[21] = 1.0 if float(prev_sp[sp_idx]) > 0.5 else 0.0\n"
    f"{indent}else:\n"
    f"{indent}    raw_features[21] = 0.5  # neutral, not off-plan\n"
)

new_src = src[:match.start()] + new_block + src[match.end():]
with open(obs_path, 'w') as f:
    f.write(new_src)
print(f"✅ PATCH 4 applied to {obs_path}")
PYEOF

# Verify
python3 -c "
import sys; sys.path.insert(0, '.')
from marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation
print('  ✅ Import OK after PATCH 4')
"

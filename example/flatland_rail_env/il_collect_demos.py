"""
IL — Phase A: Demonstration Collection
======================================

Run the DeadLockAvoidancePolicy (Shortest-Path + Deadlock-Avoidance heuristic
from `policy.heuristic_policy.shortest_path_deadlock_avoidance_policy`) on the
same environment configuration used by the MARL training pipeline.

For every transition we capture a tuple

    (temporal_state, teacher_action, action_mask, handle, episode_id)

where:
    - `temporal_state` is the same T-step sequence the DeciderNetwork sees
    (T=3 wrapped HierarchicalRoutesObservation, 90D each).
    - `teacher_action` is the integer action the heuristic chose.
    - `action_mask` is the 5-D legal-action mask we ALSO use during PPO,
      derived from the rail transitions at the agent's current cell.
    - `handle` and `episode_id` are kept for diagnostics only.

The result is pickled to `il_demos.pkl`. The next phase
(`il_pretrain.py`) consumes that file and trains the DeciderNetwork via
masked cross-entropy (Behavior Cloning).

Design notes
------------
* We deliberately reuse the EXACT same observation builder, environment factory
  and reward/curriculum scheme that PPO uses. That guarantees the BC checkpoint
  is "in distribution" for the subsequent PPO fine-tune.
* The DeadLockAvoidancePolicy is invoked through its standard solver-style API
  (`reset`, `start_step`, `act`). We do NOT use a `FlatlandSolver` here because
  we need direct control over the per-handle data capture loop.
* For the action mask we instantiate a *small* DeciderPPOPolicy purely so we
  can call `_legal_action_mask(handle)`. No optimizer, no buffer use.
"""

from __future__ import annotations

import os
import pickle
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

from flatland.envs.rail_env import RailEnvActions

from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import (
    DeadLockAvoidancePolicy,
)
from example.flatland_rail_env.flatland_rail_env_persister import (
    RailEnvironmentPersistable,
)

from marl_attention_temporal_observation.temporal_multi_agent_observation import (
    TemporalMultiAgentObservation,
)
from marl_attention_temporal_observation.hierarchical_routes_observation import (
    HierarchicalRoutesObservation,
)
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer

# We import the policy class only to reuse the action-mask helper. We
# never train it here.
from decider_policy import DeciderPPOPolicy


# ----------------------------------------------------------------------------
# Configuration mirrors marl_attention_temporal.py.
# ----------------------------------------------------------------------------
TEMPORAL_WINDOW = 3
PURE_MARL_GRID_WIDTH = 30
PURE_MARL_GRID_HEIGHT = 40
PURE_MARL_N_CITIES = 3
PURE_MARL_MAX_AGENTS = 5

# Same curriculum as the PPO trainer. Episode counts are typically lower for
# data collection -- the heuristic generalises across phases.
DEMO_PHASES = [
    {'name': 'phase1_solo', 'agent_counts': [2],    'num_envs': 12, 'episodes': 200},
    {'name': 'phase2_easy', 'agent_counts': [2, 3], 'num_envs': 16, 'episodes': 200},
    {'name': 'phase3_mid',  'agent_counts': [3, 4], 'num_envs': 18, 'episodes': 250},
    {'name': 'phase4_hard', 'agent_counts': [4, 5], 'num_envs': 22, 'episodes': 300},
]

CURRICULUM_BASE_PATH = 'generated_envs_il'
DEMO_OUTPUT_FILE = 'il_demos.pkl'
MAX_STEPS_PER_EPISODE = 200  # Flatland default truncation guard
# Toggle rendering directly in code.
ENABLE_RENDERING = False
RENDER_EACH_EPISODE = 1


# ----------------------------------------------------------------------------
# Observation factory — must match marl_attention_temporal.py exactly.
# ----------------------------------------------------------------------------
def create_temporal_obs_builder_object():
    base = HierarchicalRoutesObservation()
    return TemporalMultiAgentObservation(
        temporal_window=TEMPORAL_WINDOW,
        base_obs=base,
    )


# ----------------------------------------------------------------------------
# Episode driver.
# ----------------------------------------------------------------------------
def _run_one_episode(
    environment,
    teacher: DeadLockAvoidancePolicy,
    mask_helper: DeciderPPOPolicy,
    renderer,
    episode_id: int,
) -> Tuple[List[Tuple[Any, int, np.ndarray, int, int]], int, int]:
    """
    Run a single episode under the heuristic teacher and capture every
    decision. Returns (transitions, num_done, num_agents).
    """
    state, info = environment.reset()
    teacher.reset(environment)
    if renderer is not None:
        renderer.reset()
    # The DeciderPPOPolicy._legal_action_mask uses self._env to access the
    # raw RailEnv. We mirror what the solver normally does in solver.reset.
    mask_helper._env = environment

    transitions: List[Tuple[Any, int, np.ndarray, int, int]] = []

    raw_env = environment.get_raw_env()
    num_agents = raw_env.get_num_agents()
    handles = list(range(num_agents))
    teacher.start_episode(train=False)
    for step in range(raw_env._max_episode_steps):
        teacher.start_step(train=False)

        actions: Dict[int, int] = {}
        update_values: Dict[int, bool] = {}
        for handle in handles:
            agent = raw_env.agents[handle]
            action_required = bool(info['action_required'][handle])
            # We only record decisions for agents that are actually on the
            # map or about to depart -- otherwise the action is a no-op
            # filler that would distort the BC distribution.
            record_this_handle = (
                agent.state.is_on_map_state()
                or agent.state.name == 'READY_TO_DEPART'
            )

            if action_required:
                update_values[handle] = True
                mask = mask_helper._legal_action_mask(handle)
                t_action = int(teacher.act(handle, state[handle], 0.0))

                # Safety: if the teacher proposes an action that is illegal under
                # the mask we use during PPO, fall back to the highest-priority
                # legal action so the BC target is consistent with the masked
                # softmax later. This rarely fires in practice.
                if mask[t_action] < 0.5:
                    legal_idx = np.where(mask > 0.5)[0]
                    if len(legal_idx) > 0:
                        # Prefer movement actions in fallback ordering.
                        pref_order = [
                            int(RailEnvActions.MOVE_FORWARD),
                            int(RailEnvActions.MOVE_LEFT),
                            int(RailEnvActions.MOVE_RIGHT),
                            int(RailEnvActions.STOP_MOVING),
                            int(RailEnvActions.DO_NOTHING),
                        ]
                        fallback = next((a for a in pref_order if mask[a] > 0.5),
                                        int(legal_idx[0]))
                        t_action = fallback

                if record_this_handle:
                    transitions.append(
                        (state[handle], int(t_action), mask.astype(np.float32),
                         int(handle), int(episode_id))
                    )
            else:
                # Mirror FlatlandSolver semantics when no action is required.
                update_values[handle] = False
                t_action = int(RailEnvActions.DO_NOTHING)

            actions[handle] = t_action
        
        teacher.end_step(train=False)

        next_state, reward, terminal, info = environment.step(actions)
        terminal_all = bool(terminal['__all__'])
        for handle in handles:
            if update_values[handle] or terminal_all:
                teacher.step(
                    handle,
                    state[handle],
                    actions[handle],
                    reward[handle],
                    next_state[handle],
                    terminal[handle],
                )
        state = next_state
        if renderer is not None:
            renderer.render(episode_id + 1, step + 1, terminal_all)
        if terminal_all:
            break
    
    teacher.end_episode(train=False)
    num_done = int(sum(1 for h in handles if raw_env.agents[h].state.name == 'DONE'))
    return transitions, num_done, num_agents


def _ensure_phase_envs(environment, phase: Dict[str, Any]) -> None:
    """Generate (or regenerate) the persisted environments for one phase."""
    phase_path = f"{CURRICULUM_BASE_PATH}/{phase['name']}"
    environment.generate_and_persist_environments(
        generate_nbr_env=phase['num_envs'],
        generate_agents_per_env=phase['agent_counts'],
        path=phase_path,
        overwrite_existing=True,
    )
    environment._loaded_env = []
    environment._loaded_env_itr = 0
    environment.load_environments_from_path(path=phase_path)


def collect() -> Dict[str, Any]:
    print("=" * 80)
    print("IL — Demonstration Collection (DeadLockAvoidancePolicy teacher)")
    print("=" * 80)

    environment = RailEnvironmentPersistable(
        obs_builder_object_creator=create_temporal_obs_builder_object,
        n_cities=PURE_MARL_N_CITIES,
        grid_width=PURE_MARL_GRID_WIDTH,
        grid_height=PURE_MARL_GRID_HEIGHT,
        grid_mode=True,
        number_of_agents=PURE_MARL_MAX_AGENTS,
    )

    # Build observation once to get the obs size for mask helper.
    _probe = create_temporal_obs_builder_object()
    state_size = _probe.get_observation_size() if hasattr(_probe, 'get_observation_size') else 72
    action_size = environment.get_action_space()

    # Mask-helper policy: a parameter-sharing DeciderPPOPolicy that we
    # instantiate ONLY to call its `_legal_action_mask` method.
    mask_helper = DeciderPPOPolicy(
        state_size=state_size,
        action_size=action_size,
        learning_rate=1e-4,
        max_episodes_in_memory=1,
        train_frequency=10**9,  # never trigger an update
        temporal_window=TEMPORAL_WINDOW,
    )

    # Teacher: shares the env with the obs builder.
    teacher = DeadLockAvoidancePolicy(
        env=environment.get_raw_env(),
        action_size=action_size,
        enable_eps=False,
    )

    renderer = FlatlandSimpleRenderer(
        environment,
        render_each_episode=RENDER_EACH_EPISODE,
    ) if ENABLE_RENDERING else None

    all_transitions: List[Tuple[Any, int, np.ndarray, int, int]] = []
    stats: List[Dict[str, Any]] = []

    global_episode_id = 0
    t_total_start = time.perf_counter()

    for phase in DEMO_PHASES:
        print("\n" + "-" * 80)
        print(f"Phase: {phase['name']} | agents={phase['agent_counts']} | "
              f"envs={phase['num_envs']} | episodes={phase['episodes']}")
        print("-" * 80)
        _ensure_phase_envs(environment, phase)

        phase_done = 0
        phase_total = 0
        phase_transitions = 0
        checkpoint_interval = 100
        terminate_window: deque = deque(maxlen=checkpoint_interval)
        terminate_window.extend([0.0] * checkpoint_interval)

        for ep_in_phase in range(phase['episodes']):
            transitions, n_done, n_agents = _run_one_episode(
                environment, teacher, mask_helper, renderer, global_episode_id
            )
            all_transitions.extend(transitions)
            phase_done += n_done
            phase_total += n_agents
            phase_transitions += len(transitions)
            global_episode_id += 1

            ep_done_rate = n_done / max(1, n_agents)
            terminate_window.append(ep_done_rate)
            smoothed = np.mean(terminate_window)
            b = int(np.round(50 * smoothed))
            done_bar = '#' * b + '_' * (50 - b)
            print(
                '\rEpisode: {:5}  done: [{:^5.0f}/{:^5.0f}] : {:4.3f} : {:4.3f}  transitions: {:6d}  [{}]'.format(
                    global_episode_id,
                    n_done, n_agents,
                    ep_done_rate,
                    smoothed,
                    phase_transitions,
                    done_bar,
                ),
                end='\n' if (ep_in_phase + 1) % checkpoint_interval == 0 else '',
                flush=True,
            )

        stats.append({
            'phase': phase['name'],
            'episodes': phase['episodes'],
            'done': phase_done,
            'total_agents': phase_total,
            'transitions': phase_transitions,
            'done_rate': phase_done / max(1, phase_total),
        })

    t_total = time.perf_counter() - t_total_start
    print("\n" + "=" * 80)
    print(f"Collected {len(all_transitions):,} transitions in {t_total:.1f}s.")
    for s in stats:
        print(f"  {s['phase']:<14s}  done_rate={s['done_rate']:.2%}  "
              f"transitions={s['transitions']:>6d}")

    payload = {
        'transitions': all_transitions,
        'stats': stats,
        'state_size': state_size,
        'action_size': action_size,
        'temporal_window': TEMPORAL_WINDOW,
    }
    with open(DEMO_OUTPUT_FILE, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(DEMO_OUTPUT_FILE) / (1024 * 1024)
    print(f"\nWrote {DEMO_OUTPUT_FILE} ({size_mb:.1f} MB).")
    return payload


if __name__ == "__main__":
    collect()

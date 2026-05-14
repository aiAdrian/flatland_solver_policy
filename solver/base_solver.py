import os
import importlib.util
import importlib
from collections import deque
from typing import Union, Dict

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from environment.environment import Environment
from policy.policy import Policy
from rendering.base_renderer import BaseRenderer


class BaseSolver:
    def __init__(self,
                 env: Environment,
                 policy: Policy,
                 renderer: Union[BaseRenderer, None] = None):
        self.env = env
        self.policy: Policy = policy
        self.rendering_enabled = False
        self.renderer: Union[BaseRenderer, None] = renderer
        if renderer is not None:
            self.activate_rendering()

        self.max_steps = np.inf
        self._last_episode_agent_status = None

    def get_name(self) -> str:
        raise NotImplementedError

    def activate_rendering(self):
        self.rendering_enabled = True

    def deactivate_rendering(self):
        self.rendering_enabled = False

    def activate_renderer(self, renderer: BaseRenderer):
        self.renderer = renderer
        self.activate_rendering()

    def set_max_steps(self,
                      steps: int):
        self.max_steps = steps

    def render(self,
               episode: int,
               step: int,
               terminal: bool):
        if self.rendering_enabled:
            self.renderer.render(episode, step, terminal)

    def reset(self):
        state, info = self.env.reset()
        self.policy.reset(self.env)
        return state, info

    def run_step(self,
                 env: Environment,
                 policy: Policy,
                 state,
                 eps: float,
                 info: Dict,
                 training_mode: bool):

        tot_reward = 0
        all_terminal = True
        policy.start_step(train=training_mode)

        for handle in self.env.get_agent_handles():
            action = policy.act(handle, state, eps)

        state_next, reward, terminal, info = env.step(action)

        tot_terminal = 0
        for handle in self.env.get_agent_handles():
            all_terminal = all_terminal & terminal[handle]
            tot_terminal += terminal[handle]
            tot_reward += reward[handle]

            policy.step(handle, state, action, reward[handle], state_next, terminal[handle])
        tot_terminal /= max(1.0, len(self.env.get_agent_handles()))

        policy.end_step(train=training_mode)
        return state_next, tot_reward, all_terminal, tot_terminal, info

    def update_state(self,
                     state_next):
        return np.copy(state_next)

    def before_step_starts(self):
        return False

    def after_step_ends(self):
        return False

    def run_internal_episode(self,
                             episode: int,
                             env: Environment, policy: Policy,
                             state,
                             eps: float,
                             info: Dict,
                             training_mode: bool):
        tot_reward = 0
        tot_terminal = 0
        tot_steps = 0
        while True and tot_steps < self.max_steps:
            if self.before_step_starts():
                return tot_reward

            state_next, reward, terminal, tot_terminal, info = self.run_step(env,
                                                                             policy,
                                                                             state,
                                                                             eps,
                                                                             info,
                                                                             training_mode)
            tot_reward += reward
            state = self.update_state(state_next)
            self.render(episode, tot_steps, terminal)

            if self.after_step_ends():
                return tot_reward

            if terminal:
                break

            tot_steps += 1

        return tot_reward, tot_terminal, tot_steps

    def before_episode_starts(self):
        pass

    def after_episode_ends(self):
        pass

    def run_episode(self,
                    episode: int,
                    env: Environment,
                    policy: Policy,
                    eps: float,
                    training_mode: bool):
        state, info = self.reset()
        self.before_episode_starts()
        policy.start_episode(train=training_mode)
        tot_reward, tot_terminate, tot_steps = self.run_internal_episode(episode,
                                                                         env,
                                                                         policy,
                                                                         state,
                                                                         eps,
                                                                         info,
                                                                         training_mode)
        policy.end_episode(train=training_mode)
        self.after_episode_ends()
        self._last_episode_agent_status = self._compute_episode_agent_status()
        return tot_reward, tot_terminate, tot_steps

    def perform_evaluation(self,
                           max_episodes=2000,
                           eps=0.0,
                           write_summary=True,
                           checkpoint_interval=50):

        training_mode = False

        episode = 0
        scores_window = deque(maxlen=checkpoint_interval)
        terminate_window = deque(maxlen=checkpoint_interval)
        nbr_agents_window = deque(maxlen=checkpoint_interval)
        tot_steps_window = deque(maxlen=checkpoint_interval)
        deadlock_count_window = deque(maxlen=checkpoint_interval)

        writer = None
        if write_summary:
            writer = SummaryWriter(comment="_" + self.get_name() + "_evaluation_" + self.policy.get_name())

        while True:
            episode += 1

            tot_reward, tot_terminate, tot_steps = self.run_episode(episode, self.env, self.policy, eps, training_mode)

            scores_window.append(tot_reward)
            terminate_window.append(tot_terminate)
            nbr_agents_window.append(self.env.get_num_agents())
            tot_steps_window.append(tot_steps)

            deadlock_count = self._get_deadlock_count()
            deadlock_count_window.append(deadlock_count)

            # Optional policy hook: allow adaptive exploration control based on
            # recent done/deadlock trends without hard-coding policy internals here.
            done_mean = float(np.mean(terminate_window))
            deadlock_mean = float(np.mean(deadlock_count_window))
            if hasattr(self.policy, 'on_training_episode_end'):
                try:
                    hook_out = self.policy.on_training_episode_end(
                        episode=episode,
                        eps=eps,
                        min_eps=min_eps,
                        done_mean=done_mean,
                        deadlock_mean=deadlock_mean,
                        num_agents=self.env.get_num_agents(),
                    )
                    if isinstance(hook_out, dict) and 'eps' in hook_out:
                        eps = float(np.clip(hook_out['eps'], 0.0, 1.0))
                except Exception:
                    # Keep solver robust even if a custom policy hook fails.
                    pass

            b = int(np.round(50 * np.mean(terminate_window)))
            done_bar = ['#'] * b + ['_'] * (50 - b)

            print(
                '\rEpisode: {:5}\treward: {:9.3f} : {:9.3f}  \tdead locks  [{:^5.0f}/{:^5.0f}] : {:4.3f} : done [{:^5.0f}/{:^5.0f}] : {:4.3f}  \t [{}]'.format(
                    episode,
                    tot_reward,
                    np.mean(scores_window),
                    deadlock_count, self.env.get_num_agents(),
                    np.mean(deadlock_count_window),
                    tot_terminate * self.env.get_num_agents(), self.env.get_num_agents(),
                    np.mean(terminate_window),
                    ''.join(list(done_bar)),
                ),
                end='\n' if episode % checkpoint_interval == 0 else '')

            if writer is not None:
                writer.add_scalar(self.get_name() + "/evaluation_value_reward", tot_reward, episode)
                writer.add_scalar(self.get_name() + "/evaluation_smoothed_reward", np.mean(scores_window), episode)
                writer.add_scalar(self.get_name() + "/evaluation_value_done", tot_terminate, episode)
                writer.add_scalar(self.get_name() + "/evaluation_smoothed_done", np.mean(terminate_window), episode)
                writer.add_scalar(self.get_name() + "/evaluation_value_deadlock_count", deadlock_count, episode)
                writer.add_scalar(self.get_name() + "/evaluation_smoothed_deadlock_count", np.mean(deadlock_count_window), episode)
                writer.add_scalar(self.get_name() + "/evaluation_value_nbr_agents", self.env.get_num_agents(), episode)
                writer.add_scalar(self.get_name() + "/evaluation_smoothed_nbr_agents", np.mean(nbr_agents_window),
                                  episode)
                writer.add_scalar(self.get_name() + "/evaluation_value_nbr_steps", tot_steps, episode)
                writer.add_scalar(self.get_name() + "/evaluation_smoothed_nbr_steps", np.mean(tot_steps_window),
                                  episode)

                writer.flush()

            if episode >= max_episodes:
                break

            if episode % checkpoint_interval == 0:
                self._print_tree_search_diagnostics(episode)

        print('\ndone.')

    def perform_training(self,
                         max_episodes=2000,
                         eps=1.0,
                         eps_decay=0.995,
                         min_eps=0.001,
                         checkpoint_interval=100):

        training_mode = True

        episode = 0
        scores_window = deque(maxlen=checkpoint_interval)
        terminate_window = deque(maxlen=checkpoint_interval)
        nbr_agents_window = deque(maxlen=checkpoint_interval)
        tot_steps_window = deque(maxlen=checkpoint_interval)
        deadlock_count_window = deque(maxlen=checkpoint_interval)

        scores_window.extend([0] * checkpoint_interval)
        terminate_window.extend([0] * checkpoint_interval)
        nbr_agents_window.extend([0] * checkpoint_interval)
        tot_steps_window.extend([0] * checkpoint_interval)
        deadlock_count_window.extend([0] * checkpoint_interval)

        writer = SummaryWriter(comment="_" + self.get_name() + "_training_" + self.policy.get_name())
        
        # ==Pass writer to reward shaper if it supports it==
        if hasattr(self, '_reward_shaper') and self._reward_shaper is not None:
            if hasattr(self._reward_shaper, 'set_tensorboard_writer'):
                self._reward_shaper.set_tensorboard_writer(writer)
        
        # ==Pass writer to policy if it supports it==
        if hasattr(self.policy, 'set_tensorboard_writer'):
            self.policy.set_tensorboard_writer(writer)

        while True:
            episode += 1

            tot_reward, tot_terminate, tot_steps = self.run_episode(episode, self.env, self.policy, eps, training_mode)
            eps = max(min_eps, eps * eps_decay)

            scores_window.append(tot_reward)
            terminate_window.append(tot_terminate)
            nbr_agents_window.append(self.env.get_num_agents())
            tot_steps_window.append(tot_steps)

            deadlock_count = self._get_deadlock_count()
            deadlock_count_window.append(deadlock_count)

            # Adaptive exploration uses a short recent window so triggers are
            # not diluted by zero-initialized long buffers at training start.
            recent_w = min(20, len(terminate_window))
            done_recent = float(np.mean(list(terminate_window)[-recent_w:]))
            deadlock_recent = float(np.mean(list(deadlock_count_window)[-recent_w:]))
            if hasattr(self.policy, 'on_training_episode_end'):
                try:
                    hook_out = self.policy.on_training_episode_end(
                        episode=episode,
                        eps=eps,
                        min_eps=min_eps,
                        done_mean=done_recent,
                        deadlock_mean=deadlock_recent,
                        num_agents=self.env.get_num_agents(),
                    )
                    if isinstance(hook_out, dict) and 'eps' in hook_out:
                        eps = float(np.clip(hook_out['eps'], 0.0, 1.0))
                except Exception:
                    pass

            # Solver-level fallback rescue in case policy hook is unavailable or
            # too conservative. Keeps exploration alive during deadlock plateaus.
            n_agents = max(int(self.env.get_num_agents()), 1)
            deadlock_rate = deadlock_recent / float(n_agents)
            if done_recent < 0.10 and deadlock_rate > 0.60:
                eps = max(eps, min(0.16, max(min_eps, 0.12)))
            elif done_recent < 0.16 and deadlock_rate > 0.50:
                eps = max(eps, min(0.12, max(min_eps, 0.08)))

            b = int(np.round(50 * np.mean(terminate_window)))
            done_bar = ['#'] * b + ['_'] * (50 - b)

            print(
                '\rEpisode: {:5}\treward: {:9.3f} : {:9.3f} \tdead locks: [{:^5.0f}/{:^5.0f}] : {:4.3f} : done [{:^5.0f}/{:^5.0f}] : {:4.3f} \t [{}] \t eps: {:7.3f} '.format(
                    episode,
                    tot_reward,
                    np.mean(scores_window),
                    deadlock_count, self.env.get_num_agents(),
                    np.mean(deadlock_count_window),
                    tot_terminate * self.env.get_num_agents(), self.env.get_num_agents(),
                    np.mean(terminate_window),
                    ''.join(list(done_bar)),
                    eps
                ),
                end='\n' if episode % checkpoint_interval == 0 else '')

            writer.add_scalar(self.get_name() + "/training_value_reward", tot_reward, episode)
            writer.add_scalar(self.get_name() + "/training_smoothed_reward", np.mean(scores_window), episode)
            writer.add_scalar(self.get_name() + "/training_value_done", tot_terminate, episode)
            writer.add_scalar(self.get_name() + "/training_smoothed_done", np.mean(terminate_window), episode)
            writer.add_scalar(self.get_name() + "/training_value_deadlock_count", deadlock_count, episode)
            writer.add_scalar(self.get_name() + "/training_smoothed_deadlock_count", np.mean(deadlock_count_window), episode)
            writer.add_scalar(self.get_name() + "/training_value_nbr_agents", self.env.get_num_agents(), episode)
            writer.add_scalar(self.get_name() + "/training_smoothed_nbr_agents", np.mean(nbr_agents_window), episode)
            writer.add_scalar(self.get_name() + "/training_value_nbr_steps", tot_steps, episode)
            writer.add_scalar(self.get_name() + "/training_smoothed_nbr_steps", np.mean(tot_steps_window), episode)
            writer.flush()

            if episode % checkpoint_interval == 0 or episode >= max_episodes:
                checkpoint_path = "{}/{}_{}_{}".format(writer.get_logdir(),
                                                       self.get_name(), self.policy.get_name(),
                                                       episode)
                self.save_policy(filename=checkpoint_path)
                
                # Also save to training_output/last_checkpoint/ for easy access
                last_checkpoint_dir = "training_output/last_checkpoint"
                if not os.path.exists(last_checkpoint_dir):
                    os.makedirs(last_checkpoint_dir)
                last_checkpoint_path = f"{last_checkpoint_dir}/{self.get_name()}_{self.policy.get_name()}"
                self.save_policy(filename=last_checkpoint_path)
                
                if episode % checkpoint_interval == 0:
                    self._print_tree_search_diagnostics(episode)
                    print(f"\n💾 Checkpoint saved: Episode {episode}")
                    print(f"   Path: {checkpoint_path}")
                    print(f"   Last: {last_checkpoint_path}")
                    print("   Recover with: python marl_attention_temporal.py final_continue\n", end='')

            if episode >= max_episodes:
                break

        # --- end training --------------------------------------------------------------------------
        self.save_policy(None)
        print('\n✅ Training complete. Final checkpoint saved.')
        print(f'   Checkpoint directory: {writer.get_logdir()}')

    def save_policy(self,
                    filename: Union[str, None] = None):
        """
        If the filename is None use default policy output destination and default name.
        """
        if filename is None:
            if not os.path.exists('training_output'):
                os.makedirs('training_output')
            filename = "training_output/{}_{}".format(self.get_name(), self.policy.get_name())
        if self.policy is not None:
            self.policy.save(filename)

    def load_policy(self,
                    filename: Union[str, None] = None):
        """
        If the filename is None use default policy output source location and default name.
        """
        if filename is None:
            filename = "training_output/{}_{}".format(self.get_name(), self.policy.get_name())
        if self.policy is not None:
            self.policy.load(filename)

    def _get_deadlock_count(self) -> int:
        """Return per-episode deadlock count.

        Priority:
        1) Use reward shaper metric if available.
        2) Fallback to solver-level deadlock detection on the current env.
        """
        if hasattr(self, '_reward_shaper') and self._reward_shaper is not None:
            if hasattr(self._reward_shaper, 'get_last_episode_deadlock_count'):
                try:
                    return int(self._reward_shaper.get_last_episode_deadlock_count())
                except Exception:
                    pass

        local_deadlocks = self._estimate_deadlock_count_from_env()
        if local_deadlocks > 0:
            return int(local_deadlocks)

        # Fallback proxy: agents that are still on-map and not DONE at episode end.
        # This catches practical gridlock cases where strict corridor-deadlock
        # logic is too conservative and returns 0.
        status = self._last_episode_agent_status or {}
        return int(status.get('unfinished_on_map', 0))

    def _estimate_deadlock_count_from_env(self) -> int:
        """Best-effort deadlock count independent of reward shaper.

        This keeps training/eval logging meaningful when no reward shaper is used.
        """
        try:
            raw_env = self.env.get_raw_env() if hasattr(self.env, 'get_raw_env') else getattr(self.env, 'raw_env', self.env)
            if raw_env is None or not hasattr(raw_env, 'agents') or not hasattr(raw_env, 'rail'):
                return 0
            if not hasattr(raw_env, 'height') or not hasattr(raw_env, 'width'):
                return 0

            DecisionPointUtils = self._load_decision_point_utils()
            if DecisionPointUtils is None:
                return 0

            agent_map = np.zeros((raw_env.height, raw_env.width), dtype=np.int32) - 1
            for a in raw_env.agents:
                if getattr(a, 'position', None) is not None:
                    agent_map[a.position] = int(a.handle)

            deadlock_handles = set()
            for a in raw_env.agents:
                pos = getattr(a, 'position', None)
                direction = getattr(a, 'direction', None)
                if pos is None or direction is None:
                    continue
                state = getattr(a, 'state', None)
                state_name = getattr(state, 'name', '') if state is not None else ''
                if state_name == 'DONE':
                    continue
                try:
                    if DecisionPointUtils.is_local_deadlock(raw_env, a, agent_map):
                        deadlock_handles.add(int(a.handle))
                except Exception:
                    continue

            return int(len(deadlock_handles))
        except Exception:
            return 0

    @staticmethod
    def _load_decision_point_utils():
        """Load DecisionPointUtils without requiring global PYTHONPATH setup."""
        try:
            module = importlib.import_module("marl_attention_temporal_observation.decision_point_utils")
            return getattr(module, "DecisionPointUtils", None)
        except Exception:
            pass

        try:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            module_path = os.path.abspath(
                os.path.join(
                    this_dir,
                    "..",
                    "example",
                    "flatland_rail_env",
                    "marl_attention_temporal_observation",
                    "decision_point_utils.py",
                )
            )
            if not os.path.exists(module_path):
                return None

            spec = importlib.util.spec_from_file_location("_dp_utils_dynamic", module_path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, "DecisionPointUtils", None)
        except Exception:
            return None

    def _compute_episode_agent_status(self) -> Dict[str, int]:
        """Collect lightweight end-of-episode status counters."""
        status = {
            'total_agents': 0,
            'done_agents': 0,
            'unfinished_agents': 0,
            'unfinished_on_map': 0,
        }
        try:
            raw_env = self.env.get_raw_env() if hasattr(self.env, 'get_raw_env') else getattr(self.env, 'raw_env', self.env)
            agents = getattr(raw_env, 'agents', []) if raw_env is not None else []
            status['total_agents'] = int(len(agents))
            for a in agents:
                state = getattr(a, 'state', None)
                state_name = getattr(state, 'name', '') if state is not None else ''
                on_map = getattr(a, 'position', None) is not None
                if state_name == 'DONE':
                    status['done_agents'] += 1
                else:
                    status['unfinished_agents'] += 1
                    if on_map:
                        status['unfinished_on_map'] += 1
        except Exception:
            pass
        return status

    def _print_tree_search_diagnostics(self, episode: int):
        """Optional diagnostics for local-tree + seen-agent communication context."""
        try:
            raw_env = self.env.get_raw_env() if hasattr(self.env, 'get_raw_env') else getattr(self.env, 'raw_env', self.env)
            tree_dict = getattr(raw_env, 'dev_tree_dict', None) if raw_env is not None else None
            status = self._last_episode_agent_status or {}

            print("\n[TreeDiag] Episode {}".format(episode))
            print("  Agents: total={total} done={done} unfinished={undone} unfinished_on_map={uon}".format(
                total=status.get('total_agents', 0),
                done=status.get('done_agents', 0),
                undone=status.get('unfinished_agents', 0),
                uon=status.get('unfinished_on_map', 0),
            ))

            if not isinstance(tree_dict, dict) or len(tree_dict) == 0:
                print("  Tree payload: unavailable (env.dev_tree_dict empty)")
                return

            num_agents_payload = len(tree_dict)
            node_counts = []
            edge_counts = []
            seen_counts = []
            max_depths = []
            oncoming_nodes = 0
            total_nodes = 0

            for payload in tree_dict.values():
                nodes = payload.get('nodes', []) if isinstance(payload, dict) else []
                edges = payload.get('edges', []) if isinstance(payload, dict) else []
                seen = payload.get('seen_agents', []) if isinstance(payload, dict) else []

                node_counts.append(len(nodes))
                edge_counts.append(len(edges))
                seen_counts.append(len(seen))

                local_max_depth = 0
                for n in nodes:
                    d = int(n.get('depth', 0))
                    if d > local_max_depth:
                        local_max_depth = d
                    if bool(n.get('has_oncoming', False)):
                        oncoming_nodes += 1
                    total_nodes += 1
                max_depths.append(local_max_depth)

            mean_nodes = float(np.mean(node_counts)) if node_counts else 0.0
            mean_edges = float(np.mean(edge_counts)) if edge_counts else 0.0
            mean_seen = float(np.mean(seen_counts)) if seen_counts else 0.0
            max_depth = int(max(max_depths)) if max_depths else 0
            oncoming_ratio = (float(oncoming_nodes) / float(total_nodes)) if total_nodes > 0 else 0.0

            print("  Tree payload agents={} mean_nodes={:.2f} mean_edges={:.2f} mean_seen_agents={:.2f}".format(
                num_agents_payload, mean_nodes, mean_edges, mean_seen
            ))
            print("  Tree structure max_depth={} oncoming_node_ratio={:.3f}".format(max_depth, oncoming_ratio))
        except Exception as e:
            print(f"[TreeDiag] failed: {e}")

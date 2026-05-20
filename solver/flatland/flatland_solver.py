from typing import Union

from environment.environment import Environment
from policy.policy import Policy
from rendering.base_renderer import BaseRenderer
from solver.multi_agent_base_solver import MultiAgentBaseSolver


class FlatlandSolver(MultiAgentBaseSolver):
    def __init__(self,
                 env: Environment,
                 policy: Policy,
                 renderer: Union[BaseRenderer, None] = None):
        super(FlatlandSolver, self).__init__(env, policy, renderer)
        # Per-step pre-transition cell type cache: captured before env.step,
        # compared against post-step type in run_policy_step.
        self._pre_step_cell_types = {}

    def get_name(self) -> str:
        return self.__class__.__name__

    def run_choose_action(self, eps, handle, info, policy, state):
        # Capture cell type at time t (before env.step) for exact t -> t+1 filtering.
        if hasattr(policy, '_classify_cell_type'):
            try:
                raw_env = self.env.raw_env
                agent = raw_env.agents[handle]
                self._pre_step_cell_types[handle] = policy._classify_cell_type(agent, raw_env)
            except Exception:
                pass

        if info['action_required'][handle]:
            updated = True
            action = policy.act(handle,
                                state[handle],
                                eps)
        else:
            # An action is not required if the train hasn't joined the railway network,
            # if it already reached its target, or if the train is currently malfunctioning.
            updated = False
            action = 0
        return action, updated

    def run_policy_step(self, actions, policy, reward, state, state_next, terminal, terminal_all, update_values):
        """Override: Only train on meaningful t->t+1 state transitions.
        
        This implements state-machine reduction (Laurent et al. 2021): only call policy.step()
        when the agent's cell type changes from pre-step (t) to post-step (t+1), or on terminal.
        FORWARD_ONLY -> FORWARD_ONLY stretches are skipped to reduce training noise.
        """
        # Check if policy has cell-type classification (MARL_ATT_DecisionPointPolicy)
        has_cell_classification = hasattr(policy, '_classify_cell_type')
        
        for handle in self.env.get_agent_handles():
            should_train = False
            agent_done = bool(terminal[handle] or terminal_all)
            if self.env.raw_env._elapsed_steps > (self.env.raw_env._max_episode_steps - 5):
                # State-machine filtering: only train on cell-type transitions
                should_train = True
                agent_done = True
                update_values[handle] = True

            if update_values[handle] or terminal_all or agent_done:
                
                if has_cell_classification and not agent_done:
                    try:
                        raw_env = self.env.raw_env
                        agent = raw_env.agents[handle]

                        # Post-step type at time t+1.
                        next_cell_type = policy._classify_cell_type(agent, raw_env)
                        # Pre-step type at time t (captured in run_choose_action).
                        prev_cell_type = self._pre_step_cell_types.get(handle, next_cell_type)

                        # Skip only trivial FORWARD_ONLY->FORWARD_ONLY stretches.
                        # Always train at decision cells (SWITCH, MERGING, OUTSIDE) even if same type.
                        should_train = not (prev_cell_type == 'FORWARD_ONLY' and next_cell_type == 'FORWARD_ONLY')
                        
                    except Exception as e:
                        # Fallback: if classification fails, train normally
                        should_train = True
                        print(f"[Warning] Cell-type classification failed for agent {handle}: {e}")
                
                # Only call policy.step() if transition is meaningful
                if should_train or agent_done:
                    agent_finished = bool(terminal[handle])
                    try:
                        policy.step(handle,
                                    state[handle],
                                    actions[handle],
                                    reward[handle],
                                    state_next[handle],
                                    agent_done,
                                    agent_finished=agent_finished)
                    except TypeError:
                        policy.step(handle,
                                    state[handle],
                                    actions[handle],
                                    reward[handle],
                                    state_next[handle],
                                    agent_done)
                
                # Clear pre-step cache on terminal
                if agent_done:
                    self._pre_step_cell_types.pop(handle, None)

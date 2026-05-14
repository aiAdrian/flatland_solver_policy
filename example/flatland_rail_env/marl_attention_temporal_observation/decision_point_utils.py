from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero


class DecisionPointUtils:
    """Utility functions for deadlock detection and corridor blockage analysis.
    
    Centralized deadlock-detection logic used by:
    - DecisionPointObservation (for feature [65])
    - Reward shapers (FlatlandPBRSShaper, SimpleDoneRewardShaper)
    
    Deadlock Detection Strategy:
    - Detect confirmed corridor blockage on mandatory track segments
    - A corridor is "mandatory" when there's only one valid direction (no branching)
    - If an agent is blocked by another agent in opposite direction, it's a deadlock
    - Recursive: follow the chain of blocking agents to detect cycles
    """
    
    @staticmethod
    def is_opposite_direction(dir1, dir2) -> bool:
        """Check if two directions are opposite (head-on conflict).
        
        In Flatland: 0=North, 1=East, 2=South, 3=West
        Opposite pairs: 0↔2 (North↔South), 1↔3 (East↔West)
        
        Args:
            dir1: Direction index [0-3]
            dir2: Direction index [0-3]
            
        Returns:
            True if directions are exactly opposite, False otherwise
        """
        return (dir1 + 2) % 4 == dir2 

    @staticmethod
    def is_head_on_same_edge(raw_env, my_pos, my_dir, other_pos, other_dir) -> bool:
        """Check true head-on conflict along the same mandatory rail edge.

        This is curve-safe: on curved mandatory corridors, two agents can be in
        conflict even when direction ids are not numeric opposites.
        """
        my_transitions = raw_env.rail.get_transitions(*my_pos, my_dir)
        if fast_count_nonzero(my_transitions) != 1:
            return False
        my_next_dir = fast_argmax(my_transitions)
        my_next_pos = get_new_position(my_pos, my_next_dir)
        if my_next_pos != other_pos:
            return False

        other_transitions = raw_env.rail.get_transitions(*other_pos, other_dir)
        if fast_count_nonzero(other_transitions) != 1:
            return False
        other_next_dir = fast_argmax(other_transitions)
        other_next_pos = get_new_position(other_pos, other_next_dir)
        return other_next_pos == my_pos

    @staticmethod
    def is_local_deadlock(raw_env, agent, agent_map) -> bool:
        """Check if an agent is in a confirmed deadlock state.
        
        Deadlock = confirmed corridor blockage ahead on a mandatory (single-direction) corridor.
        
        Args:
            raw_env: Flatland RailEnv
            agent: Agent to check (must have position and direction)
            agent_map: [height, width] array mapping positions to agent handles
            
        Returns:
            True if agent is deadlocked, False otherwise
        """
        if agent.position is None or agent.direction is None:
            return False
        return DecisionPointUtils.detect_corridor_blockage(
            raw_env,
            agent_map,
            agent.handle,
            agent.position,
            agent.direction,
            {agent.handle},
            128,
            0,
        ) > 0

    @staticmethod
    def detect_corridor_blockage(raw_env, agent_map, handle, pos, direction, seen_agents, max_steps: int, step_offset: int) -> int:
        """Recursively detect if corridor is blocked by a cycle of agents.
        
        Walk forward along a mandatory corridor (single transitions only).
        If we encounter another agent:
        - If opposite direction → confirmed blockage (deadlock)
        - If same direction but trapped → recursively check their corridor
        - If already visited → cycle detected → deadlock
        
        Args:
            raw_env: Flatland RailEnv
            agent_map: Position→handle mapping
            handle: Current agent handle
            pos: Current position tuple (row, col)
            direction: Current direction [0-3]
            seen_agents: Set of handles already checked (prevents infinite recursion)
            max_steps: Maximum corridor length to check (default 128)
            step_offset: Starting step count
            
        Returns:
            Positive int (step distance) = deadlock confirmed at that distance
            0 = blocking agent has escape route
            -1 = safe (no blockage or reached a switch)
        """
        s = step_offset

        while s < max_steps:
            transitions = raw_env.rail.get_transitions(*pos, direction)
            num_trans = fast_count_nonzero(transitions)
            if num_trans == 0:
                return -1
            if num_trans > 1:
                # A switch can still resolve the conflict; do not mark as confirmed.
                return -1

            ndir = fast_argmax(transitions)
            if not transitions[ndir]:
                return -1

            npos = get_new_position(pos, ndir)
            s += 1

            agent_idx = agent_map[npos] if agent_map is not None else -1
            if agent_idx != -1 and agent_idx != handle:
                other = raw_env.agents[agent_idx]
                other_pos = other.position  
                other_dir = other.direction  
                if other_pos is not None:
                    # Cyclic blocking chain on a mandatory corridor.
                    if agent_idx in seen_agents:
                        return s
                     
                    if DecisionPointUtils.is_head_on_same_edge(raw_env, pos, direction, other_pos, other_dir):
                        pass
                    elif DecisionPointUtils.is_opposite_direction(ndir, other_dir):
                        other_transitions = raw_env.rail.get_transitions(*other_pos, other_dir)
                        if fast_count_nonzero(other_transitions) != 1:
                            # The blocking agent still has a local escape option.
                            return 0

                    seen_next = set(seen_agents)
                    seen_next.add(agent_idx)
                    return DecisionPointUtils.detect_corridor_blockage(
                        raw_env,
                        agent_map,
                        agent_idx,
                        other_pos,
                        other_dir,
                        seen_next,
                        max_steps,
                        s,
                    )

            pos = npos
            direction = ndir

        # CRITICAL FIX: Timeout on max_steps does NOT mean safe!
        # If we couldn't complete the analysis, flag as suspicious deadlock.
        # Was: return 0 (treat as safe) ← risky on long corridors
        # Now: return s (flag as potential deadlock at distance s)
        return s if s >= max_steps else -1

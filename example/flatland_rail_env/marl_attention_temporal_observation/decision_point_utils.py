from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero


class DecisionPointUtils:
    """Utility functions for simplified deadlock detection.
    
    Centralized deadlock-detection logic used by:
    - DecisionPointObservation (for feature [65])
    - Reward shapers (FlatlandPBRSShaper, SimpleDoneRewardShaper)
    
        Deadlock Detection Strategy (simplified):
        - Two agents facing each other on the same edge => deadlock.
        - If the next cell is occupied by an agent moving in the same direction,
            recursively check that lead agent.
        - If the lead agent is deadlocked, followers are also deadlocked.
    """

    @staticmethod
    def _pos_tuple(pos):
        if pos is None:
            return None
        return (int(pos[0]), int(pos[1]))

    @staticmethod
    def _rail_get_transitions(raw_env, pos, direction):
        """Get transitions from rail. Standard signature: get_transitions(row, col, dir)"""
        p = DecisionPointUtils._pos_tuple(pos)
        if p is None:
            return (0, 0, 0, 0)
        d = int(direction)
        return raw_env.rail.get_transitions(p[0], p[1], d)

    @staticmethod
    def _agent_at_pos(agent_map, pos) -> int:
        if agent_map is None:
            return -1
        p = DecisionPointUtils._pos_tuple(pos)
        if p is None:
            return -1
        try:
            return int(agent_map[p])
        except Exception:
            try:
                return int(agent_map[p[0], p[1]])
            except Exception:
                return -1
    
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
        my_transitions = DecisionPointUtils._rail_get_transitions(raw_env, my_pos, my_dir)
        if fast_count_nonzero(my_transitions) > 1:
            return False
        my_next_dir = fast_argmax(my_transitions)
        my_next_pos = get_new_position(my_pos, my_next_dir)
        if my_next_pos != other_pos:
            return False

        other_transitions = DecisionPointUtils._rail_get_transitions(raw_env, other_pos, other_dir)
        if fast_count_nonzero(other_transitions) > 1:
            return False
        other_next_dir = fast_argmax(other_transitions)
        other_next_pos = get_new_position(other_pos, other_next_dir)
        return other_next_pos == my_pos and my_next_pos == other_pos

    @staticmethod
    def is_local_deadlock(raw_env, agent, agent_map) -> bool:
        """Return True if simplified recursive deadlock rule is met."""
        if agent.position is None or agent.direction is None:
            return False
        return DecisionPointUtils._is_local_head_on_deadlock(raw_env, 
                                                            agent.handle, 
                                                            agent.position, 
                                                            agent.direction,
                                                            agent_map,
                                                            0,
                                                            32)
        
    @staticmethod
    def _is_local_head_on_deadlock(raw_env, handle, position, direction, agent_map, depth, max_depth) -> bool:
        if depth > max_depth:
            return False
        
        transitions = DecisionPointUtils._rail_get_transitions(raw_env, position, direction)
        if fast_count_nonzero(transitions) > 1:
            return False
        
        ndir = fast_argmax(transitions)
        npos = get_new_position(position, ndir)

        other_agent_id = DecisionPointUtils._agent_at_pos(agent_map, npos)
        if handle == other_agent_id:
            return False 
        
        if other_agent_id != -1:
            other_dir = raw_env.agents[other_agent_id].direction
            if other_dir != ndir:
                other_transitions = DecisionPointUtils._rail_get_transitions(raw_env, npos, other_dir)
                if fast_count_nonzero(other_transitions) == 1:
                    return True
        return DecisionPointUtils._is_local_head_on_deadlock(raw_env,
                                                            handle,                                                             
                                                            npos, 
                                                            ndir, 
                                                            agent_map, 
                                                            depth+1,
                                                            max_depth)
        

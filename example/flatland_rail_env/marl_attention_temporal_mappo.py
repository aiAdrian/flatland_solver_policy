import copy
import math
import os
import time
from collections import namedtuple, deque
from typing import Union, List, Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence

from policy.learning_policy.learning_policy import LearningPolicy
from marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation


def _load_state_dict_compatible(module: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """Load only matching-shape tensors to keep checkpoint compatibility across small architecture changes."""
    current = module.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current:
            skipped.append(key)
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        filtered[key] = value
    missing, unexpected = module.load_state_dict(filtered, strict=False)
    return missing, unexpected, skipped


# =============================================================================
# EPISODE BUFFERS 
# =============================================================================

class EpisodeBuffers:
    def __init__(self):
        self.reset()

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = {}

    def get_transitions(self, handle):
        return self.memory.get(handle, [])

    def push_transition(self, handle, transition):
        transitions = self.get_transitions(handle)

        # Keep exactly one terminal transition per agent.
        # If another done-transition arrives later (e.g., global all-done bonus),
        # merge its reward into the existing terminal transition instead of dropping it.
        if len(transitions) > 0:
            last_transition = transitions[-1]
            # Struktur: (state, action, reward, next_state, done, aux_deadlock)
            done_flag = last_transition[4]
            if done_flag: 
                merged_reward = \
                    float(last_transition[2]) \
                    + float(transition[2])
                merged_aux_deadlock = \
                    float(max(float(last_transition[5]), float(transition[5])))
                transitions[-1] = (
                    last_transition[0],
                    last_transition[1],
                    merged_reward,
                    transition[3],
                    True,
                    merged_aux_deadlock,
                )
                self.memory.update({handle: transitions})
                return

        transitions.append(transition)
        self.memory.update({handle: transitions})


# =============================================================================
# LOCAL TREE ENCODER (legacy)
# -----------------------------------------------------------------------------
# Legacy encoder kept only for old checkpoint compatibility.
# Active training path uses TreePayloadEncoder on raw tree payload.
#
# Node features (8D per node, legacy flattened representation):
#   [0] deadlock_risk        float 0-1
#   [1] norm_transitions     num_transitions / 3 → 0-1
#   [2] has_oncoming         binary 0/1  (Gegenverkehr am Knoten)
#   [3] norm_inflow          backward_inflow_count / 2 → 0-1
#   [4] norm_depth           depth / depth_limit → 0-1
#   [5] has_agents           1.0 if any agents_encountered else 0
#   [6] incoming_rel_dir     rel_dir_bin / 2: 0=links, 0.5=geradeaus, 1=rechts
#   [7] edge_has_agents      1.0 if agents on incoming edge else 0
#
# Topologie via DFS Pre-Order:
#   - Elternknoten immer vor Kindern in der Sequenz
#   - LSTM akkumuliert: "welche Abzweigung führt zu Deadlock?"
#   - incoming_rel_dir: "wir haben hier links/geradeaus/rechts abgebogen"
#   - norm_depth: auf welcher Tiefe sind wir?
#   - Variable Knotenanzahl: padding mit Nullen, pack_padded_sequence
#
# Architecture: input_proj → LSTM(DFS-Sequenz) → letzter hidden state → output_proj
# Ref: vereinfachte Variante von Tai et al. (2015) "Improved Semantic
#      Representations From Tree-Structured LSTM" (TreeLSTM)
# =============================================================================
class LocalTreeEncoder(nn.Module):
    """LSTM encoder over DFS-ordered tree nodes from DecisionPointObservation._local_search().

    Nodes are serialized in DFS pre-order so parents always precede their children.
    Each node carries 8D features including the incoming edge (rel_dir_bin, edge_has_agents)
    so the LSTM can reconstruct the branching topology from the sequence.

    Input:  (MAX_NODES * NODE_DIM,) = (120,) DFS-ordered flat sequence (legacy)
    Output: (hidden_dim,) topology-aware tree embedding
    """

    NODE_DIM = 8    # 6 node features + 2 incoming edge features
    MAX_NODES = 15  # 15 × 8 = 120D  (same block size as previous 20×6=120D)

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.node_dim = self.NODE_DIM
        self.max_nodes = self.MAX_NODES

        # Project 8D node+edge features to LSTM input dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.NODE_DIM, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.01),
        )

        # LSTM reads DFS sequence and accumulates topology context
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            dropout=0.2,  # Add dropout in LSTM
            num_layers=1,
            batch_first=True,
        )

        # Final projection from LSTM hidden state
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # Near-zero init → starts neutral, learns gradually during PPO
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tree_flat: torch.Tensor) -> torch.Tensor:
        """Single-agent: (MAX_NODES * NODE_DIM,) = (120,) → (hidden_dim,)"""
        logger.info("Forward pass with input shape: %s", tree_flat.shape)
        nodes = tree_flat.view(self.max_nodes, self.node_dim)  # (15, 8)
        mask = (nodes.abs().sum(dim=-1) > 1e-6)               # (15,) bool
        n_valid = int(mask.sum().clamp(min=1).item())
        # Feed only real nodes to LSTM (truncate padding)
        x = self.input_proj(nodes[:n_valid]).unsqueeze(0)      # (1, n_valid, H//2)
        _, (h_n, _) = self.lstm(x)                             # h_n: (1, 1, H)
        h = h_n.squeeze(0).squeeze(0)                          # (H,)
        return self.output_proj(h)                             # (H,)

    def forward_batch(self, tree_flat_batch: torch.Tensor) -> torch.Tensor:
        """Batch: (B, MAX_NODES * NODE_DIM) → (B, hidden_dim)"""
        B = tree_flat_batch.shape[0]
        nodes = tree_flat_batch.view(B, self.max_nodes, self.node_dim)  # (B, 15, 8)
        mask = (nodes.abs().sum(dim=-1) > 1e-6)                         # (B, 15) bool
        lengths = mask.sum(dim=-1).clamp(min=1).cpu()                    # (B,) int
        x = self.input_proj(nodes)                                       # (B, 15, H//2)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)                                  # h_n: (1, B, H)
        h = h_n.squeeze(0)                                               # (B, H)
        return self.output_proj(h)                                       # (B, H)


class TreePayloadEncoder(nn.Module):
    """Edge-aware encoder for variable-size tree payloads (nodes + edges).

    Handles dynamic graph size per sample using:
    - node padding + node masks
    - sparse edge message passing
    - masked mean pooling
    """

    NODE_DIM = 12
    EDGE_DIM = 20
    NODE_FEATURE_NAMES = [
        "deadlock_risk",
        "num_transitions_norm",
        "has_oncoming",
        "backward_inflow_norm",
        "depth_norm",
        "has_agents_encountered",
        "incoming_agent_count_norm",
        "is_branch",
        "deadlock_distance_norm",
        "deadlock_hard_distance_norm",
        "deadlock_exists_within_probe",
        "alternative_routes_count_norm",
    ]
    EDGE_FEATURE_NAMES = [
        "action_left",
        "action_forward",
        "action_right",
        "has_agents_on_edge",
        "has_oncoming_edge",
        "edge_len_cells_norm",
        "src_dist_to_target",
        "dst_dist_to_target",
        "delta_from_root",
        "improves_over_current",
        "target_on_edge",
        "dst_deadlock_risk",
        "dst_deadlock_hard_block",
        "dst_deadlock_distance_norm",
        "dst_deadlock_hard_distance_norm",
        "src_deadlock_distance_norm",
        "deadlock_distance_delta",
        "is_shortest_path_edge",
        "branch_choice_prob",
        "alternative_routes_count_norm",
    ]
    # Safe upper cap for dynamic per-batch padding.
    # Local search can emit >15 nodes, so payload path should not silently
    # collapse to the serialized-tree limit.
    MAX_NODES = 48

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.node_proj = nn.Sequential(
            nn.Linear(self.NODE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(self.EDGE_DIM, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        
        # ========================================================================
        # LEVEL ATTENTION (Hierarchisch: Root → Level 1 → Level 2 → Level 3)
        # ========================================================================
        self.level_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,  # Reduced from 4 to 2 for smaller trees
            batch_first=True
        )
        
        # Level Prior: Root wichtiger als Leaves
        # WICHTIG: Parameter (nicht Buffer) → wird mit MAPPO trainiert!
        self.level_importance = nn.Parameter(
            torch.tensor([1.0, 0.8, 0.6, 0.4], dtype=torch.float32)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _node_key(node: Dict[str, Any]) -> Tuple[int, int, int, int]:
        pos = node.get("pos", (0, 0))
        return int(pos[0]), int(pos[1]), int(node.get("dir", 0)), int(node.get("depth", 0))

    @staticmethod
    def _edge_key(pos, direction, depth) -> Tuple[int, int, int, int]:
        return int(pos[0]), int(pos[1]), int(direction), int(depth)

    @classmethod
    def _encode_node_feature(cls, node: Dict[str, Any]) -> np.ndarray:
        depth = int(node.get("depth", 0))
        node_type = int(node.get("type", -1))
        is_switch = bool(node.get("is_switch", node_type == 1))
        is_pre_merge = bool(node.get("is_pre_merge", node_type == 2))
        is_branch = bool(node.get("is_branch", False) or is_switch)
        alt_routes = int(node.get("alternative_routes_count", 0))
        if is_pre_merge and alt_routes <= 0:
            # Preserve pre-merge signal in the fixed 12D node feature budget.
            alt_routes = 1
        feat = np.array([
            float(node.get("deadlock_risk", 0.0)),
            min(1.0, float(node.get("num_transitions", 1)) / 3.0),
            1.0 if node.get("has_oncoming", False) else 0.0,
            min(1.0, float(node.get("backward_inflow_count", 0)) / 2.0),
            min(1.0, float(max(depth, 0)) / 12.0),
            1.0 if node.get("has_agents_encountered", False) else 0.0,
            min(1.0, float(node.get("incoming_agent_count", 0)) / 2.0),
            1.0 if is_branch else 0.0,
            float(node.get("deadlock_distance_norm", 0.0)),
            float(node.get("deadlock_hard_distance_norm", 0.0)),
            1.0 if node.get("deadlock_exists_within_probe", False) else 0.0,
            min(1.0, float(alt_routes) / 3.0),
        ], dtype=np.float32)
        np.clip(feat, 0.0, 1.0, out=feat)
        return feat

    @classmethod
    def _encode_edge_feature(cls, edge: Dict[str, Any]) -> np.ndarray:
        def _safe_float(value: Any, default: float = 0.0) -> float:
            if value is None:
                return default
            try:
                out = float(value)
            except (TypeError, ValueError):
                return default
            if not np.isfinite(out):
                return default
            return out

        def _safe_int(value: Any, default: int = 0) -> int:
            if value is None:
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _norm_dist(value: Any) -> float:
            # Compress raw distance-map values (often >>1) into [0,1].
            d = _safe_float(value, 1.0)
            if d <= 1.0:
                return float(np.clip(d, 0.0, 1.0))
            return float(d / (d + 32.0))

        rel_bin = _safe_int(edge.get("rel_dir_bin", 1), 1)
        action_left = _safe_float(edge.get("action_left", 1.0 if rel_bin == 0 else 0.0), 0.0)
        action_forward = _safe_float(edge.get("action_forward", 1.0 if rel_bin == 1 else 0.0), 0.0)
        action_right = _safe_float(edge.get("action_right", 1.0 if rel_bin == 2 else 0.0), 0.0)
        agents_on_edge_count = _safe_int(edge.get("agents_on_edge_count", 0), 0)
        if agents_on_edge_count <= 0:
            agents_on_edge = edge.get("agents_on_edge", [])
            if isinstance(agents_on_edge, list):
                agents_on_edge_count = len(agents_on_edge)
        feat = np.array([
            action_left,
            action_forward,
            action_right,
            1.0 if agents_on_edge_count > 0 else 0.0,
            1.0 if edge.get("has_oncoming_edge", False) else 0.0,
            min(1.0, _safe_float(edge.get("edge_len_cells", 1), 1.0) / 4.0),
            _norm_dist(edge.get("src_dist_to_target", 1.0)),
            _norm_dist(edge.get("dst_dist_to_target", 1.0)),
            _safe_float(edge.get("delta_from_root", 0.5), 0.5),
            _safe_float(edge.get("improves_over_current", 0.0), 0.0),
            1.0 if edge.get("target_on_edge", False) else 0.0,
            _safe_float(edge.get("dst_deadlock_risk", 0.0), 0.0),
            _safe_float(edge.get("dst_deadlock_hard_block", 0.0), 0.0),
            _safe_float(edge.get("dst_deadlock_distance_norm", 0.0), 0.0),
            _safe_float(edge.get("dst_deadlock_hard_distance_norm", 0.0), 0.0),
            _safe_float(edge.get("src_deadlock_distance_norm", 0.0), 0.0),
            _safe_float(edge.get("deadlock_distance_delta", 0.0), 0.0),
            _safe_float(edge.get("is_shortest_path_edge", 0.0), 0.0),
            _safe_float(edge.get("branch_choice_prob", 0.0), 0.0),
            min(1.0, _safe_float(edge.get("alternative_routes_count", 0), 0.0) / 3.0),
        ], dtype=np.float32)
        np.clip(feat, 0.0, 1.0, out=feat)
        return feat

    def _estimate_node_depths_batch(self, payload_batch: List[Dict[str, Any]], max_nodes: int) -> torch.Tensor:
        """
        Schätze Tiefe (depth) jedes Nodes pro Batch-Sample.
        Gruppiert in Level: 0 (0.0-0.25), 1 (0.25-0.50), 2 (0.50-0.75), 3 (0.75-1.0)
        
        Fallback-Strategie:
        1. Nutze depth_norm aus Node (falls vorhanden)
        2. Fallback: Nutze depth (raw depth count)
        3. Fallback: Nutze Node-Typ (INIT=0, SWITCH=1, PRE_M=2) → heuristische Tiefe
        4. Fallback: Nutze Node-Index (Node 0 = root, später = tiffer)
        
        Returns: (B, N) mit Level-Indizes {0, 1, 2, 3}
        """
        bsz = len(payload_batch)
        depth_bins = torch.zeros((bsz, max_nodes), dtype=torch.long)
        
        for b, payload in enumerate(payload_batch):
            if not isinstance(payload, dict):
                continue
            
            nodes = payload.get("nodes", []) or []
            max_depth = max([float(n.get("depth", 0)) for n in nodes], default=4.0)
            if max_depth <= 0:
                max_depth = 4.0
            
            for node_idx, node in enumerate(nodes):
                if node_idx >= max_nodes:
                    break
                
                # Versuch 1: depth_norm direkt
                if "depth_norm" in node:
                    depth_norm = float(node.get("depth_norm", 0.0))
                # Versuch 2: Berechne aus depth (raw depth value)
                elif "depth" in node:
                    depth_raw = float(node.get("depth", 0))
                    depth_norm = min(1.0, depth_raw / max_depth) if max_depth > 0 else 0.0
                # Versuch 3: Heuristische Tiefe aus Node-Typ
                else:
                    node_type = int(node.get("type", -1))
                    # INIT=0 → depth=0, SWITCH=1 → depth≈0.5, PRE_M=2 → depth≈0.75
                    type_hints = {0: 0.0, 1: 0.5, 2: 0.75}
                    depth_norm = type_hints.get(node_type, float(node_idx) / (len(nodes) + 1))
                
                depth_norm = max(0.0, min(1.0, depth_norm))  # Clamp to [0, 1]
                
                # Bin Tiefe in 4 Levels
                if depth_norm < 0.25:
                    level = 0
                elif depth_norm < 0.50:
                    level = 1
                elif depth_norm < 0.75:
                    level = 2
                else:
                    level = 3
                
                depth_bins[b, node_idx] = level
        
        return depth_bins

    def _payload_to_graph(self, payload: Dict[str, Any], max_nodes: int) -> Tuple[np.ndarray, List[Tuple[int, int, np.ndarray]], int]:
        max_nodes = int(max(1, max_nodes))
        node_feats = np.zeros((max_nodes, self.NODE_DIM), dtype=np.float32)
        edge_list: List[Tuple[int, int, np.ndarray]] = []

        if not isinstance(payload, dict):
            return node_feats, edge_list, 0

        nodes = payload.get("nodes", []) or []
        edges = payload.get("edges", []) or []

        idx_exact: Dict[Tuple[int, int, int, int], int] = {}
        idx_simple: Dict[Tuple[int, int, int], int] = {}
        idx_pos: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        n_valid = min(len(nodes), max_nodes)
        for i in range(n_valid):
            node = nodes[i]
            depth = int(node.get("depth", 0))
            node_feats[i] = self._encode_node_feature(node)

            key = self._node_key(node)
            idx_exact[key] = i
            idx_simple[(key[0], key[1], key[2])] = i
            pos_key = (key[0], key[1])
            idx_pos.setdefault(pos_key, []).append((depth, i))

        # Stable nearest-depth lookup for contracted edges or depth-mismatched payloads.
        for pos_key, depth_idx_pairs in idx_pos.items():
            depth_idx_pairs.sort(key=lambda x: x[0])

        def _nearest_by_depth(depth_idx_pairs: List[Tuple[int, int]], depth: int) -> int:
            best_idx = depth_idx_pairs[0][1]
            best_delta = abs(int(depth_idx_pairs[0][0]) - int(depth))
            for d_val, i_val in depth_idx_pairs[1:]:
                delta = abs(int(d_val) - int(depth))
                if delta < best_delta:
                    best_delta = delta
                    best_idx = i_val
            return int(best_idx)

        # Cache for node index resolutions to avoid repeated lookups (2-3× speedup)
        node_resolution_cache: Dict[Tuple[int, int, int, int], Optional[int]] = {}

        def _resolve_node_index(key4: Tuple[int, int, int, int]) -> Optional[int]:
            # Check cache first
            if key4 in node_resolution_cache:
                return node_resolution_cache[key4]
            
            # 1) Exact tuple match (pos,dir,depth)
            idx = idx_exact.get(key4)
            if idx is not None:
                node_resolution_cache[key4] = int(idx)
                return int(idx)

            # 2) Same pos+dir (depth mismatch tolerant)
            idx = idx_simple.get((key4[0], key4[1], key4[2]))
            if idx is not None:
                node_resolution_cache[key4] = int(idx)
                return int(idx)

            # 3) Same pos, nearest depth
            pos_key = (key4[0], key4[1])
            depth_idx_pairs = idx_pos.get(pos_key)
            if depth_idx_pairs:
                result = _nearest_by_depth(depth_idx_pairs, key4[3])
                node_resolution_cache[key4] = result
                return result

            # 4) Last-resort nearest position (L1), then nearest depth at that position
            best_pos = None
            best_dist = None
            for p_key in idx_pos.keys():
                dist = abs(int(p_key[0]) - int(key4[0])) + abs(int(p_key[1]) - int(key4[1]))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_pos = p_key
            if best_pos is not None:
                result = _nearest_by_depth(idx_pos[best_pos], key4[3])
                node_resolution_cache[key4] = result
                return result
            
            node_resolution_cache[key4] = None
            return None

        for edge in edges:
            if "src" in edge and "dst" in edge:
                s_idx = int(edge.get("src"))
                d_idx = int(edge.get("dst"))
            else:
                s_key = self._edge_key(edge.get("src_pos", (0, 0)), edge.get("src_dir", 0), edge.get("src_depth", 0))
                d_key = self._edge_key(edge.get("dst_pos", (0, 0)), edge.get("dst_dir", 0), edge.get("dst_depth", 0))
                s_idx = _resolve_node_index(s_key)
                d_idx = _resolve_node_index(d_key)

            if s_idx is None or d_idx is None:
                continue

            edge_feat = self._encode_edge_feature(edge)
            edge_list.append((int(s_idx), int(d_idx), edge_feat))

        np.clip(node_feats, 0.0, 1.0, out=node_feats)
        return node_feats, edge_list, n_valid

    def forward_batch(self, payload_batch: List[Dict[str, Any]]) -> torch.Tensor:
        if len(payload_batch) == 0:
            return torch.empty(0, self.hidden_dim, device=next(self.parameters()).device)

        device = next(self.parameters()).device
        bsz = len(payload_batch)

        # Dynamic padding size per batch (bounded by MAX_NODES for stability).
        max_nodes_in_batch = 1
        for payload in payload_batch:
            if isinstance(payload, dict):
                nodes = payload.get("nodes", []) or []
                if isinstance(nodes, list):
                    max_nodes_in_batch = max(max_nodes_in_batch, len(nodes))
        max_nodes = min(self.MAX_NODES, max_nodes_in_batch)

        node_arr = np.zeros((bsz, max_nodes, self.NODE_DIM), dtype=np.float32)
        node_mask = torch.zeros((bsz, max_nodes), dtype=torch.float32, device=device)
        edge_graphs: List[List[Tuple[int, int, np.ndarray]]] = []

        for b, payload in enumerate(payload_batch):
            n_feat, e_list, n_valid = self._payload_to_graph(payload, max_nodes)
            node_arr[b] = n_feat
            edge_graphs.append(e_list)
            if n_valid > 0:
                node_mask[b, :n_valid] = 1.0

        nodes = torch.as_tensor(node_arr, dtype=torch.float32, device=device)
        node_h = self.node_proj(nodes)  # (B, N, H)
        messages = torch.zeros_like(node_h)

        for b, e_list in enumerate(edge_graphs):
            if len(e_list) == 0:
                continue
            for src, dst, e_feat_np in e_list:
                if src >= max_nodes or dst >= max_nodes:
                    continue
                e_feat = torch.as_tensor(e_feat_np, dtype=torch.float32, device=device)
                gate = torch.sigmoid(self.edge_gate(e_feat)).squeeze(-1)
                messages[b, dst] += gate * node_h[b, src]

        msg_h = self.msg_proj(messages)
        node_u = self.update_proj(torch.cat([node_h, msg_h], dim=-1))

        # ========================================================================
        # HIERARCHICAL POOLING (Topologie-erhaltend für variable Größen)
        # ========================================================================
        # Statt einfaches Mean-Pooling: Gruppiere nach Tiefe & aggregiere pro Level
        # So bleibt Root vs. Leaves Struktur erhalten!
        
        # Zuerst: Schätze depth pro Node aus Payload
        depth_bins = self._estimate_node_depths_batch(payload_batch, max_nodes)
        depth_bins = depth_bins.to(device)  # (B, N) mit Werten {0, 1, 2, 3}
        
        # Für jeden Level: Masked Average der Node-Updates
        level_embeddings = []
        for level_idx in range(4):  # Levels 0-3 (depth groups)
            level_mask = (depth_bins == level_idx).float()  # (B, N)
            level_mask_expanded = level_mask.unsqueeze(-1)  # (B, N, 1)
            
            # Masked pooling für diesen Level
            level_sum = (node_u * level_mask_expanded).sum(dim=1)  # (B, H)
            level_count = level_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
            level_pool = level_sum / level_count  # (B, H)
            level_embeddings.append(level_pool)
        
        # Stack alle Level-Embeddings
        levels_stacked = torch.stack(level_embeddings, dim=1)  # (B, 4, H)
        
        # ========================================================================
        # LEVEL ATTENTION (Lernt welche Tiefen wichtig sind)
        # ========================================================================
        # Selbst-Attention über die 4 Tiefe-Level
        level_context, _ = self.level_attention(
            query=levels_stacked,
            key=levels_stacked,
            value=levels_stacked
        )  # (B, 4, H)
        
        # Gewichte aggregieren (Root hat höchstes Gewicht, Leaves flexible)
        level_weights = torch.softmax(
            self.level_importance.unsqueeze(0).expand(bsz, -1),
            dim=-1
        )  # (B, 4) - Prior: Root am wichtigsten
        
        # Finale Aggregation
        pooled = (level_context * level_weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
        
        return self.output_proj(pooled)


# Constants shared across encoder classes
_BASE_OBS_DIM = int(DecisionPointObservation.BASE_OBS_SIZE)


# =============================================================================
# NEW: TEMPORAL TRANSFORMER ENCODER - 2-Level Attention!
# -----------------------------------------------------------------------------
# Architektur basiert auf:
#   Vaswani et al. (2017) "Attention Is All You Need", arXiv:1706.03762
#       -> Multi-Head Self-Attention, Positional Encoding
#   Ba, Kiros, Hinton (2016) "Layer Normalization", arXiv:1607.06450
#       -> stabilisiert Training tiefer Encoder ohne Batch-Statistik
#   He et al. (2015) "Delving Deep into Rectifiers", arXiv:1502.01852
#       -> Kaiming-Init für (Leaky)ReLU-Netze
#
# Multi-Agent-Anwendung mit Attention zwischen Agenten:
#   Iqbal & Sha (2019) "Actor-Attention-Critic for Multi-Agent RL",
#       arXiv:1810.02912 (MAAC) -- Spatial-Attention zwischen Agenten
#   Yu et al. (2022) "The Surprising Effectiveness of PPO in Cooperative
#       Multi-Agent Games" (MAPPO), arXiv:2103.01955
# =============================================================================

class TemporalTransformerEncoder(nn.Module):
    """
    🚀 INNOVATION: Hierarchical Temporal-Spatial Transformer
    
    Architecture:
    1. Observation Encoder: Maps base observation vector → hidden embedding
    2. TEMPORAL Attention: Links t-2, t-1, t → learns movement patterns
    3. SPATIAL Attention: Links self + opponents → learns interactions
    4. Output Projection: Final 128D context embedding
    
    Advantages over simple MAEncoderWithAttention:
    ✅ Sees movement (not just snapshot)
    ✅ Predicts collisions (velocity awareness)
    ✅ Smoother trajectories (temporal context)
    """
    
    def __init__(self, 
                 obs_dim: int,
                 hidden_dim: int,        # 128D
                 num_heads: int = 4,
                 temporal_window: int = 3,
                 device="cpu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window
        self.device = torch.device(device)
        
        # ========================================================================
        # LEVEL 1: Observation Encoder (spatial features → embeddings)
        # obs_encoder processes only the handcrafted base vector.
        # Tree information is provided separately as raw payload.
        # ========================================================================
        self.base_obs_dim = min(obs_dim, _BASE_OBS_DIM)  # = 35
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.base_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01)
        )
        
        # ========================================================================
        # LEVEL 2: Temporal Self-Attention
        # Processes sequence [obs_t-T+1, ..., obs_t-1, obs_t] → temporal_context
        # ========================================================================
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Positional Encoding for timesteps (differentiates t-2 vs t-1 vs t)
        self.temporal_pe = nn.Parameter(
            torch.randn(temporal_window, hidden_dim) * 0.01
        )
        
        # ========================================================================
        # LEVEL 3: Spatial Multi-Agent Attention
        # Processes [self_context, opp1_context, opp2_context, ...]
        # ========================================================================
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # ========================================================================
        # LEVEL 4: Output Projection
        # ========================================================================
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01)
        )

        # Tree signal comes exclusively from raw local-search payload.
        self.tree_payload_encoder = TreePayloadEncoder(hidden_dim)
        self.tree_norm = nn.LayerNorm(hidden_dim)
        self.use_tree_payload_encoder = True

        # Explicit communication: sender message + receiver addressing.
        self.comm_msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.comm_sender_gate = nn.Linear(hidden_dim, 1)
        self.comm_sender_key = nn.Linear(hidden_dim, hidden_dim)
        self.comm_receiver_query = nn.Linear(hidden_dim, hidden_dim)
        self.comm_intent_head = nn.Linear(hidden_dim, 3)  # [WAIT, GO, YIELD]
        self.comm_intent_embedding = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.comm_norm = nn.LayerNorm(hidden_dim)
        self.comm_dropout = nn.Dropout(p=0.10)
        self.last_comm_reg = torch.tensor(0.0, device=self.device)
        self.last_comm_gate_mean = 0.0
        self.last_comm_intent_mean = [0.0, 0.0, 0.0]
        self.last_comm_valid_count = 0
        
        self._init_weights()
        self.to(self.device)
    
    def _init_weights(self):
        """Kaiming initialization for LeakyReLU"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.comm_sender_gate.bias is not None:
            # Open communication channel at init; regularization will prune later.
            nn.init.constant_(self.comm_sender_gate.bias, 1.0)
    
    def _to_1d_tensor(self, x):
        """Convert to 1D tensor on device with NaN/Inf protection"""
        if x is None:
            raise ValueError("_to_1d_tensor received None")
        if isinstance(x, torch.Tensor):
            t = x.to(self.device)
        else:
            t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        
        t = t.view(-1).to(self.device)
        
        # ⚠️ PROTECTION: Replace NaN/Inf with 0
        if torch.isnan(t).any() or torch.isinf(t).any():
            print("⚠️ WARNING: Input contains NaN/Inf, replacing with 0")
            t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # ⚠️ PROTECTION: Clamp extreme values
        t = torch.clamp(t, min=-10.0, max=10.0)

        return t

    def _unpack_temporal_step(self, step) -> Tuple[Any, List[Any], Dict[str, Any]]:
        """Support both legacy (obs, opps) and new (obs, opps, tree_payload) steps."""
        if isinstance(step, (list, tuple)):
            if len(step) >= 3:
                obs_self, opponents, tree_payload = step[0], step[1], step[2]
                if not isinstance(opponents, list):
                    opponents = []
                if not isinstance(tree_payload, dict):
                    tree_payload = {}
                return obs_self, opponents, tree_payload
            if len(step) == 2:
                obs_self, opponents = step
                if not isinstance(opponents, list):
                    opponents = []
                return obs_self, opponents, {}
            if len(step) == 1:
                return step[0], [], {}
        return step, [], {}

    def _encode_tree_payload(self, tree_payload: Dict[str, Any]) -> torch.Tensor:
        return self.tree_payload_encoder.forward_batch([tree_payload])[0]

    def _encode_tree_signal(self, obs_1d: torch.Tensor, tree_payload: Dict[str, Any]) -> torch.Tensor:
        del obs_1d
        if not bool(getattr(self, 'use_tree_payload_encoder', True)):
            return torch.zeros(self.hidden_dim, device=self.device)
        return self._encode_tree_payload(tree_payload if isinstance(tree_payload, dict) else {})

    def _encode_tree_signal_batch(self, last_obs_b: torch.Tensor, tree_payloads: List[Dict[str, Any]]) -> torch.Tensor:
        del last_obs_b
        if not bool(getattr(self, 'use_tree_payload_encoder', True)):
            bsz = len(tree_payloads)
            return torch.zeros((bsz, self.hidden_dim), device=self.device)
        return self.tree_payload_encoder.forward_batch(tree_payloads)

    def _apply_communication(self, self_context: torch.Tensor, opp_embeddings: List[torch.Tensor]):
        """Fuse explicit communication from opponents into receiver context."""
        if len(opp_embeddings) == 0:
            zero = torch.tensor(0.0, device=self.device)
            zero_intent = torch.zeros(3, device=self.device)
            return self_context, zero, zero, zero_intent

        all_agents = [self_context] + opp_embeddings
        stack = torch.stack(all_agents, dim=0)  # (N, H)

        # Active communication token per sender (differentiable intent distribution).
        intent_logits = self.comm_intent_head(stack)              # (N, 3)
        intent_probs = torch.softmax(intent_logits, dim=-1)       # (N, 3)
        intent_vec = torch.matmul(intent_probs, self.comm_intent_embedding)  # (N, H)

        messages = torch.tanh(self.comm_msg_proj(stack) + intent_vec)  # (N, H)
        messages = self.comm_dropout(messages)
        sender_gate = torch.sigmoid(self.comm_sender_gate(stack)).squeeze(-1)  # (N,)

        recv_q = self.comm_receiver_query(self_context)  # (H,)
        send_k = self.comm_sender_key(stack)  # (N, H)

        logits = torch.matmul(send_k, recv_q) / math.sqrt(float(self.hidden_dim))
        addr = torch.softmax(logits, dim=0)  # which sender to listen to
        weights = addr * sender_gate
        weights = weights / (weights.sum() + 1e-6)

        comm_vec = torch.sum(messages * weights.unsqueeze(-1), dim=0)
        gate_mean = sender_gate.mean()
        # Small L1-like penalty to discourage communication spam.
        comm_reg = gate_mean
        intent_mean = intent_probs.mean(dim=0)
        return self.comm_norm(self_context + comm_vec), comm_reg, gate_mean, intent_mean
    
    def forward_agent(self, temporal_seq: List, handle: int = 0):
        """
        Process temporal sequence for one agent
        
        Args:
            temporal_seq: [(obs_t-2, opp_t-2), (obs_t-1, opp_t-1), (obs_t, opp_t)]
                         Each obs is a base observation vector
            handle: Agent handle (for debugging)
        
        Returns:
            context_embedding: (hidden_dim,) - Temporal + Spatial Context
        """
        # ========================================================================
        # STEP 1: Encode all observations across time
        # ========================================================================
        
        # Extract self observations over time
        self_obs_sequence = []  # Will be [emb_t-2, emb_t-1, emb_t]
        for step in temporal_seq:
            obs_self, _, _ = self._unpack_temporal_step(step)
            obs_t = self._to_1d_tensor(obs_self)
            base_obs = obs_t[:self.base_obs_dim]
            emb_t = self.obs_encoder(base_obs)    # (H,)
            self_obs_sequence.append(emb_t)
        
        # Stack to (T, hidden_dim)
        self_seq_tensor = torch.stack(self_obs_sequence, dim=0)  # (3, 128)
        
        # ========================================================================
        # STEP 2: TEMPORAL ATTENTION - Learn movement patterns
        # ========================================================================
        
        # Add positional encoding (differentiates timesteps)
        self_seq_with_pe = self_seq_tensor + self.temporal_pe  # (temporal_window, 128)
        
        # Self-Attention over time
        self_seq_batched = self_seq_with_pe.unsqueeze(0)  # (1, temporal_window, 128)
        
        temporal_output, _ = self.temporal_attention(
            query=self_seq_batched,
            key=self_seq_batched,
            value=self_seq_batched
        )
        # temporal_output: (1, temporal_window, 128) - Each timestep now has temporal context!
        
        # Take only t (current timestep) as representation
        self_temporal_context = temporal_output[0, -1, :]  # (128,)

        # Encode raw local-tree payload at current timestep.
        last_obs, _, last_payload = self._unpack_temporal_step(temporal_seq[-1])
        last_obs_t = self._to_1d_tensor(last_obs)
        tree_emb = self._encode_tree_signal(last_obs_t, last_payload)
        self_temporal_context = self.tree_norm(self_temporal_context + tree_emb)
        
        # ========================================================================
        # STEP 3: SPATIAL ATTENTION - Multi-Agent Interaction (optional)
        # ========================================================================
        
        # Check if spatial attention is enabled (can be disabled for speed)
        use_spatial = getattr(self, 'use_spatial_attention', True)
        
        # Get current opponent observations (only from t, not entire history)
        _, current_opponents, _ = self._unpack_temporal_step(temporal_seq[-1])  # Last timestep
        
        # Encode opponents (current timestep base features) if spatial attention is enabled.
        opp_embeddings = []
        if use_spatial and len(current_opponents) > 0:
            for opp_obs in current_opponents:
                opp_t = self._to_1d_tensor(opp_obs)
                opp_base = opp_t[:self.base_obs_dim]
                opp_emb = self.obs_encoder(opp_base)
                opp_embeddings.append(opp_emb)
        
        # Combine self + opponents (only if spatial attention is enabled AND opponents exist)
        if use_spatial and len(opp_embeddings) > 0:
            all_agents = [self_temporal_context] + opp_embeddings
            all_agents_tensor = torch.stack(all_agents, dim=0).unsqueeze(0)
            query = self_temporal_context.unsqueeze(0).unsqueeze(0)
            spatial_output, _ = self.spatial_attention(
                query=query,
                key=all_agents_tensor,
                value=all_agents_tensor
            )
            context = spatial_output.squeeze(0).squeeze(0) + self_temporal_context
            context, comm_reg, gate_mean, intent_mean = self._apply_communication(context, opp_embeddings)
            self.last_comm_reg = comm_reg
            self.last_comm_gate_mean = float(gate_mean.detach().cpu().item())
            self.last_comm_intent_mean = [float(x) for x in intent_mean.detach().cpu().tolist()]
            self.last_comm_valid_count = 1
        else:
            # Spatial attention disabled OR no opponents → only own temporal context
            context = self_temporal_context
            self.last_comm_reg = torch.tensor(0.0, device=self.device)
            self.last_comm_gate_mean = 0.0
            self.last_comm_intent_mean = [0.0, 0.0, 0.0]
            self.last_comm_valid_count = 0
        
        # ========================================================================
        # STEP 4: Output Projection
        # ========================================================================
        
        final_embedding = self.output_proj(context)
        
        return final_embedding
    
    def forward_batch(self, temporal_sequences: List):
        """
        Batch processing of multiple agents - OPTIMIZED
        
        Args:
            temporal_sequences: List of temporal_seq per agent
        
        Returns:
            (batch_size, hidden_dim)
        """
        if len(temporal_sequences) == 0:
            return torch.empty(0, self.hidden_dim, device=self.device)
        
        # Fast path for single agent (no stacking overhead)
        if len(temporal_sequences) == 1:
            return self.forward_agent(temporal_sequences[0], 0).unsqueeze(0)
        
        # ⚡ OPTIMIZATION: Process all agents in parallel batches
        batch_size = len(temporal_sequences)
        
        # Extract all self observations and stack for parallel processing
        all_self_obs = []  # (batch_size, temporal_window, 33)
        all_opponents = []  # List of opponent lists per agent
        
        all_tree_payloads = []

        for temp_seq in temporal_sequences:
            self_seq = [self._unpack_temporal_step(step)[0] for step in temp_seq]
            all_self_obs.append(torch.stack([self._to_1d_tensor(obs) for obs in self_seq]))
            _, current_opps, payload = self._unpack_temporal_step(temp_seq[-1])
            all_opponents.append(current_opps)
            all_tree_payloads.append(payload if isinstance(payload, dict) else {})
        
        # Stack: (batch_size, temporal_window, 33)
        all_self_obs_tensor = torch.stack(all_self_obs, dim=0)
        
        # Encode base obs features; tree signal comes from payload path.
        flat_obs = all_self_obs_tensor.view(-1, all_self_obs_tensor.shape[-1])
        flat_base_obs = flat_obs[:, :self.base_obs_dim]  # (B*T, 35)
        flat_embeddings = self.obs_encoder(flat_base_obs)  # (B*T, H)
        
        # Reshape back: (batch_size, temporal_window, hidden_dim)
        self_embeddings = flat_embeddings.view(batch_size, self.temporal_window, self.hidden_dim)
        
        # Add positional encoding: (batch_size, temporal_window, hidden_dim)
        # Use only the first temporal_window positional encodings
        self_with_pe = self_embeddings + self.temporal_pe[:self.temporal_window].unsqueeze(0)
        
        # Temporal attention (batch_first=True supports batching!)
        temporal_output, _ = self.temporal_attention(
            query=self_with_pe,
            key=self_with_pe,
            value=self_with_pe
        )
        
        # Extract current timestep context: (batch_size, hidden_dim)
        self_temporal_contexts = temporal_output[:, -1, :]

        # Tree signal (batch): fuse serialized block and raw tree_payload.
        last_obs_b = all_self_obs_tensor[:, -1, :]  # (B, obs_dim)
        tree_emb_b = self._encode_tree_signal_batch(last_obs_b, all_tree_payloads)
        self_temporal_contexts = self.tree_norm(self_temporal_contexts + tree_emb_b)
        
        # Spatial attention (process per agent due to varying opponent counts)
        use_spatial = bool(getattr(self, 'use_spatial_attention', True))
        final_embeddings = []
        comm_regs = []
        comm_gate_means = []
        comm_intents = []
        for i in range(batch_size):
            self_ctx = self_temporal_contexts[i]
            opps = all_opponents[i]
            
            if use_spatial and len(opps) > 0:
                opp_embs = []
                for opp_obs in opps:
                    opp_t = self._to_1d_tensor(opp_obs)
                    opp_base = opp_t[:self.base_obs_dim]
                    opp_embs.append(self.obs_encoder(opp_base))

                all_agents = [self_ctx] + opp_embs
                all_agents_tensor = torch.stack(all_agents, dim=0).unsqueeze(0)
                query = self_ctx.unsqueeze(0).unsqueeze(0)
                spatial_out, _ = self.spatial_attention(
                    query=query,
                    key=all_agents_tensor,
                    value=all_agents_tensor
                )
                context = spatial_out.squeeze(0).squeeze(0) + self_ctx
                context, comm_reg, gate_mean, intent_mean = self._apply_communication(context, opp_embs)
                comm_regs.append(comm_reg)
                comm_gate_means.append(gate_mean)
                comm_intents.append(intent_mean)
            else:
                context = self_ctx
                comm_regs.append(torch.tensor(0.0, device=self.device))
                comm_gate_means.append(torch.tensor(0.0, device=self.device))
                comm_intents.append(torch.zeros(3, device=self.device))

            final_embeddings.append(self.output_proj(context))

        self.last_comm_reg = torch.stack(comm_regs).mean()
        self.last_comm_gate_mean = float(torch.stack(comm_gate_means).mean().detach().cpu().item())
        valid_intents = [it for it, opps in zip(comm_intents, all_opponents) if len(opps) > 0]
        if len(valid_intents) > 0:
            intent_avg = torch.stack(valid_intents).mean(dim=0).detach().cpu().tolist()
            self.last_comm_intent_mean = [float(x) for x in intent_avg]
            self.last_comm_valid_count = len(valid_intents)
        else:
            self.last_comm_intent_mean = [0.0, 0.0, 0.0]
            self.last_comm_valid_count = 0
        return torch.stack(final_embeddings, dim=0)
    
    def save(self, filename: str):
        torch.save(self.state_dict(), filename + ".temporal_encoder")
    
    def load(self, filename: str):
        state_file = filename + ".temporal_encoder"
        if os.path.exists(state_file):
            sd = torch.load(state_file, map_location=self.device)
            missing, unexpected, skipped = _load_state_dict_compatible(self, sd)
            if skipped:
                print(f"[TemporalTransformerEncoder] Skipped incompatible weights: {skipped}")
            if missing:
                print(f"[TemporalTransformerEncoder] New weights (fresh init): {missing}")


class TemporalLSTMEncoder(nn.Module):
    """Lightweight temporal encoder with LSTM over self-history.

    It keeps the same interface as TemporalTransformerEncoder so it can be
    swapped without changing the PPO training loop.
    """

    def __init__(self,
                 obs_dim: int,
                 hidden_dim: int,
                 num_heads: int = 4,
                 temporal_window: int = 3,
                 device="cpu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window
        self.device = torch.device(device)

        # obs_encoder processes only the handcrafted base vector.
        # Tree information is provided separately as raw payload.
        self.base_obs_dim = min(obs_dim, _BASE_OBS_DIM)  # = 35
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.base_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01)
        )

        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01)
        )

        self.comm_msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.comm_sender_gate = nn.Linear(hidden_dim, 1)
        self.comm_sender_key = nn.Linear(hidden_dim, hidden_dim)
        self.comm_receiver_query = nn.Linear(hidden_dim, hidden_dim)
        self.comm_intent_head = nn.Linear(hidden_dim, 3)  # [WAIT, GO, YIELD]
        self.comm_intent_embedding = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.comm_norm = nn.LayerNorm(hidden_dim)
        self.comm_dropout = nn.Dropout(p=0.10)

        # Tree signal comes exclusively from raw local-search payload.
        self._tree_slots = None  # Backward-compatibility stub for old checkpoints.
        self.tree_payload_encoder = TreePayloadEncoder(hidden_dim)
        self.tree_norm = nn.LayerNorm(hidden_dim)
        self.use_tree_payload_encoder = True

        self.last_comm_reg = torch.tensor(0.0, device=self.device)
        self.last_comm_gate_mean = 0.0
        self.last_comm_intent_mean = [0.0, 0.0, 0.0]
        self.last_comm_valid_count = 0

        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.comm_sender_gate.bias is not None:
            # Open communication channel at init; regularization will prune later.
            nn.init.constant_(self.comm_sender_gate.bias, 1.0)

    def _to_1d_tensor(self, x):
        if x is None:
            raise ValueError("_to_1d_tensor received None")
        if isinstance(x, torch.Tensor):
            t = x.to(self.device)
        else:
            t = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        t = t.view(-1).to(self.device)
        if torch.isnan(t).any() or torch.isinf(t).any():
            t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
        t = torch.clamp(t, min=-10.0, max=10.0)
        return t

    def _unpack_temporal_step(self, step) -> Tuple[Any, List[Any], Dict[str, Any]]:
        """Support both legacy (obs, opps) and new (obs, opps, tree_payload) steps."""
        if isinstance(step, (list, tuple)):
            if len(step) >= 3:
                obs_self, opponents, tree_payload = step[0], step[1], step[2]
                if not isinstance(opponents, list):
                    opponents = []
                if not isinstance(tree_payload, dict):
                    tree_payload = {}
                return obs_self, opponents, tree_payload
            if len(step) == 2:
                obs_self, opponents = step
                if not isinstance(opponents, list):
                    opponents = []
                return obs_self, opponents, {}
            if len(step) == 1:
                return step[0], [], {}
        return step, [], {}

    def _encode_tree_payload(self, tree_payload: Dict[str, Any]) -> torch.Tensor:
        return self.tree_payload_encoder.forward_batch([tree_payload])[0]

    def _encode_tree_signal(self, obs_1d: torch.Tensor, tree_payload: Dict[str, Any]) -> torch.Tensor:
        del obs_1d
        if not bool(getattr(self, 'use_tree_payload_encoder', True)):
            return torch.zeros(self.hidden_dim, device=self.device)
        return self._encode_tree_payload(tree_payload if isinstance(tree_payload, dict) else {})

    def _encode_tree_signal_batch(self, last_obs_b: torch.Tensor, tree_payloads: List[Dict[str, Any]]) -> torch.Tensor:
        del last_obs_b
        if not bool(getattr(self, 'use_tree_payload_encoder', True)):
            bsz = len(tree_payloads)
            return torch.zeros((bsz, self.hidden_dim), device=self.device)
        return self.tree_payload_encoder.forward_batch(tree_payloads)

    def _apply_communication(self, self_context: torch.Tensor, opp_embeddings: List[torch.Tensor]):
        if len(opp_embeddings) == 0:
            zero = torch.tensor(0.0, device=self.device)
            zero_intent = torch.zeros(3, device=self.device)
            return self_context, zero, zero, zero_intent

        all_agents = [self_context] + opp_embeddings
        stack = torch.stack(all_agents, dim=0)

        intent_logits = self.comm_intent_head(stack)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        intent_vec = torch.matmul(intent_probs, self.comm_intent_embedding)

        messages = torch.tanh(self.comm_msg_proj(stack) + intent_vec)
        messages = self.comm_dropout(messages)
        sender_gate = torch.sigmoid(self.comm_sender_gate(stack)).squeeze(-1)

        recv_q = self.comm_receiver_query(self_context)
        send_k = self.comm_sender_key(stack)

        logits = torch.matmul(send_k, recv_q) / math.sqrt(float(self.hidden_dim))
        addr = torch.softmax(logits, dim=0)
        weights = addr * sender_gate
        weights = weights / (weights.sum() + 1e-6)

        comm_vec = torch.sum(messages * weights.unsqueeze(-1), dim=0)
        gate_mean = sender_gate.mean()
        comm_reg = gate_mean
        intent_mean = intent_probs.mean(dim=0)
        return self.comm_norm(self_context + comm_vec), comm_reg, gate_mean, intent_mean

    def forward_agent(self, temporal_seq: List, handle: int = 0):
        self_obs_sequence = []
        for step in temporal_seq:
            obs_self, _, _ = self._unpack_temporal_step(step)
            obs_t = self._to_1d_tensor(obs_self)
            base_obs = obs_t[:self.base_obs_dim]
            emb_t = self.obs_encoder(base_obs)
            self_obs_sequence.append(emb_t)

        self_seq_tensor = torch.stack(self_obs_sequence, dim=0).unsqueeze(0)
        temporal_output, _ = self.temporal_lstm(self_seq_tensor)
        self_temporal_context = temporal_output[0, -1, :]

        # Encode raw local-tree payload at current timestep.
        last_obs, _, last_payload = self._unpack_temporal_step(temporal_seq[-1])
        last_obs_t = self._to_1d_tensor(last_obs)
        tree_emb = self._encode_tree_signal(last_obs_t, last_payload)
        self_temporal_context = self.tree_norm(self_temporal_context + tree_emb)

        # Check if spatial attention is enabled (can be disabled for speed)
        use_spatial = getattr(self, 'use_spatial_attention', True)
        
        _, current_opponents, _ = self._unpack_temporal_step(temporal_seq[-1])
        opp_embeddings = []
        if use_spatial and len(current_opponents) > 0:
            for opp_obs in current_opponents:
                opp_t = self._to_1d_tensor(opp_obs)
                opp_base = opp_t[:self.base_obs_dim]
                opp_embeddings.append(self.obs_encoder(opp_base))

        if use_spatial and len(opp_embeddings) > 0:
            all_agents = [self_temporal_context] + opp_embeddings
            all_agents_tensor = torch.stack(all_agents, dim=0).unsqueeze(0)
            query = self_temporal_context.unsqueeze(0).unsqueeze(0)
            spatial_out, _ = self.spatial_attention(
                query=query,
                key=all_agents_tensor,
                value=all_agents_tensor
            )
            context = spatial_out.squeeze(0).squeeze(0) + self_temporal_context
            context, comm_reg, gate_mean, intent_mean = self._apply_communication(context, opp_embeddings)
            self.last_comm_reg = comm_reg
            self.last_comm_gate_mean = float(gate_mean.detach().cpu().item())
            self.last_comm_intent_mean = [float(x) for x in intent_mean.detach().cpu().tolist()]
            self.last_comm_valid_count = 1
        else:
            context = self_temporal_context
            self.last_comm_reg = torch.tensor(0.0, device=self.device)
            self.last_comm_gate_mean = 0.0
            self.last_comm_intent_mean = [0.0, 0.0, 0.0]
            self.last_comm_valid_count = 0

        return self.output_proj(context)

    def forward_batch(self, temporal_sequences: List):
        if len(temporal_sequences) == 0:
            return torch.empty(0, self.hidden_dim, device=self.device)
        if len(temporal_sequences) == 1:
            return self.forward_agent(temporal_sequences[0], 0).unsqueeze(0)

        batch_size = len(temporal_sequences)
        all_self_obs = []
        all_opponents = []

        all_tree_payloads = []

        for temp_seq in temporal_sequences:
            self_seq = [self._unpack_temporal_step(step)[0] for step in temp_seq]
            all_self_obs.append(torch.stack([self._to_1d_tensor(obs) for obs in self_seq]))
            _, current_opps, payload = self._unpack_temporal_step(temp_seq[-1])
            all_opponents.append(current_opps)
            all_tree_payloads.append(payload if isinstance(payload, dict) else {})

        all_self_obs_tensor = torch.stack(all_self_obs, dim=0)
        flat_obs = all_self_obs_tensor.view(-1, all_self_obs_tensor.shape[-1])
        flat_base_obs = flat_obs[:, :self.base_obs_dim]  # (B*T, 35)
        flat_embeddings = self.obs_encoder(flat_base_obs)  # (B*T, H)
        self_embeddings = flat_embeddings.view(batch_size, self.temporal_window, self.hidden_dim)

        temporal_output, _ = self.temporal_lstm(self_embeddings)
        self_temporal_contexts = temporal_output[:, -1, :]

        # Tree signal (batch): fuse serialized block and raw tree_payload.
        last_obs_b = all_self_obs_tensor[:, -1, :]  # (B, obs_dim)
        tree_emb_b = self._encode_tree_signal_batch(last_obs_b, all_tree_payloads)
        self_temporal_contexts = self.tree_norm(self_temporal_contexts + tree_emb_b)

        use_spatial = bool(getattr(self, 'use_spatial_attention', True))
        final_embeddings = []
        comm_regs = []
        comm_gate_means = []
        comm_intents = []
        for i in range(batch_size):
            self_ctx = self_temporal_contexts[i]
            opps = all_opponents[i]

            if use_spatial and len(opps) > 0:
                opp_embs = []
                for opp_obs in opps:
                    opp_t = self._to_1d_tensor(opp_obs)
                    opp_base = opp_t[:self.base_obs_dim]
                    opp_embs.append(self.obs_encoder(opp_base))

                all_agents = [self_ctx] + opp_embs
                all_agents_tensor = torch.stack(all_agents, dim=0).unsqueeze(0)
                query = self_ctx.unsqueeze(0).unsqueeze(0)
                spatial_out, _ = self.spatial_attention(
                    query=query,
                    key=all_agents_tensor,
                    value=all_agents_tensor
                )
                context = spatial_out.squeeze(0).squeeze(0) + self_ctx
                context, comm_reg, gate_mean, intent_mean = self._apply_communication(context, opp_embs)
                comm_regs.append(comm_reg)
                comm_gate_means.append(gate_mean)
                comm_intents.append(intent_mean)
            else:
                context = self_ctx
                comm_regs.append(torch.tensor(0.0, device=self.device))
                comm_gate_means.append(torch.tensor(0.0, device=self.device))
                comm_intents.append(torch.zeros(3, device=self.device))

            final_embeddings.append(self.output_proj(context))

        self.last_comm_reg = torch.stack(comm_regs).mean()
        self.last_comm_gate_mean = float(torch.stack(comm_gate_means).mean().detach().cpu().item())
        valid_intents = [it for it, opps in zip(comm_intents, all_opponents) if len(opps) > 0]
        if len(valid_intents) > 0:
            intent_avg = torch.stack(valid_intents).mean(dim=0).detach().cpu().tolist()
            self.last_comm_intent_mean = [float(x) for x in intent_avg]
            self.last_comm_valid_count = len(valid_intents)
        else:
            self.last_comm_intent_mean = [0.0, 0.0, 0.0]
            self.last_comm_valid_count = 0
        return torch.stack(final_embeddings, dim=0)

    def save(self, filename: str):
        torch.save(self.state_dict(), filename + ".temporal_encoder")

    def load(self, filename: str):
        state_file = filename + ".temporal_encoder"
        if os.path.exists(state_file):
            sd = torch.load(state_file, map_location=self.device)
            missing, unexpected, skipped = _load_state_dict_compatible(self, sd)
            if skipped:
                print(f"[TemporalLSTMEncoder] Skipped incompatible weights: {skipped}")
            if missing:
                print(f"[TemporalLSTMEncoder] New weights (fresh init): {missing}")


# =============================================================================
# ACTOR-CRITIC MODEL  
# =============================================================================

class ActorCriticModel(nn.Module):
    def __init__(self, state_size, action_size, device,
                 hidsize1=512, hidsize2=256,
                 critic_hidsize1=None, critic_hidsize2=None):
        super(ActorCriticModel, self).__init__()
        self.device = device
        critic_hidsize1 = hidsize1 if critic_hidsize1 is None else int(critic_hidsize1)
        critic_hidsize2 = hidsize2 if critic_hidsize2 is None else int(critic_hidsize2)
        
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, action_size)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(state_size, critic_hidsize1),
            nn.Tanh(),
            nn.Linear(critic_hidsize1, critic_hidsize2),
            nn.Tanh(),
            nn.Linear(critic_hidsize2, 1)
        ).to(self.device)

        # Auxiliary head: predicts one-step deadlock risk from actor embedding.
        # This is used as an auxiliary task only (no action override).
        self.deadlock_head = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, 1)
        ).to(self.device)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_actor_dist(self, state):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        return dist

    def evaluate(self, states, actions):
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_value = self.critic(states)
        return action_logprobs, torch.squeeze(state_value, dim=-1), dist_entropy

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + ".actor")
        torch.save(self.critic.state_dict(), filename + ".value")
        torch.save(self.deadlock_head.state_dict(), filename + ".deadlock")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            obj.load_state_dict(torch.load(filename, map_location=self.device))
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.actor = self._load(self.actor, filename + ".actor")
        self.critic = self._load(self.critic, filename + ".value")
        self.deadlock_head = self._load(self.deadlock_head, filename + ".deadlock")


# =============================================================================
# PPO PARAMETERS (extended with temporal_window + optimization flags)
# =============================================================================

"""MARL_ATTENTION_TEMPORAL_MAPPO_Param: Configuration namedtuple for Multi-Agent PPO policy

STANDARD PARAMETERS:
  hidden_size (int)
    - Encoder hidden dimension (default 64)
    - Lower values = faster training but reduced capacity
    - Range: 32, 64, 128, 256
    
  batch_size (int)
    - Mini-batch size for PPO updates (default 128)
    - Larger values = more stable but more memory
    
  learning_rate (float)
    - Adam optimizer learning rate (default 2.5e-5)
    - Critical for convergence stability
    
  discount (float)
    - Discount factor γ (default 0.99)
    - Controls how much future rewards matter
    
  temporal_window (int)
    - History window size in timesteps (default 3)
    - How many past observations to include in sequence
    
  encoder_type (str)
    - 'lstm' or 'transformer' (default 'transformer')
    - Which encoder architecture to use

OPTIMIZATION FLAGS (NEW - v2.0):
  encoder_shared (bool)
    - Share single encoder between actor+critic
    - False (default): Separate encoders
      • Full model capacity: each head has dedicated feature extractor
      • ~2× parameters and ~2× forward time
      • Better feature specialization
      • Use when: CPU not bottleneck, or convergence issues with sharing
      
    - True: Shared encoder
      • ~50% faster training (one encoder forward instead of two)
      • -50% parameters (single encoder vs two)
      • Forced feature alignment between actor and critic
      • Potential conflicting gradients (actor vs critic objectives)
      • Use when: CPU bottleneck or memory-limited
      
  use_spatial_attention (bool)
    - Enable multi-head attention between agent and opponents (MAAC)
    - True (default): Full MAAC (Multi-Agent Actor-Critic)
      • Agent attends to self + all opponents
      • Learns coordination, collision-avoidance patterns
      • ~20% slower but essential for interaction-rich scenarios
      • From: Iqbal & Sha (2019) "Actor-Attention-Critic\"
      
    - False: Skip spatial attention
      • ~20% faster training
      • Use temporal context only (no opponent information)
      • Each agent learns independent policy
      • Better for: simple scenarios with minimal interactions

ENVIRONMENT VARIABLES (for launching training):
  FLATLAND_HIDDEN_SIZE=<int>
    Sets hidden_size. Default: 64
    Example: export FLATLAND_HIDDEN_SIZE=32
    
  FLATLAND_ENCODER_TYPE=<'lstm'|'transformer'>
    Sets encoder architecture. Default: 'transformer'
    Example: export FLATLAND_ENCODER_TYPE=lstm
    
  FLATLAND_ENCODER_SHARED=<'true'|'false'>
    Enable shared encoder mode. Default: 'false'
    Example: export FLATLAND_ENCODER_SHARED=true
    
  FLATLAND_USE_SPATIAL_ATTENTION=<'true'|'false'>
    Enable spatial attention. Default: 'true'
    Example: export FLATLAND_USE_SPATIAL_ATTENTION=false

LAUNCH EXAMPLES:
  # Default (full MAAC, separate encoders)
  python marl_attention_temporal.py final_continue
  
  # Lightweight CPU-friendly (70% faster)
  FLATLAND_ENCODER_SHARED=true FLATLAND_USE_SPATIAL_ATTENTION=false \\
  python marl_attention_temporal.py final_continue
  
  # Balanced (shared encoder, keep spatial attention)
  FLATLAND_ENCODER_SHARED=true python marl_attention_temporal.py final_continue
"""

MARL_ATTENTION_TEMPORAL_MAPPO_Param = namedtuple('MARL_ATTENTION_TEMPORAL_MAPPO_Param',
                            ['hidden_size', 'batch_size', 'learning_rate',
                             'discount', 'gae_lambda', 'use_gpu',
                             'max_episodes_in_training_memory', 'batch_fraction', 'k_epochs',
                             'max_batches_per_training', 'temporal_window', 'encoder_type',
                             'encoder_shared', 'use_spatial_attention'])


# =============================================================================
# PPO POLICY - With Temporal Transformer!
# =============================================================================

class MARL_ATTENTION_TEMPORAL_PPOPolicy(LearningPolicy):
    """
    🚀 Temporal Multi-Agent PPO Policy
    
    - TemporalTransformerEncoder (2-Level Attention: Temporal + Spatial)
    - Handles temporal sequences instead of single observations
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 in_parameters: Union[MARL_ATTENTION_TEMPORAL_MAPPO_Param, None] = None,
                 show_pre_train_debug_msg = False,
                 show_progress_bar = False,
                 train_frequency = 1,
                 optimizer_mode: str = 'single'):
        super(MARL_ATTENTION_TEMPORAL_PPOPolicy, self).__init__()

        self.show_debug_msg = False
        self.show_pre_train_debug_msg = show_pre_train_debug_msg
        self.show_progress_bar = show_progress_bar
        self.train_frequency = train_frequency
        self.episode_count = 0
        # Default: 'single' (empfohlen, stabiler, Standard in Flatland)
        self.optimizer_mode = (optimizer_mode or 'single').lower()  # 'single' or 'multiple'

        self.state_size = state_size  # temporal obs size per timestep
        self.action_size = action_size
        self.num_heads = 4

        # Parameters
        self.ppo_parameters = in_parameters
        if self.ppo_parameters is not None:
            self.hidden_size = self.ppo_parameters.hidden_size
            self.batch_size = self.ppo_parameters.batch_size
            self.learning_rate = self.ppo_parameters.learning_rate
            self.discount = self.ppo_parameters.discount
            self.temporal_window = getattr(self.ppo_parameters, 'temporal_window', 3)
            self.encoder_type = getattr(self.ppo_parameters, 'encoder_type', 'transformer')
            self.encoder_shared = getattr(self.ppo_parameters, 'encoder_shared', False)  # Share Actor+Critic encoder
            self.use_spatial_attention = getattr(self.ppo_parameters, 'use_spatial_attention', True)   # Enable spatial attention
        else:
            self.hidden_size = 256
            self.learning_rate = 5.0e-3
            self.discount = 0.99
            self.batch_size = 128  # Back to baseline
            self.temporal_window = 3
            self.encoder_type = 'transformer'
            self.encoder_shared = False
            self.use_spatial_attention = True

        # Device
        if self.ppo_parameters is not None and getattr(self.ppo_parameters, 'use_gpu', False) and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        # PPO Hyperparameters
        if self.ppo_parameters is not None:
            self.K_epoch = getattr(self.ppo_parameters, 'k_epochs', 3)
        else:
            self.K_epoch = 3  # Back to baseline
            
        self.surrogate_eps_clip = 0.10  # tighter trust region to reduce KL spikes
        self.weight_loss = 0.75  # lower critic pressure so shared encoder does not swamp actor updates
        self.weight_entropy = 0.13  # stronger exploration for deadlock-heavy sparse decision regimes
        # Exploration boost: increase decision_eps_floor and max_eps_random
        self.decision_eps_floor = 0.14  # keep exploration active in sparse-decision regimes
        self.max_eps_random = 0.20      # raise global random exploration ceiling
        self.weight_policy = 1.80
        
        # ========================================================================
        # SIMPLIFIED MODE: Core PPO only, with optional agent count override
        # Set FLATLAND_SIMPLIFIED_MAPPO=N where:
        #   0 = Full Mode (Aux, Diversity, Comm enabled)
        #   1,5,10,100 = Core PPO only with N agents
        # ========================================================================
        simplified_val = str(os.getenv('FLATLAND_SIMPLIFIED_MAPPO', '0')).strip()
        try:
            self.simplified_mode = int(simplified_val)
        except ValueError:
            self.simplified_mode = 1 if simplified_val.lower() in ('1', 'true', 'yes', 'on') else 0
        
        if self.simplified_mode > 0:
            print(f"\n🧠 CORE PPO MODE: {self.simplified_mode} agents, no Aux/Diversity/Comm")
        else:
            print("\n🚀 FULL MODE: Aux deadlock, Diversity, Communication enabled")

        # In CORE mode default to a scalable architecture baseline:
        # no spatial/communication coupling unless explicitly requested.
        if self.simplified_mode > 0:
            disable_spatial_in_core = str(os.getenv('FLATLAND_SIMPLE_DISABLE_SPATIAL', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
            if disable_spatial_in_core:
                self.use_spatial_attention = False
                print("   - CORE override: use_spatial_attention=False")
        
        # Optional auxiliary losses (AuxDL disabled entirely: does not converge, noises gradient)
        self.weight_aux_deadlock = 0.0
        self.weight_action_diversity = 0.0 if self.simplified_mode > 0 else 0.07
        self.weight_comm = 0.0 if self.simplified_mode > 0 else 3.0e-4
        
        # Action diversity parameters (used only if weight_action_diversity > 0)
        self.forward_prob_soft_max = 0.66
        self.lr_prob_soft_min = 0.11
        self.idle_prob_soft_max = 0.22
        # Apply action-diversity shaping only at meaningful conflict/decision contexts.
        self.action_diversity_gate_enabled = True
        self.action_diversity_gate_threshold = 0.65
        self.aux_deadlock_pos_weight = 4.0
        self.comm_reg_start_episode = 300
        self.comm_reg_full_episode = 600
        self.comm_dropout_early = 0.00
        self.comm_dropout_late = 0.05
        self.stability_guard_start_episode = 600
        self.stability_guard_hard_episode = 900
        self.ppo_target_kl = 0.020
        self.ppo_max_kl = 0.050
        self.ratio_guard_soft = 1.15
        self.ratio_guard_hard = 1.22
        self.ratio_guard_soft_low = 0.85
        self.ratio_guard_hard_low = 0.75
        self.ppo_emergency_kl = 0.12
        self.ppo_emergency_kl_hard = 0.25
        self.comm_gate_target = 0.032
        self.max_hard_batches_before_lr_decay = 4
        self.hard_spike_streak_limit = 2
        self.actor_lr_decay_on_instability = 0.75
        self.actor_lr_recover_rate = 1.01
        self.actor_lr_min_factor = 0.35
        self.actor_lr_max_factor = 1.00
        # Pre-clip norms are often O(1e2..1e3) in early Flatland MAPPO updates.
        # Keep soft/hard guards aligned with this scale to avoid over-decaying LR.
        self.grad_norm_soft = 600.0
        self.grad_norm_hard = 900.0
        self.max_grad_norm_single = 0.40
        self.max_grad_norm_actor = 0.45
        self.max_grad_norm_critic = 0.45
        self.grad_norm_skip_step_hard = 400.0
        self.adv_clip_abs = 5.0
        self.gae_lambda = self.ppo_parameters.gae_lambda if self.ppo_parameters else 0.95

        # Reward scaling: raw per-step rewards are O(-0.5), giving discounted returns
        # in [-30, +3]. With SmoothL1(beta=1) in the linear regime, V_Loss stays ≈ 10
        # and advantages remain noisy. Scaling rewards to [-3, +0.3] drives V_Loss to
        # <0.1, enabling the critic to converge. Policy gradient is unaffected because
        # advantages are normalized per mini-batch regardless of absolute scale.
        self.reward_scale = 0.08

        # Memory
        self.current_episode_memory = EpisodeBuffers()
        
        if self.ppo_parameters is not None:
            self.max_episodes_in_training_memory = getattr(self.ppo_parameters, 'max_episodes_in_training_memory', 10)  # Back to baseline
            self.batch_fraction = getattr(self.ppo_parameters, 'batch_fraction', 1.0)
            self.max_batches_per_training = getattr(self.ppo_parameters, 'max_batches_per_training', None)
        else:
            self.max_episodes_in_training_memory = 10  # Back to baseline
            self.batch_fraction = 1.0
            self.max_batches_per_training = None
        
        self.accumulated_episodes: deque = deque(maxlen=self.max_episodes_in_training_memory)
        self.replay_sample_percent = float(np.clip(float(os.getenv('FLATLAND_REPLAY_SAMPLE_PERCENT', '0.10')), 0.02, 1.0))
        
        self.loss = 0

        # ========================================================================
        # NEW: TEMPORAL TRANSFORMER ENCODERS (separate for Actor and Critic)
        # ========================================================================
        print("\n🚀 Creating Temporal Encoders:")
        print(f"   - obs_dim: {state_size}")
        print(f"   - hidden_dim: {self.hidden_size}")
        print(f"   - temporal_window: {self.temporal_window}")
        print(f"   - num_heads: {self.num_heads}")
        print(f"   - encoder_type: {self.encoder_type}")
        print(f"   - optimizer_mode: {self.optimizer_mode}")
        print(
            f"   - ppo: clip={self.surrogate_eps_clip:.3f}, target_kl={self.ppo_target_kl:.3f}, "
            f"max_kl={self.ppo_max_kl:.3f}, guard_start={self.stability_guard_start_episode}, "
            f"guard_hard={self.stability_guard_hard_episode}"
        )
        print(
            f"   - actor_lr: decay={self.actor_lr_decay_on_instability:.3f}, "
            f"recover={self.actor_lr_recover_rate:.3f}, min_factor={self.actor_lr_min_factor:.3f}"
        )

        encoder_cls = TemporalLSTMEncoder if str(self.encoder_type).lower() == 'lstm' else TemporalTransformerEncoder

        if self.encoder_shared:
            # ⚡ SHARED ENCODER MODE: Actor + Critic use same encoder (~50% faster)
            self.encoder_actor = encoder_cls(
                obs_dim=state_size,
                hidden_dim=self.hidden_size,
                num_heads=self.num_heads,
                temporal_window=self.temporal_window,
                device=self.device
            )
            self.encoder_critic = self.encoder_actor  # Pointer to same object
            print("✅ Shared Encoder: actor + critic use 1 encoder (50% faster)")
        else:
            # Default: separate encoders for actor + critic
            self.encoder_actor = encoder_cls(
                obs_dim=state_size,
                hidden_dim=self.hidden_size,
                num_heads=self.num_heads,
                temporal_window=self.temporal_window,
                device=self.device
            )
            self.encoder_critic = encoder_cls(
                obs_dim=state_size,
                hidden_dim=self.hidden_size,
                num_heads=self.num_heads,
                temporal_window=self.temporal_window,
                device=self.device
            )
        
        # Set spatial attention flag on encoders (used in forward_agent())
        self.encoder_actor.use_spatial_attention = bool(self.use_spatial_attention)
        self.encoder_critic.use_spatial_attention = bool(self.use_spatial_attention)

        # Optional performance switch: disable tree-payload encoding in core/simple runs.
        # Default is ON in CORE mode to keep runtime stable from 1 -> 5 -> 10 -> 100 agents
        # without extra manual environment flags.
        disable_tree_payload = str(os.getenv('FLATLAND_DISABLE_TREE_PAYLOAD_ENCODER', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
        if self.simplified_mode > 0 and disable_tree_payload:
            self.encoder_actor.use_tree_payload_encoder = False
            self.encoder_critic.use_tree_payload_encoder = False
            print("   - CORE override: tree_payload_encoder=OFF")

        # Actor-Critic Model (heads only, encoders are separate!)
        critic_hidden = max(192, int(self.hidden_size))
        self.actor_critic_model = ActorCriticModel(
            self.hidden_size, action_size, self.device,
            hidsize1=self.hidden_size,
            hidsize2=self.hidden_size,
            critic_hidsize1=critic_hidden,
            critic_hidsize2=critic_hidden
        )

        # Adaptive Learning Rates - Choose between Option A (consolidated) and Option B (synchronized decay)
        base_lr = self.learning_rate
        
        if self.optimizer_mode == 'single':
            # ===================================================================
            # SINGLE: CONSOLIDATED SINGLE OPTIMIZER (Recommended for MAPPO)
            # ===================================================================
            # All parameters (actor + critic) use ONE shared Adam optimizer.
            # This is the standard MAPPO design (OpenAI/IPPO literature).
            # Ensures symmetric learning rates and synchronized training pace.
            print("[Optimizer] Mode SINGLE: Consolidated single optimizer for actor + critic")
            
            all_params_raw = list(self.encoder_actor.parameters()) + \
                             list(self.actor_critic_model.actor.parameters()) + \
                             list(self.encoder_critic.parameters()) + \
                             list(self.actor_critic_model.critic.parameters())
            # Shared encoder mode can otherwise add identical tensors twice.
            # Deduplicate by tensor id to keep optimizer state compact and stable.
            seen_param_ids = set()
            all_params = []
            for p in all_params_raw:
                pid = id(p)
                if pid in seen_param_ids:
                    continue
                seen_param_ids.add(pid)
                all_params.append(p)
            
            self.optimizer = optim.AdamW(all_params, lr=base_lr)
            
            # Keep these for compatibility with existing code that references them
            self.optimizer_actor = self.optimizer
            self.optimizer_critic = self.optimizer
            self.optimizer_encoder_actor = self.optimizer
            self.optimizer_actor_head = self.optimizer
            self.optimizer_encoder_critic = self.optimizer
            self.optimizer_critic_head = self.optimizer
            
            # Store base LRs for reference (single base_lr for all)
            self.base_lr_all = base_lr
            self.base_lr_encoder_actor = base_lr
            self.base_lr_actor_head = base_lr
            self.base_lr_encoder_critic = base_lr
            self.base_lr_critic_head = base_lr
            
        else:  # multiple
            # ===================================================================
            # MULTIPLE: FOUR OPTIMIZERS WITH SYNCHRONIZED DECAY
            # ===================================================================
            # Keeps 4 separate optimizers (encoder_actor, actor_head, encoder_critic, critic_head)
            # but applies decay to ALL of them, not just actor.
            # This maintains symmetric learning rates: when actor_lr_factor changes,
            # all 4 optimizers decay proportionally.
            print("[Optimizer] Mode MULTIPLE: 4 optimizers with synchronized decay")
            
            self.optimizer_encoder_actor = optim.AdamW(
                self.encoder_actor.parameters(),
                lr=base_lr * 1.2
            )
            
            self.optimizer_actor_head = optim.AdamW(
                self.actor_critic_model.actor.parameters(),
                lr=base_lr * 1.0
            )
            
            self.optimizer_encoder_critic = optim.AdamW(
                self.encoder_critic.parameters(),
                lr=base_lr * 1.4
            )
            
            self.optimizer_critic_head = optim.AdamW(
                self.actor_critic_model.critic.parameters(),
                lr=base_lr * 1.1
            )
            
            self.optimizer_actor = self.optimizer_actor_head
            self.optimizer_critic = self.optimizer_critic_head
            self.optimizer = self.optimizer_actor_head
            self.base_lr_encoder_actor = base_lr * 1.2
            self.base_lr_actor_head = base_lr * 1.0
            self.base_lr_encoder_critic = base_lr * 1.4
            self.base_lr_critic_head = base_lr * 1.1
        
        self.actor_lr_factor = 1.0
        # Entropy rescue prevents late deterministic collapse around local minima.
        self.entropy_rescue_start_episode = 350      # ⬆️ Activate VERY early (was 700, now immediately!)
        self.entropy_floor = 0.70       # Higher floor to avoid premature deterministic collapse.
        self.entropy_recovery_scale = 6.0  # Stronger entropy rescue when floor is violated.

        self.loss_function = nn.SmoothL1Loss(beta=1.0)
        self.training_step_count = 0
        
        # TensorBoard writer (optional, can be set via set_tensorboard_writer)
        self.writer = None
        self._tensorboard_batch_counter = 0

        # Observation sanity statistics (gesammelt über 100 Episoden)
        self._obs_stat_buffer: list = []   # rohe Feature-Vektoren der letzten 100 Ep.
        self._obs_stat_interval = max(10, int(os.getenv('FLATLAND_DIAG_INTERVAL_EPISODES', '20')))
        # Tree-payload sanity statistics (nodes/edges validity over same window)
        self._tree_stat_buffer: list = []

        # Training-Kennzahlen je Batch (alle _obs_stat_interval Episoden geleert)
        self._stat_buf: dict = {
            'v_loss': [], 'p_loss': [], 'e_loss': [], 'aux_dl': [],
            'kl': [], 'ratio': [], 'entropy': [],
            'adv_mean': [], 'adv_std': [], 'grad_norm': [], 'grad_norm_post': [],
            'tree_grad_actor': [], 'tree_grad_critic': [],
            'tree_payload_grad_actor': [], 'tree_payload_grad_critic': [],
            'ret_min': [], 'ret_max': [],
            'comm_loss': [], 'action_div_loss': [], 'action_div_gate_ratio': [], 'total_loss': [],
            'action_hist': [],
            'action_hist_decision': [],
        }
        # Episode-Kennzahlen (Reward + Done-Rate)
        self._ep_stat_buf: dict = {'reward': [], 'done_frac': []}
        self._rollout_diag_window = 100
        self._rollout_diag_buf: dict = {
            'sp_match': deque(maxlen=self._rollout_diag_window),
            'sp_total': deque(maxlen=self._rollout_diag_window),
            'timeout_frac': deque(maxlen=self._rollout_diag_window),
            'ep_len': deque(maxlen=self._rollout_diag_window),
            'final_aux': deque(maxlen=self._rollout_diag_window),
            'done_frac': deque(maxlen=self._rollout_diag_window),
        }
        self.time_profile_enabled = str(os.getenv('FLATLAND_TIME_PROFILE', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
        self._time_profile_buf: dict = {
            'total': [],
            'pool_collect': [],
            'gae_prep': [],
            'concat_sample': [],
            'old_logprobs': [],
            'encode_batch': [],
            'forward_loss': [],
            'backward_clip': [],
            'optimizer_step': [],
        }
        self._perf_log_interval = max(1, int(os.getenv('FLATLAND_PERF_LOG_INTERVAL_EPISODES', str(self._obs_stat_interval))))
        self._episode_wall_t0 = time.perf_counter()
        self._episode_act_time = 0.0
        self._episode_step_time = 0.0
        self._episode_act_calls = 0
        self._episode_step_calls = 0
        self._episode_perf_buf: dict = {
            'episode_wall': deque(maxlen=self._rollout_diag_window),
            'act_time': deque(maxlen=self._rollout_diag_window),
            'step_time': deque(maxlen=self._rollout_diag_window),
            'act_calls': deque(maxlen=self._rollout_diag_window),
            'step_calls': deque(maxlen=self._rollout_diag_window),
        }

    def _comm_progress(self) -> float:
        start_ep = int(self.comm_reg_start_episode)
        full_ep = int(self.comm_reg_full_episode)
        if full_ep <= start_ep:
            return 1.0
        if self.episode_count <= start_ep:
            return 0.0
        return float(np.clip((self.episode_count - start_ep) / float(full_ep - start_ep), 0.0, 1.0))

    def set_tensorboard_writer(self, writer):
        """Set TensorBoard SummaryWriter for metric logging."""
        self.writer = writer
    
    def _log_batch_metrics(self, batch_metrics: dict):
        """Log per-batch metrics to TensorBoard if writer is available.
        
        Uses pattern: {policy_name}/training_value_{metric_name}
        matching the base_solver.py convention.
        """
        if self.writer is None:
            return
        
        policy_prefix = self.get_name()
        global_step = self._tensorboard_batch_counter
        
        # Loss metrics
        self.writer.add_scalar(f'{policy_prefix}/training_value_loss_policy', batch_metrics.get('p_loss', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_loss_value', batch_metrics.get('v_loss', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_loss_entropy', batch_metrics.get('e_loss', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_loss_aux_deadlock', batch_metrics.get('aux_dl', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_loss_action_diversity', batch_metrics.get('action_div', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_action_div_gate_ratio', batch_metrics.get('action_div_gate_ratio', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_loss_communication', batch_metrics.get('comm_loss', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_loss_total', batch_metrics.get('loss', 0.0), global_step)
        
        # PPO metrics
        self.writer.add_scalar(f'{policy_prefix}/training_value_ppo_kl', batch_metrics.get('kl', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_ppo_ratio', batch_metrics.get('ratio', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_ppo_entropy', batch_metrics.get('entropy', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_ppo_clip', batch_metrics.get('clip', 0.0), global_step)
        
        # Advantage metrics
        self.writer.add_scalar(f'{policy_prefix}/training_value_advantage_mean', batch_metrics.get('adv_mean', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_advantage_std', batch_metrics.get('adv_std', 0.0), global_step)
        
        # Gradient & learning rate
        self.writer.add_scalar(f'{policy_prefix}/training_value_grad_norm', batch_metrics.get('grad_norm', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_grad_norm_post', batch_metrics.get('grad_norm_post', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_lr_factor', batch_metrics.get('lr_factor', 1.0), global_step)
        
        # Communication metrics
        self.writer.add_scalar(f'{policy_prefix}/training_value_comm_gate', batch_metrics.get('comm_gate', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_comm_intent_wait', batch_metrics.get('intent_wait', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_comm_intent_go', batch_metrics.get('intent_go', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_comm_intent_yellow', batch_metrics.get('intent_yellow', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_comm_dropout', batch_metrics.get('comm_dropout', 0.0), global_step)
        self.writer.add_scalar(f'{policy_prefix}/training_value_comm_weight', batch_metrics.get('comm_weight', 0.0), global_step)
        
        # Policy weight
        self.writer.add_scalar(f'{policy_prefix}/training_value_policy_weight', batch_metrics.get('policy_weight', 0.0), global_step)
        
        # Action distribution (per action type)
        action_labels = ['DN', 'L', 'F', 'R', 'S']
        action_hist = batch_metrics.get('action_hist', np.zeros(5))
        for i, label in enumerate(action_labels):
            self.writer.add_scalar(f'{policy_prefix}/training_value_action_{label}', action_hist[i], global_step)
        action_hist_dec = batch_metrics.get('action_hist_decision', np.zeros(5))
        for i, label in enumerate(action_labels):
            self.writer.add_scalar(f'{policy_prefix}/training_value_action_decision_{label}', action_hist_dec[i], global_step)
        
        self._tensorboard_batch_counter += 1
    
    def _log_episode_metrics(self, episode_metrics: dict):
        """Log per-episode aggregate metrics to TensorBoard if writer is available.
        
        Uses pattern: {policy_name}/training_smoothed_{metric_name}
        (smoothed = aggregated over 100 episode buffer, matching base_solver.py convention)
        """
        if self.writer is None:
            return
        
        policy_prefix = self.get_name()
        ep_num = self.episode_count
        
        # Aggregate loss metrics
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_loss_policy', episode_metrics.get('p_loss_mean', 0.0), ep_num)
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_loss_value', episode_metrics.get('v_loss_mean', 0.0), ep_num)
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_loss_entropy', episode_metrics.get('e_loss_mean', 0.0), ep_num)
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_loss_aux_deadlock', episode_metrics.get('aux_dl_mean', 0.0), ep_num)
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_action_div_gate_ratio', episode_metrics.get('action_div_gate_ratio_mean', 0.0), ep_num)
        
        # Aggregate PPO metrics
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_ppo_kl', episode_metrics.get('kl_mean', 0.0), ep_num)
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_ppo_entropy', episode_metrics.get('entropy_mean', 0.0), ep_num)
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_ppo_ratio', episode_metrics.get('ratio_mean', 0.0), ep_num)
        
        # Aggregate gradient metric
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_grad_norm', episode_metrics.get('grad_norm_mean', 0.0), ep_num)
        
        # Episode performance (from reward shaper)
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_reward', episode_metrics.get('reward_mean', 0.0), ep_num)
        self.writer.add_scalar(f'{policy_prefix}/training_smoothed_done', episode_metrics.get('done_frac', 0.0), ep_num)

    def _apply_comm_schedule(self):
        progress = self._comm_progress()
        dropout_p = self.comm_dropout_early + (self.comm_dropout_late - self.comm_dropout_early) * progress
        self.encoder_actor.comm_dropout.p = float(dropout_p)
        self.encoder_critic.comm_dropout.p = float(dropout_p)
        return progress

    def _effective_clip_eps(self) -> float:
        # Narrow PPO trust region in late training to avoid destructive policy jumps.
        if self.episode_count < 200:
            # Early phase needs stronger actor movement to escape deadlock basins.
            return min(0.18, float(self.surrogate_eps_clip))
        if self.episode_count < self.stability_guard_start_episode:
            return float(self.surrogate_eps_clip)
        if self.episode_count >= self.stability_guard_hard_episode:
            return max(0.08, self.surrogate_eps_clip * 0.60)
        return max(0.10, self.surrogate_eps_clip * 0.75)

    def _effective_k_epochs(self) -> int:
        if self.episode_count < 200:
            return min(2, int(self.K_epoch))
        if self.episode_count < 300:
            return min(2, int(self.K_epoch))
        if self.episode_count >= self.stability_guard_hard_episode:
            return 1
        if self.episode_count >= self.stability_guard_start_episode:
            return min(2, int(self.K_epoch))
        return int(self.K_epoch)

    def _set_actor_lr_factor(self, factor: float):
        """Update learning rates based on actor_lr_factor (used for decay/recovery).
        
        SINGLE: Update single shared optimizer's lr
        MULTIPLE: Apply decay to ALL 4 optimizers (not just actor), ensuring synchronized pacing
        """
        self.actor_lr_factor = float(np.clip(factor, self.actor_lr_min_factor, self.actor_lr_max_factor))
        
        if self.optimizer_mode == 'single':
            # Single optimizer: scale all parameters by same factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr_all * self.actor_lr_factor
        else:  # Mode B
            # 4 optimizers: apply decay to ALL, not just actor (synchronized decay)
            for param_group in self.optimizer_encoder_actor.param_groups:
                param_group['lr'] = self.base_lr_encoder_actor * self.actor_lr_factor
            for param_group in self.optimizer_actor_head.param_groups:
                param_group['lr'] = self.base_lr_actor_head * self.actor_lr_factor
            # ⚠️ NEW: Apply decay to critic too! (previously was constant)
            for param_group in self.optimizer_encoder_critic.param_groups:
                param_group['lr'] = self.base_lr_encoder_critic * self.actor_lr_factor
            for param_group in self.optimizer_critic_head.param_groups:
                param_group['lr'] = self.base_lr_critic_head * self.actor_lr_factor

    def _set_critic_lr_defaults(self):
        """Reset critic LRs to default (only needed in MULTIPLE mode; SINGLE uses synchronized decay)."""
        if self.optimizer_mode == 'multiple':
            for param_group in self.optimizer_encoder_critic.param_groups:
                param_group['lr'] = self.base_lr_encoder_critic
            for param_group in self.optimizer_critic_head.param_groups:
                param_group['lr'] = self.base_lr_critic_head

    def get_name(self):
        return self.__class__.__name__

    def act(self, handle, temporal_state, eps=None):
        """
        Action selection using TEMPORAL ACTOR encoder
        
        Args:
            temporal_state: [(obs_t-2, opp_t-2), (obs_t-1, opp_t-1), (obs_t, opp_t)]
        """
        t0 = time.perf_counter() if self.time_profile_enabled else 0.0
        with torch.no_grad():
            emb = self.encoder_actor.forward_agent(temporal_state, handle)
            emb_batch = emb.unsqueeze(0)  # (1, hidden_dim)
            dist = self.actor_critic_model.get_actor_dist(emb_batch)
            action = dist.sample()

        if self.time_profile_enabled:
            self._episode_act_time += (time.perf_counter() - t0)
            self._episode_act_calls += 1

        return action.item()

    def step(self, handle, state, action, reward, next_state, done):
        """Store transition - state is now temporal sequence!"""
        t0 = time.perf_counter() if self.time_profile_enabled else 0.0
        aux_deadlock = self._extract_deadlock_label_from_temporal_state(next_state)
        transition = (state, action, reward, next_state, done, aux_deadlock)
        self.current_episode_memory.push_transition(handle, transition)
        # Observation-Statistik: letzten Frame des temporalen Fensters sammeln
        if not isinstance(state, (list, tuple)) or len(state) == 0:
            raise ValueError("step() expected non-empty temporal state sequence")
        last_state_step = state[-1]
        if not isinstance(last_state_step, (list, tuple)) or len(last_state_step) < 1:
            raise ValueError("step() expected temporal state entries as tuple/list with base observation")
        last_frame = np.asarray(last_state_step[0], dtype=np.float32).reshape(-1)
        if last_frame.shape[0] > 0:
            self._obs_stat_buffer.append(last_frame)

        # Tree-payload stats: verify variable node/edge counts and edge index validity.
        payload = state[-1][2] if isinstance(state[-1], (list, tuple)) and len(state[-1]) >= 3 else {}
        if not isinstance(payload, dict):
            raise ValueError("step() expected tree payload as dict in temporal state")

        nodes = payload.get('nodes', [])
        edges = payload.get('edges', [])
        if not isinstance(nodes, list):
            raise TypeError("tree payload 'nodes' must be a list")
        if not isinstance(edges, list):
            raise TypeError("tree payload 'edges' must be a list")

        node_count = int(len(nodes))
        edge_count = int(len(edges))
        invalid_edges = 0
        unmapped_edges = 0
        node_feat_sum = np.zeros(TreePayloadEncoder.NODE_DIM, dtype=np.float64)
        node_feat_sq_sum = np.zeros(TreePayloadEncoder.NODE_DIM, dtype=np.float64)
        edge_feat_sum = np.zeros(TreePayloadEncoder.EDGE_DIM, dtype=np.float64)
        edge_feat_sq_sum = np.zeros(TreePayloadEncoder.EDGE_DIM, dtype=np.float64)
        node_feat_count = 0
        edge_feat_count = 0
        idx_exact = {}
        idx_simple = {}

        for i, n in enumerate(nodes):
            if not isinstance(n, dict):
                raise TypeError(f"tree payload node at index {i} must be a dict")
            pos = n.get('pos', (0, 0))
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                raise ValueError(f"tree payload node at index {i} has invalid pos field")
            d = int(n.get('dir', 0))
            dep = int(n.get('depth', i))
            pr, pc = int(pos[0]), int(pos[1])
            idx_exact[(pr, pc, d, dep)] = i
            idx_simple[(pr, pc, d)] = i
            n_feat = TreePayloadEncoder._encode_node_feature(n).astype(np.float64)
            node_feat_sum += n_feat
            node_feat_sq_sum += (n_feat * n_feat)
            node_feat_count += 1

        for e in edges:
            if not isinstance(e, dict):
                raise TypeError("tree payload edge entries must be dicts")

            # Support both positional edge schema (src_pos/dst_pos)
            # and optional index schema (src/dst).
            si = None
            di = None

            if 'src' in e and 'dst' in e:
                si = int(e.get('src'))
                di = int(e.get('dst'))
            else:
                sp = e.get('src_pos', (0, 0))
                dp = e.get('dst_pos', (0, 0))
                if not isinstance(sp, (list, tuple)) or len(sp) < 2:
                    raise ValueError("tree payload edge has invalid src_pos")
                if not isinstance(dp, (list, tuple)) or len(dp) < 2:
                    raise ValueError("tree payload edge has invalid dst_pos")
                sd = int(e.get('src_dir', 0))
                sdep = int(e.get('src_depth', 0))
                dd = int(e.get('dst_dir', 0))
                ddep = int(e.get('dst_depth', 0))
                spk = (int(sp[0]), int(sp[1]), sd, sdep)
                dpk = (int(dp[0]), int(dp[1]), dd, ddep)

                si = idx_exact.get(spk)
                di = idx_exact.get(dpk)
                if si is None:
                    si = idx_simple.get((spk[0], spk[1], spk[2]))
                if di is None:
                    di = idx_simple.get((dpk[0], dpk[1], dpk[2]))

            if si is None or di is None:
                # Endpoint cannot be mapped to currently materialized nodes.
                # This can happen with aggressive node budgets / contraction.
                unmapped_edges += 1
                continue
            if si < 0 or di < 0 or si >= node_count or di >= node_count:
                unmapped_edges += 1
                continue

            e_feat = TreePayloadEncoder._encode_edge_feature(e).astype(np.float64)
            edge_feat_sum += e_feat
            edge_feat_sq_sum += (e_feat * e_feat)
            edge_feat_count += 1

        self._tree_stat_buffer.append({
            'nodes': node_count,
            'edges': edge_count,
            'invalid_edges': int(invalid_edges),
            'unmapped_edges': int(unmapped_edges),
            'empty_payload': 1 if (node_count == 0 and edge_count == 0) else 0,
            'node_feat_sum': node_feat_sum,
            'node_feat_sq_sum': node_feat_sq_sum,
            'node_feat_count': int(node_feat_count),
            'edge_feat_sum': edge_feat_sum,
            'edge_feat_sq_sum': edge_feat_sq_sum,
            'edge_feat_count': int(edge_feat_count),
        })
        if self.time_profile_enabled:
            self._episode_step_time += (time.perf_counter() - t0)
            self._episode_step_calls += 1

    @staticmethod
    def _extract_deadlock_label_from_temporal_state(temporal_state) -> float:
        """Build a robust [0,1] deadlock-risk label from temporal payload data."""
        if not isinstance(temporal_state, (list, tuple)) or len(temporal_state) == 0:
            raise ValueError("temporal_state must be a non-empty sequence")
        last_step = temporal_state[-1]

        payload = {}
        if isinstance(last_step, (list, tuple)) and len(last_step) >= 3:
            if not isinstance(last_step[2], dict):
                raise TypeError("temporal_state payload entry must be a dict")
            payload = last_step[2]

        nodes = payload.get("nodes", []) if isinstance(payload.get("nodes", []), list) else []
        edges = payload.get("edges", []) if isinstance(payload.get("edges", []), list) else []

        if not nodes and not edges:
            return 0.0

        max_deadlock = max((float(n.get("deadlock_risk", 0.0)) for n in nodes), default=0.0)
        oncoming_node = 1.0 if any(bool(n.get("has_oncoming", False)) for n in nodes) else 0.0
        oncoming_edge = 1.0 if any(bool(e.get("has_oncoming_edge", False)) for e in edges) else 0.0
        dead_end_ratio = 0.0
        if nodes:
            dead_end_ratio = float(sum(1 for n in nodes if int(n.get("num_transitions", 0)) == 0)) / float(len(nodes))

        risk = max(
            max_deadlock,
            0.70 * oncoming_node,
            0.70 * oncoming_edge,
            0.45 * dead_end_ratio,
        )
        return float(np.clip(risk, 0.0, 1.0))

    @staticmethod
    def _extract_local_shortest_action_from_temporal_state(temporal_state) -> Optional[int]:
        """Infer shortest-path action from payload edges (1=L, 2=F, 3=R)."""
        def _safe_float(value: Any, default: float = 0.0) -> float:
            if value is None:
                return default
            try:
                out = float(value)
            except (TypeError, ValueError):
                return default
            if not np.isfinite(out):
                return default
            return out

        if not isinstance(temporal_state, (list, tuple)) or len(temporal_state) == 0:
            return None
        last_step = temporal_state[-1]
        if not isinstance(last_step, (list, tuple)) or len(last_step) < 3:
            return None
        payload = last_step[2]
        if not isinstance(payload, dict):
            return None
        edges = payload.get("edges", [])
        if not isinstance(edges, list) or len(edges) == 0:
            return None

        best_action = None
        best_score = float("inf")
        for edge in edges:
            if not isinstance(edge, dict):
                continue

            a_l = _safe_float(edge.get("action_left", 0.0), 0.0)
            a_f = _safe_float(edge.get("action_forward", 0.0), 0.0)
            a_r = _safe_float(edge.get("action_right", 0.0), 0.0)
            if a_l >= a_f and a_l >= a_r:
                action = 1
                confidence = a_l
            elif a_f >= a_l and a_f >= a_r:
                action = 2
                confidence = a_f
            else:
                action = 3
                confidence = a_r

            if confidence <= 0.0:
                continue

            dst = _safe_float(edge.get("dst_dist_to_target", 1.0), 1.0)
            deadlock_pen = 0.10 * _safe_float(edge.get("dst_deadlock_risk", 0.0), 0.0)
            score = dst + deadlock_pen
            if score < best_score:
                best_score = score
                best_action = action

        return best_action

    def _extract_action_diversity_gate_from_temporal_state(self, temporal_state) -> float:
        """Return gate in [0,1] where diversity shaping should be active.

        Gate is driven by tree-payload conflict/branching signals instead of
        fixed observation indices.
        """
        if not isinstance(temporal_state, (list, tuple)) or len(temporal_state) == 0:
            raise ValueError("temporal_state must be a non-empty sequence")
        last_step = temporal_state[-1]

        payload = {}
        if isinstance(last_step, (list, tuple)) and len(last_step) >= 3:
            if not isinstance(last_step[2], dict):
                raise TypeError("temporal_state payload entry must be a dict")
            payload = last_step[2]

        nodes = payload.get("nodes", []) if isinstance(payload.get("nodes", []), list) else []
        if len(nodes) == 0:
            return 0.0

        branching_ratio = float(sum(1 for n in nodes if int(n.get("num_transitions", 0)) > 1)) / float(len(nodes))
        max_deadlock = max((float(n.get("deadlock_risk", 0.0)) for n in nodes), default=0.0)
        has_oncoming = 1.0 if any(bool(n.get("has_oncoming", False)) for n in nodes) else 0.0
        gate_signal = max(branching_ratio, max_deadlock, 0.75 * has_oncoming)
        thr = float(getattr(self, 'action_diversity_gate_threshold', 0.5))
        if gate_signal < thr:
            return 0.0
        return float(np.clip(gate_signal, 0.0, 1.0))

    def _convert_transitions_to_torch_tensors(self, transitions_array):
        """Convert episode transitions to tensors"""
        state_list, action_list, reward_list, state_next_list, dones_list, aux_deadlock_list = [], [], [], [], [], []

        for transition in transitions_array:
            if len(transition) >= 6:
                state_i, action_i, reward_i, state_next_i, done_i, aux_deadlock_i = transition
            else:
                state_i, action_i, reward_i, state_next_i, done_i = transition
                aux_deadlock_i = self._extract_deadlock_label_from_temporal_state(state_next_i)

            state_list.append(state_i)
            action_list.append(action_i)
            reward_list.append(reward_i)
            state_next_list.append(state_next_i)
            dones_list.append(1 if done_i else 0)
            aux_deadlock_list.append(float(aux_deadlock_i))

        actions = torch.tensor(action_list, dtype=torch.long).to(self.device)
        rewards = torch.tensor(reward_list, dtype=torch.float).to(self.device) * self.reward_scale
        dones = torch.tensor(dones_list, dtype=torch.float).to(self.device)
        aux_deadlock = torch.tensor(aux_deadlock_list, dtype=torch.float).to(self.device)

        return state_list, actions, rewards, state_next_list, dones, aux_deadlock
    
    def _compute_gae(self, rewards, values, dones, next_values):
        """Generalized Advantage Estimation (GAE).

        Schulman, Moritz, Levine, Jordan, Abbeel (2016)
        "High-Dimensional Continuous Control Using Generalized Advantage
        Estimation", ICLR. arXiv:1506.02438

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + gamma * lambda * A_{t+1} * (1 - done_t)
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.discount * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.discount * self.gae_lambda * gae * (1 - dones[t])
            advantages[t] = gae
        
        returns = advantages + values
        
        return advantages, returns

    def train_net_accumulated(self):
        """Training loop - ORIGINAL VERSION (no early sampling, no cached logprobs)"""
        self.encoder_actor.train()
        self.encoder_critic.train()
        self.actor_critic_model.train()

        profile_enabled = bool(getattr(self, 'time_profile_enabled', True))
        update_t0 = time.perf_counter() if profile_enabled else 0.0
        timing = {}

        def _tic() -> float:
            return time.perf_counter()

        def _add_t(name: str, dt: float):
            timing[name] = timing.get(name, 0.0) + float(dt)

        episode_data = []
        trajectory_count = 0
        num_recent_episodes = len(self.accumulated_episodes)
        
        if num_recent_episodes < self.max_episodes_in_training_memory:
            if self.show_pre_train_debug_msg:
                print(f"\n🔍 Collect episodes {num_recent_episodes}/{self.max_episodes_in_training_memory}")
            return

        t0 = _tic() if profile_enabled else 0.0
        training_episodes = list(self.accumulated_episodes)
        if profile_enabled:
            _add_t('pool_collect', _tic() - t0)
        
        # Replay sampling first, then GAE on selected trajectories only.
        t0 = _tic() if profile_enabled else 0.0
        trajectory_pool = []
        total_samples_pool = 0
        for episode_memory in training_episodes:
            for handle in range(len(episode_memory)):
                agent_episode_history = episode_memory.get_transitions(handle)
                traj_len = len(agent_episode_history)
                if traj_len <= 0:
                    continue
                trajectory_pool.append((agent_episode_history, traj_len))
                total_samples_pool += traj_len

        if len(trajectory_pool) == 0 or total_samples_pool == 0:
            print("⚠️ No transitions to train on!")
            return

        trajectory_count = len(trajectory_pool)
        total_possible_batches = (total_samples_pool + self.batch_size - 1) // self.batch_size
        if self.max_batches_per_training is not None:
            num_batches = min(self.max_batches_per_training, total_possible_batches)
            base_samples_to_use = min(total_samples_pool, num_batches * self.batch_size)
        else:
            num_batches = max(1, int(total_possible_batches * self.batch_fraction))
            base_samples_to_use = min(total_samples_pool, num_batches * self.batch_size)

        replay_samples_to_use = int(max(self.batch_size, round(base_samples_to_use * self.replay_sample_percent)))
        samples_to_use = int(min(total_samples_pool, replay_samples_to_use))

        if samples_to_use >= total_samples_pool:
            sampled_order = list(range(len(trajectory_pool)))
        else:
            sampled_order = np.random.permutation(len(trajectory_pool)).tolist()

        selected_trajectories = []
        selected_samples = 0
        for idx in sampled_order:
            traj, traj_len = trajectory_pool[idx]
            selected_trajectories.append(traj)
            selected_samples += int(traj_len)
            if selected_samples >= samples_to_use:
                break

        for agent_episode_history in selected_trajectories:
            state_tuples, actions, rewards, state_next_tuples, dones, aux_deadlock = \
                self._convert_transitions_to_torch_tensors(agent_episode_history)

            # Compute GAE (encoder calls are batched internally)
            with torch.no_grad():
                states_critic = self.encoder_critic.forward_batch(state_tuples)
                values = torch.squeeze(self.actor_critic_model.critic(states_critic), dim=-1)
                # FIXED: Removed torch.clamp(values, -5, 5) — was truncating returns [-19.85, +2.97].
                # Let critic learn to predict true return values without artificial bounds.

                next_states_critic = self.encoder_critic.forward_batch(state_next_tuples)
                next_values = torch.squeeze(self.actor_critic_model.critic(next_states_critic), dim=-1)
                # Critic now free to fit the full range of returns for better GAE advantage signals.

                traj_gae_advantages, traj_gae_returns = self._compute_gae(
                    rewards, values, dones, next_values
                )

            episode_data.append((
                state_tuples,
                actions,
                traj_gae_advantages,
                traj_gae_returns,
                aux_deadlock,
            ))
        if profile_enabled:
            _add_t('gae_prep', _tic() - t0)
        
        if len(episode_data) == 0:
            print("⚠️ No transitions to train on!")
            return
        
        # Concatenate all episode data
        t0 = _tic() if profile_enabled else 0.0
        all_state_tuples = []
        all_actions = []
        all_gae_advantages = []
        all_gae_returns = []
        all_aux_deadlock = []

        for ep_idx, (ep_states, ep_actions, ep_advantages, ep_returns, ep_aux_deadlock) in enumerate(episode_data):
            all_state_tuples.extend(ep_states)
            all_actions.append(ep_actions)
            all_gae_advantages.append(ep_advantages)
            all_gae_returns.append(ep_returns)
            all_aux_deadlock.append(ep_aux_deadlock)
        
        all_actions = torch.cat(all_actions, dim=0)
        all_gae_advantages = torch.cat(all_gae_advantages, dim=0)
        all_gae_returns = torch.cat(all_gae_returns, dim=0)
        all_aux_deadlock = torch.cat(all_aux_deadlock, dim=0)

        total_samples = len(all_state_tuples)

        # Determine batch configuration
        total_possible_batches = (total_samples + self.batch_size - 1) // self.batch_size

        if self.max_batches_per_training is not None:
            num_batches = min(self.max_batches_per_training, total_possible_batches)
        else:
            num_batches = max(1, int(total_possible_batches * self.batch_fraction))
        samples_to_use = total_samples
        if profile_enabled:
            _add_t('concat_sample', _tic() - t0)
        
        if self.show_pre_train_debug_msg:
            print(f"📦 Using {samples_to_use}/{total_samples} samples ({samples_to_use/total_samples*100:.1f}%)")
            k_epochs_eff = self._effective_k_epochs()
            print(
                f"📦 Batch Config: {num_batches} batches (batch_size={self.batch_size}) over {k_epochs_eff} epochs "
                f"| replay_sample_percent={self.replay_sample_percent:.2f} | trajectories={trajectory_count}"
            )
        
        # 🎯 KRITISCH: Berechne old_logprobs VOR dem Training
        # Dies ist die EINZIGE korrekte Methode für PPO!
        batch_size_encoding = 256  # ⚡ WICHTIG: Außerhalb definieren!
        
        if self.show_pre_train_debug_msg:
            print(f"\n🔍 Computing initial old_logprobs for {len(all_state_tuples)} samples...")
        
        t0 = _tic() if profile_enabled else 0.0
        with torch.no_grad():
            # Encode states in batches to avoid OOM
            all_old_logprobs = []
            
            for i in range(0, len(all_state_tuples), batch_size_encoding):
                batch_states = all_state_tuples[i:i+batch_size_encoding]
                batch_acts = all_actions[i:i+batch_size_encoding]
                
                states_enc = self.encoder_actor.forward_batch(batch_states)
                logits = self.actor_critic_model.actor(states_enc)
                old_lp = Categorical(logits=logits).log_prob(batch_acts)
                all_old_logprobs.append(old_lp)
            
            all_old_logprobs = torch.cat(all_old_logprobs, dim=0)
        if profile_enabled:
            _add_t('old_logprobs', _tic() - t0)
        
        if self.show_pre_train_debug_msg:
            print(f"✅ Initial old_logprobs computed (mean={all_old_logprobs.mean().item():.4f})")
        
        k_epochs_eff = self._effective_k_epochs()
        total_iterations = k_epochs_eff * num_batches
        current_iteration = 0
        hard_spike_batches_total = 0
        hard_spike_streak = 0
        _n_batches_update = 0
        action_hist_total = torch.zeros(self.action_size, dtype=torch.long)
        action_hist_decision_total = torch.zeros(self.action_size, dtype=torch.long)
        action_labels = ['DN', 'L', 'F', 'R', 'S'] if self.action_size == 5 else [f'A{i}' for i in range(self.action_size)]

        if self.show_progress_bar:
            print("")
        
        # Training epochs
        for k_loop in range(k_epochs_eff):
            break_current_epoch = False
            # Keep old_logprobs fixed for the whole PPO update cycle.
            # Recomputing them after policy updates weakens PPO's trust-region effect.
            
            # Shuffle data each epoch for better training
            indices = torch.randperm(samples_to_use)
            
            for batch_idx in range(num_batches):
                current_iteration += 1
                
               
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, samples_to_use)
                batch_indices = indices[start_idx:end_idx]

                # Skip tiny tail batches; they create very noisy gradients and
                # unstable diagnostics (e.g. 2-10 sample action histograms).
                if batch_indices.numel() < 16:
                    continue
                
                batch_state_tuples = [all_state_tuples[i] for i in batch_indices]
                batch_actions = all_actions[batch_indices]
                batch_action_hist = torch.bincount(batch_actions.detach().cpu(), minlength=self.action_size)
                batch_action_hist_decision = torch.zeros(self.action_size, dtype=torch.long)
                action_hist_total += batch_action_hist
                batch_gae_advantages = all_gae_advantages[batch_indices]
                batch_gae_returns = all_gae_returns[batch_indices]
                batch_aux_deadlock = all_aux_deadlock[batch_indices]
                batch_old_logprobs = all_old_logprobs[batch_indices]  # 🎯 Pre-computed!

                comm_progress = self._apply_comm_schedule()

                # Encode states
                t0 = _tic() if profile_enabled else 0.0
                states_actor = self.encoder_actor.forward_batch(batch_state_tuples)
                states_critic = self.encoder_critic.forward_batch(batch_state_tuples)
                if profile_enabled:
                    _add_t('encode_batch', _tic() - t0)

                # ⚠️ NaN Check: Detect gradient explosion early
                if torch.isnan(states_actor).any():
                    print(f"\n⚠️ WARNING: NaN detected in states_actor at epoch {k_loop}, batch {batch_idx}")
                    print("   Skipping this batch to prevent crash...")
                    continue
                if torch.isnan(states_critic).any():
                    print(f"\n⚠️ WARNING: NaN detected in states_critic at epoch {k_loop}, batch {batch_idx}")
                    print("   Skipping this batch to prevent crash...")
                    continue

                # Evaluate actions (NEW policy - WITH gradients!)
                t0 = _tic() if profile_enabled else 0.0
                logits = self.actor_critic_model.actor(states_actor)
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(batch_actions)
                
                # ⚡ CRITICAL: Clip logprobs to prevent extreme ratios
                logprobs = torch.clamp(logprobs, -10, 0)  # Log-probs are always negative
                batch_old_logprobs = torch.clamp(batch_old_logprobs, -10, 0)
                
                dist_entropy = dist.entropy()
                entropy_mean = dist_entropy.mean().item()
                probs = torch.softmax(logits, dim=-1)

                # 🔍 DEBUG: Verify old_logprobs are different from new ones (NUR Epoch 1)
                if self.show_debug_msg:
                    if k_loop == 0 and batch_idx < 3:  # Erste Epoch, erste 3 Batches
                        diff_mean = (logprobs - batch_old_logprobs).abs().mean().item()
                        ratio_raw = torch.exp(logprobs - batch_old_logprobs).mean().item()
                        print(f"🔍 E1 B{batch_idx}: Diff={diff_mean:.6f} Ratio_raw={ratio_raw:.4f}", flush=True)
                
                state_values = torch.squeeze(self.actor_critic_model.critic(states_critic), dim=-1)
                
                # ------------------------------------------------------------
                # PPO Clipped Surrogate Objective
                # Schulman et al. (2017) "Proximal Policy Optimization Algorithms",
                # arXiv:1707.06347 -- L^CLIP = E[ min(r_t * A_t,
                #                                 clip(r_t, 1-eps, 1+eps) * A_t) ]
                #
                # Wichtig: NICHT zusätzlich r_t selbst hart clampen, sonst
                # entfernt man genau die seltenen großen Lernsignale, die
                # man eigentlich braucht (z.B. STOP-vor-Merge -> +Done-Bonus).
                # Hier nur ein WEITER NaN-Schutz [0.05, 20] -- der eigentliche
                # Trust-Region-Mechanismus passiert über min(surr1, surr2).
                # ------------------------------------------------------------
                ratios = torch.exp(logprobs - batch_old_logprobs)
                ratios = torch.clamp(ratios, 0.05, 20.0)
                
                # Normalize advantages
                advantages = batch_gae_advantages
                eps = 1e-8
                
                # Robust against tiny mini-batches: use population std and
                # guard against near-zero variance to avoid NaN explosions.
                raw_adv_mean = advantages.mean().item()
                raw_adv_std = advantages.std(unbiased=False).item()

                if advantages.numel() <= 1 or raw_adv_std < 1e-8:
                    advantages_normalized = advantages - advantages.mean()
                else:
                    advantages_normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)
                # Standard PPO: Advantage-Normalisierung pro Mini-Batch
                # (Andrychowicz et al. 2021 "What Matters in On-Policy RL?",
                # arXiv:2006.05990 -- Empfehlung #4). Eine zusätzliche
                # Skalierung (vorher *0.3) drosselt alle Updates und hat das
                # Lernen in einem 60%-Lokal-Optimum gefangen gehalten.
                advantages = torch.clamp(
                    advantages_normalized,
                    -float(self.adv_clip_abs),
                    float(self.adv_clip_abs),
                )

                # PPO loss
                clip_eps_eff = self._effective_clip_eps()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_eps_eff,
                                   1.0 + clip_eps_eff) * advantages

                policy_loss_component = -torch.min(surr1, surr2).mean()
                value_loss_component = self.loss_function(state_values, batch_gae_returns)
                entropy_loss_component = -dist_entropy.mean()
                action_diversity_loss_component = torch.tensor(0.0, device=self.device)
                action_diversity_gate_ratio = 0.0
                if self.action_size == 5:
                    if bool(getattr(self, 'action_diversity_gate_enabled', True)):
                        gate_vals = [
                            self._extract_action_diversity_gate_from_temporal_state(ts)
                            for ts in batch_state_tuples
                        ]
                        adiv_gate = torch.tensor(
                            gate_vals,
                            dtype=probs.dtype,
                            device=self.device,
                        )
                    else:
                        adiv_gate = torch.ones(probs.shape[0], dtype=probs.dtype, device=self.device)

                    gate_sum = float(adiv_gate.sum().item())
                    action_diversity_gate_ratio = float(adiv_gate.mean().item())

                    # Soft constraints via squared hinge losses:
                    # 1) keep forward probability below a soft cap
                    # 2) keep combined left+right probability above a soft floor
                    # 3) keep idle actions (DO_NOTHING + STOP) below a soft cap
                    if gate_sum >= 1.0:
                        gate_active = adiv_gate > 0.0
                        if bool(torch.any(gate_active).item()):
                            gated_actions = batch_actions[gate_active]
                            batch_action_hist_decision = torch.bincount(
                                gated_actions.detach().cpu(),
                                minlength=self.action_size,
                            )
                            action_hist_decision_total += batch_action_hist_decision

                        gate_col = adiv_gate.unsqueeze(1)
                        mean_probs = (probs * gate_col).sum(dim=0) / max(gate_sum, 1.0)
                        forward_prob = mean_probs[2]
                        lr_prob = mean_probs[1] + mean_probs[3]
                        idle_prob = mean_probs[0] + mean_probs[4]
                        forward_excess = torch.relu(forward_prob - self.forward_prob_soft_max)
                        lr_shortfall = torch.relu(self.lr_prob_soft_min - lr_prob)
                        idle_excess = torch.relu(idle_prob - self.idle_prob_soft_max)
                        action_diversity_loss_component = (
                            forward_excess.pow(2)
                            + 0.5 * lr_shortfall.pow(2)
                            + 0.30 * idle_excess.pow(2)
                        )
                if self.weight_aux_deadlock > 0.0:
                    deadlock_logits = torch.squeeze(self.actor_critic_model.deadlock_head(states_actor), dim=-1)
                    aux_targets = torch.clamp(batch_aux_deadlock, 0.0, 1.0)
                    pos_weight = torch.full_like(aux_targets, self.aux_deadlock_pos_weight)
                    aux_deadlock_loss_component = nn.functional.binary_cross_entropy_with_logits(
                        deadlock_logits,
                        aux_targets,
                        pos_weight=pos_weight,
                    )
                else:
                    aux_deadlock_loss_component = torch.zeros((), device=self.device)
                comm_loss_component = self.encoder_actor.last_comm_reg + self.encoder_critic.last_comm_reg
                ratio_mean = ratios.mean().item()
                approx_kl = torch.abs((batch_old_logprobs - logprobs).mean()).item()

                # ========================================================================
                # WEIGHT SCHEDULING: In SIMPLIFIED_MODE, use fixed weights only
                # ========================================================================
                if self.simplified_mode:
                    # CORE PPO: Fixed weights, no dynamic scheduling
                    policy_weight_eff = self.weight_policy
                    value_weight_eff = self.weight_loss
                    entropy_weight_eff = self.weight_entropy
                    comm_weight_eff = 0.0
                else:
                    # COMPLEX MODE: Dynamic weights based on KL, ratio, entropy
                    gate_mean = (self.encoder_actor.last_comm_gate_mean + self.encoder_critic.last_comm_gate_mean) / 2.0
                    comm_boost = max(0.0, (gate_mean - self.comm_gate_target) / max(self.comm_gate_target, 1e-6))
                    comm_weight_eff = self.weight_comm * comm_progress * (1.0 + min(comm_boost, 2.0))

                    policy_weight_eff = self.weight_policy
                    entropy_weight_eff = self.weight_entropy
                    if self.episode_count >= self.entropy_rescue_start_episode and entropy_mean < self.entropy_floor:
                        entropy_weight_eff = max(entropy_weight_eff, self.weight_entropy * self.entropy_recovery_scale)
                        policy_weight_eff *= 0.90

                    ratio_soft_viol = (ratio_mean > self.ratio_guard_soft) or (ratio_mean < self.ratio_guard_soft_low)
                    ratio_hard_viol = (ratio_mean > self.ratio_guard_hard) or (ratio_mean < self.ratio_guard_hard_low)

                    if approx_kl > self.ppo_target_kl or ratio_soft_viol:
                        policy_weight_eff *= 0.75
                        entropy_weight_eff *= 0.7

                    if approx_kl > self.ppo_max_kl or ratio_hard_viol:
                        # Keep small actor updates alive during hard spikes to avoid
                        # long Pw=0 plateaus where policy stops improving.
                        policy_weight_eff = max(policy_weight_eff * 0.55, 0.35)
                        entropy_weight_eff *= 0.4
                        comm_weight_eff *= 1.35
                        hard_spike_batches_total += 1
                        hard_spike_streak += 1
                    elif approx_kl > self.ppo_emergency_kl or ratio_soft_viol:
                        hard_spike_streak = max(hard_spike_streak, 1)
                    else:
                        hard_spike_streak = 0

                    if approx_kl > self.ppo_emergency_kl_hard:
                        policy_weight_eff = max(policy_weight_eff * 0.45, 0.30)
                        entropy_weight_eff *= 0.25
                        comm_weight_eff *= 1.45
                        hard_spike_batches_total += 1
                        hard_spike_streak += 1
                    
                    # Keep policy and critic progress coupled: when critic error is high,
                    # increase critic pressure and slightly damp policy updates.
                    value_weight_eff = self.weight_loss
                    value_loss_scalar = float(value_loss_component.detach().item())
                    if value_loss_scalar > 0.90:
                        value_weight_eff *= 1.35
                        policy_weight_eff *= 0.97
                    elif value_loss_scalar < 0.45:
                        value_weight_eff *= 0.90
                        policy_weight_eff *= 1.05

                    # When PPO is overly conservative (very low KL, ratio near 1,
                    # near-zero policy loss), softly boost actor weight.
                    if (
                        abs(float(policy_loss_component.detach().item())) < 0.003
                        and approx_kl < 0.010
                        and abs(ratio_mean - 1.0) < 0.03
                    ):
                        policy_weight_eff *= 1.30

                loss = \
                    policy_weight_eff * policy_loss_component \
                    + value_weight_eff * value_loss_component \
                    + entropy_weight_eff * entropy_loss_component \
                    + self.weight_action_diversity * action_diversity_loss_component \
                    + self.weight_aux_deadlock * aux_deadlock_loss_component \
                    + comm_weight_eff * comm_loss_component
                if profile_enabled:
                    _add_t('forward_loss', _tic() - t0)

                # Backward pass
                t0 = _tic() if profile_enabled else 0.0
                self.optimizer_encoder_actor.zero_grad()
                self.optimizer_actor_head.zero_grad()
                self.optimizer_encoder_critic.zero_grad()
                self.optimizer_critic_head.zero_grad()
                
                # Check for NaN in loss before backward
                if torch.isnan(loss):
                    print(f"\n⚠️ WARNING: NaN detected in loss at epoch {k_loop}, batch {batch_idx}")
                    print("   Skipping backward pass to prevent crash...")
                    continue
                
                loss.backward()
                
                # Gradient clipping (returns PRE-clip norm). We track PRE and POST.
                def _dedup_params(param_list):
                    out = []
                    seen = set()
                    for p in param_list:
                        if p is None or (not p.requires_grad):
                            continue
                        pid = id(p)
                        if pid in seen:
                            continue
                        seen.add(pid)
                        out.append(p)
                    return out

                def _grad_norm_params(param_list):
                    sq = 0.0
                    for p in param_list:
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        if torch.isnan(g).any() or torch.isinf(g).any():
                            continue
                        sq += float(torch.sum(g * g).item())
                    return float(math.sqrt(max(sq, 0.0)))

                if self.optimizer_mode == 'single':
                    all_params = []
                    for group in self.optimizer.param_groups:
                        all_params.extend(group['params'])
                    all_params = _dedup_params(all_params)
                    grad_norm_pre = torch.nn.utils.clip_grad_norm_(
                        all_params,
                        max_norm=float(self.max_grad_norm_single)
                    )
                    grad_norm_post = _grad_norm_params(all_params)
                else:
                    actor_params = _dedup_params(
                        list(self.encoder_actor.parameters()) +
                        list(self.actor_critic_model.actor.parameters())
                    )
                    critic_params = _dedup_params(
                        list(self.encoder_critic.parameters()) +
                        list(self.actor_critic_model.critic.parameters())
                    )
                    grad_norm_actor_pre = torch.nn.utils.clip_grad_norm_(
                        actor_params,
                        max_norm=float(self.max_grad_norm_actor)
                    )
                    grad_norm_critic_pre = torch.nn.utils.clip_grad_norm_(
                        critic_params,
                        max_norm=float(self.max_grad_norm_critic)
                    )
                    grad_norm_pre = max(float(grad_norm_actor_pre.item()), float(grad_norm_critic_pre.item()))
                    grad_norm_post = max(_grad_norm_params(actor_params), _grad_norm_params(critic_params))
                if profile_enabled:
                    _add_t('backward_clip', _tic() - t0)

                grad_norm = float(grad_norm_pre.item()) if isinstance(grad_norm_pre, torch.Tensor) else float(grad_norm_pre)

                # Only skip on true NaN/Inf — clip_grad_norm_ already bounds large norms.
                # The old "> grad_norm_skip_step_hard" threshold blocked ALL optimizer steps
                # at startup (pre-clip norms >400 are normal with random init + large rewards).
                if not np.isfinite(grad_norm):
                    hard_spike_batches_total += 1
                    hard_spike_streak += 1
                    self._set_actor_lr_factor(self.actor_lr_factor * self.actor_lr_decay_on_instability)
                    print(
                        f"\n⚠️ Skip optimizer step: non-finite grad norm pre-clip={grad_norm:.2f} "
                        f"(post={grad_norm_post:.2f})"
                    )
                    continue

                if grad_norm > float(self.grad_norm_skip_step_hard):
                    # Large but finite: log and continue training (clipping handled it).
                    if self.show_pre_train_debug_msg:
                        print(
                            f"\n⚠️ Large grad norm pre-clip={grad_norm:.2f} "
                            f"(post={grad_norm_post:.2f}, thr={self.grad_norm_skip_step_hard:.1f}) — clipped, continuing"
                        )

                if grad_norm > self.grad_norm_hard:
                    hard_spike_batches_total += 1
                    hard_spike_streak += 1
                    self._set_actor_lr_factor(self.actor_lr_factor * self.actor_lr_decay_on_instability)
                elif grad_norm > self.grad_norm_soft:
                    hard_spike_streak = max(hard_spike_streak, 1)

                def _module_grad_norm(module: nn.Module) -> float:
                    sq = 0.0
                    for p in module.parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        if torch.isnan(g).any() or torch.isinf(g).any():
                            continue
                        sq += float(torch.sum(g * g).item())
                    return float(math.sqrt(max(sq, 0.0)))

                tree_grad_actor = 0.0
                tree_grad_critic = 0.0
                tree_payload_grad_actor = 0.0
                tree_payload_grad_critic = 0.0
                if hasattr(self.encoder_actor, 'tree_encoder'):
                    tree_grad_actor = _module_grad_norm(self.encoder_actor.tree_encoder)
                if hasattr(self.encoder_critic, 'tree_encoder'):
                    tree_grad_critic = _module_grad_norm(self.encoder_critic.tree_encoder)
                if hasattr(self.encoder_actor, 'tree_payload_encoder'):
                    tree_payload_grad_actor = _module_grad_norm(self.encoder_actor.tree_payload_encoder)
                if hasattr(self.encoder_critic, 'tree_payload_encoder'):
                    tree_payload_grad_critic = _module_grad_norm(self.encoder_critic.tree_payload_encoder)
                
                # Check for NaN in gradients after clipping
                has_nan_grad = False
                for param in list(self.encoder_actor.parameters()) + list(self.actor_critic_model.actor.parameters()):
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"\n⚠️ WARNING: NaN detected in gradients at epoch {k_loop}, batch {batch_idx}")
                    print("   Skipping optimizer step to prevent crash...")
                    continue
                
                # Update optimizers - mode-dependent
                t0 = _tic() if profile_enabled else 0.0
                if self.optimizer_mode == 'single':
                    # Single shared optimizer: one step for all parameters
                    self.optimizer.step()
                else:  # multiple
                    # 4 separate optimizers: update all 4
                    if policy_weight_eff > 0.0:
                        self.optimizer_encoder_actor.step()
                        self.optimizer_actor_head.step()
                    self.optimizer_encoder_critic.step()
                    self.optimizer_critic_head.step()
                if profile_enabled:
                    _add_t('optimizer_step', _tic() - t0)
                
                self.loss = loss.detach().cpu().numpy()

                # Accumulate for 100-episode master report
                self._stat_buf['v_loss'].append(value_loss_component.item())
                self._stat_buf['p_loss'].append(policy_loss_component.item())
                self._stat_buf['e_loss'].append(entropy_loss_component.item())
                self._stat_buf['aux_dl'].append(aux_deadlock_loss_component.item())
                self._stat_buf['kl'].append(approx_kl)
                self._stat_buf['ratio'].append(ratio_mean)
                self._stat_buf['entropy'].append(entropy_mean)
                self._stat_buf['adv_mean'].append(raw_adv_mean)
                self._stat_buf['adv_std'].append(raw_adv_std)
                self._stat_buf['grad_norm'].append(grad_norm)
                self._stat_buf['grad_norm_post'].append(float(grad_norm_post))
                self._stat_buf['tree_grad_actor'].append(tree_grad_actor)
                self._stat_buf['tree_grad_critic'].append(tree_grad_critic)
                self._stat_buf['tree_payload_grad_actor'].append(tree_payload_grad_actor)
                self._stat_buf['tree_payload_grad_critic'].append(tree_payload_grad_critic)
                self._stat_buf['comm_loss'].append(comm_loss_component.item())
                self._stat_buf['action_div_loss'].append(action_diversity_loss_component.item())
                self._stat_buf['action_div_gate_ratio'].append(action_diversity_gate_ratio)
                self._stat_buf['total_loss'].append(loss.item())
                self._stat_buf['action_hist'].append(batch_action_hist.detach().cpu().numpy())
                self._stat_buf['action_hist_decision'].append(batch_action_hist_decision.detach().cpu().numpy())
                
                # Calculate communication intent metrics (needed for batch_metrics)
                actor_int = self.encoder_actor.last_comm_intent_mean
                critic_int = self.encoder_critic.last_comm_intent_mean
                valid_cnt = self.encoder_actor.last_comm_valid_count + self.encoder_critic.last_comm_valid_count
                if valid_cnt > 0:
                    wait_mean = 0.5 * (actor_int[0] + critic_int[0])
                    go_mean = 0.5 * (actor_int[1] + critic_int[1])
                    yield_mean = 0.5 * (actor_int[2] + critic_int[2])
                else:
                    wait_mean, go_mean, yield_mean = 0.0, 0.0, 0.0
                
                # Log per-batch metrics to TensorBoard
                batch_metrics = {
                    'loss': loss.item(),
                    'p_loss': policy_loss_component.item(),
                    'v_loss': value_loss_component.item(),
                    'e_loss': entropy_loss_component.item(),
                    'aux_dl': aux_deadlock_loss_component.item(),
                    'action_div': action_diversity_loss_component.item(),
                    'action_div_gate_ratio': action_diversity_gate_ratio,
                    'comm_loss': comm_loss_component.item(),
                    'kl': approx_kl,
                    'ratio': ratio_mean,
                    'entropy': entropy_mean,
                    'adv_mean': raw_adv_mean,
                    'adv_std': raw_adv_std,
                    'grad_norm': grad_norm,
                    'grad_norm_post': float(grad_norm_post),
                    'clip': clip_eps_eff,
                    'lr_factor': self.actor_lr_factor,
                    'comm_gate': (self.encoder_actor.last_comm_gate_mean + self.encoder_critic.last_comm_gate_mean) / 2.0,
                    'intent_wait': wait_mean if valid_cnt > 0 else 0.0,
                    'intent_go': go_mean if valid_cnt > 0 else 0.0,
                    'intent_yellow': yield_mean if valid_cnt > 0 else 0.0,
                    'comm_dropout': self.encoder_actor.comm_dropout.p,
                    'comm_weight': comm_weight_eff,
                    'policy_weight': policy_weight_eff,
                    'action_hist': batch_action_hist.numpy() if hasattr(batch_action_hist, 'numpy') else np.array(batch_action_hist),
                    'action_hist_decision': batch_action_hist_decision.numpy() if hasattr(batch_action_hist_decision, 'numpy') else np.array(batch_action_hist_decision),
                }
                self._log_batch_metrics(batch_metrics)
                self._stat_buf['ret_min'].append(batch_gae_returns.min().item())
                self._stat_buf['ret_max'].append(batch_gae_returns.max().item())
                _n_batches_update += 1

                # 📊 Log metrics for this iteration
                adv_mean = raw_adv_mean  # ⚡ RAW advantage mean (BEFORE normalization)
                adv_std = raw_adv_std    # ⚡ RAW advantage std (BEFORE normalization)
                
                if self.show_progress_bar:
                    progress = current_iteration / total_iterations
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r  [{bar}] Epoch {k_loop+1}/{k_epochs_eff}, Batch {batch_idx+1}/{num_batches} ({progress*100:3.1f}%)", end='')
                    print("\t", end='')
                    print(f"| Loss: {loss.item():.4f}", end='')
                    print(f"| P_Loss: {policy_loss_component.item():.4f}", end='')
                    print(f"| V_Loss: {value_loss_component.item():.4f}", end='')
                    print(f"| E_Loss: {entropy_loss_component.item():.4f}", end='')
                    print(f"| Adiv: {action_diversity_loss_component.item():.4f}", end='')
                    print(f"| AdivMask: {action_diversity_gate_ratio:.2f}", end='')
                    print(f"| AuxDL: {aux_deadlock_loss_component.item():.4f}", end='')
                    print(f"| C_Loss: {comm_loss_component.item():.4f}", end='')
                    print(f"| Adv: {adv_mean:.2f}±{adv_std:.2f}", end='')
                    print(f"| Ratio: {ratio_mean:.4f}", end='')
                    print(f"| KL: {approx_kl:.4f}", end='')
                    print(f"| Gate: {(self.encoder_actor.last_comm_gate_mean + self.encoder_critic.last_comm_gate_mean)/2.0:.3f}", end='')
                    print(f"| Intent W/G/Y: {wait_mean:.2f}/{go_mean:.2f}/{yield_mean:.2f}", end='')
                    print(f"| CommW: {comm_weight_eff:.5f}", end='')
                    print(f"| Drop: {self.encoder_actor.comm_dropout.p:.2f}", end='')
                    action_hist_str = '/'.join(str(int(x)) for x in batch_action_hist.tolist())
                    print(f"| Clip: {clip_eps_eff:.3f}", end='')
                    print(f"| Pw: {policy_weight_eff:.2f}", end='')
                    print(f"| LRf: {self.actor_lr_factor:.2f}", end='')
                    print(f"| H: {entropy_mean:.3f}", end='')
                    print(f"| Act {'/'.join(action_labels)}: {action_hist_str}", end='')
                    print(f"| Grad_Norm: {grad_norm:.4f}", end='')
                    print("", end='', flush=True)

                if hard_spike_streak >= self.hard_spike_streak_limit:
                    break_current_epoch = True
                    hard_spike_streak = 0
                    if self.show_pre_train_debug_msg:
                        print(f"\n⚠️ Early epoch stop due to repeated hard PPO spikes (ep={self.episode_count}, epoch={k_loop+1})")
                    break

            if break_current_epoch:
                continue

        if hard_spike_batches_total >= self.max_hard_batches_before_lr_decay:
            self._set_actor_lr_factor(self.actor_lr_factor * self.actor_lr_decay_on_instability)
            if self.show_pre_train_debug_msg:
                print(f"⚠️ Actor LR decayed to factor={self.actor_lr_factor:.3f} after {hard_spike_batches_total} hard spikes")
        elif hard_spike_batches_total == 0 and self.actor_lr_factor < self.actor_lr_max_factor:
            self._set_actor_lr_factor(self.actor_lr_factor * self.actor_lr_recover_rate)
            if self.show_pre_train_debug_msg:
                print(f"✅ Actor LR recovered to factor={self.actor_lr_factor:.3f}")

        if self.show_progress_bar:
            total_actions = int(action_hist_total.sum().item())
            if total_actions > 0:
                action_pct = (100.0 * action_hist_total.float() / float(total_actions)).tolist()
                summary = ', '.join(
                    f"{label}:{int(cnt)} ({pct:.1f}%)"
                    for label, cnt, pct in zip(action_labels, action_hist_total.tolist(), action_pct)
                )
                print(f"\nAction stats this PPO update -> {summary}")
            total_actions_decision = int(action_hist_decision_total.sum().item())
            if total_actions_decision > 0:
                action_pct_dec = (100.0 * action_hist_decision_total.float() / float(total_actions_decision)).tolist()
                summary_dec = ', '.join(
                    f"{label}:{int(cnt)} ({pct:.1f}%)"
                    for label, cnt, pct in zip(action_labels, action_hist_decision_total.tolist(), action_pct_dec)
                )
                print(f"Action stats (decision-gated) -> {summary_dec}")
            print("\n")  # New line after progress bar
        
        if not hasattr(self, 'training_step'):
            self.training_step = 0
        self.training_step += 1

        # Always-visible training summary per PPO update
        if _n_batches_update > 0:
            def _safe_mean_last(lst, n):
                if n <= 0 or not lst:
                    return 0.0
                return float(np.mean(lst[-n:]))
            _vl = _safe_mean_last(self._stat_buf['v_loss'], _n_batches_update)
            _pl = _safe_mean_last(self._stat_buf['p_loss'], _n_batches_update)
            _kl = _safe_mean_last(self._stat_buf['kl'], _n_batches_update)
            _gn = _safe_mean_last(self._stat_buf['grad_norm'], _n_batches_update)
            _gn_post = _safe_mean_last(self._stat_buf['grad_norm_post'], _n_batches_update)
            _en = _safe_mean_last(self._stat_buf['entropy'], _n_batches_update)
            _rt = _safe_mean_last(self._stat_buf['ratio'], _n_batches_update)
            print(
                f"[Train] ep={self.episode_count} batches={_n_batches_update} "
                f"v_loss={_vl:.4f} p_loss={_pl:.4f} kl={_kl:.4f} "
                f"ratio={_rt:.3f} grad_pre={_gn:.3f} grad_post={_gn_post:.3f} entropy={_en:.3f} "
                f"lr_factor={self.actor_lr_factor:.3f}"
            )
        else:
            print(f"[Train] ep={self.episode_count} — no batches executed (all skipped by guards)")

        if profile_enabled:
            timing['total'] = _tic() - update_t0
            for k in self._time_profile_buf:
                self._time_profile_buf[k].append(float(timing.get(k, 0.0)))

        # Clear data
        del all_state_tuples
        del all_actions
        del all_gae_advantages
        del all_gae_returns
        del all_aux_deadlock
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.current_episode_memory.reset()

    def get_training_summary(self):
        """Network architecture summary"""
        print("\n" + "="*80)
        print("📊 TRAINING SUMMARY - Temporal Transformer Architecture")
        print("="*80)

        encoder_actor_params = sum(p.numel() for p in self.encoder_actor.parameters())
        encoder_critic_params = sum(p.numel() for p in self.encoder_critic.parameters())
        actor_params = sum(p.numel() for p in self.actor_critic_model.actor.parameters())
        critic_params = sum(p.numel() for p in self.actor_critic_model.critic.parameters())
        total_params = encoder_actor_params + encoder_critic_params + actor_params + critic_params

        print("\n🧠 TEMPORAL_ENCODER_ACTOR (2-Level Attention):")
        print(f"   Total Parameters: {encoder_actor_params:,}")

        print("\n🧠 TEMPORAL_ENCODER_CRITIC (2-Level Attention):")
        print(f"   Total Parameters: {encoder_critic_params:,}")

        print("\n🎭 ACTOR Network:")
        print(f"   Total Parameters: {actor_params:,}")

        print("\n💎 CRITIC Network:")
        print(f"   Total Parameters: {critic_params:,}")

        print("\n📈 TOTAL:")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Temporal Window: {self.temporal_window} timesteps")

        print("\n" + "="*80 + "\n")

    # Feature names/descriptions for current DecisionPointObservation layout
    # (15D base vector; tree context is provided out-of-band via payload).
    _OBS_FEATURE_NAMES = [
        "path_left", "path_forward", "path_right",
        "delta_left", "delta_forward", "delta_right",
        "st_3", "st_4", "st_6",
        "priority_rank",
        "is_pre_merge", "is_switch",
        "sp_left", "sp_forward", "sp_right",
    ]

    _OBS_FEATURE_DESC = [
        "Relative transition exists: left",
        "Relative transition exists: forward",
        "Relative transition exists: right",
        "Distance delta to left successor (exp-squashed; -1 if no transition)",
        "Distance delta to forward successor (exp-squashed; -1 if no transition)",
        "Distance delta to right successor (exp-squashed; -1 if no transition)",
        "TrainState: READY_TO_DEPART",
        "TrainState: MALFUNCTION",
        "State flag: not started (position is None)",
        "Normalized priority rank",
        "One step before merge-conflict point",
        "Current cell is a switch",
        "Shortest-path hint: left",
        "Shortest-path hint: forward",
        "Shortest-path hint: right",
    ]

    def _print_obs_statistics(self):
        """Master-Diagnostik-Report alle 100 Episoden — LLM-paste-ready."""
        W = "=" * 80
        C_RESET = "\033[0m"
        C_BOLD = "\033[1m"
        C_RED = "\033[91m"
        C_GREEN = "\033[92m"
        C_YELLOW = "\033[93m"
        C_BLUE = "\033[94m"
        C_CYAN = "\033[96m"
        n_names = len(self._OBS_FEATURE_NAMES)
        n_descs = len(self._OBS_FEATURE_DESC)

        def color(text, code):
            return f"{code}{text}{C_RESET}"

        def status_text(status):
            if status == "OK":
                return color(status, C_GREEN)
            if status == "WARN":
                return color(status, C_YELLOW)
            if status == "ALERT":
                return color(status, C_RED)
            return status

        def severity(val, lo_ok, hi_ok, hi_alert=None):
            if lo_ok <= val <= hi_ok:
                return "OK"
            if hi_alert is not None and val > hi_alert:
                return "ALERT"
            return "WARN"

        def fname(i):
            return self._OBS_FEATURE_NAMES[i] if i < n_names else f"feat_{i}"

        def fdesc(i):
            if i < n_descs:
                return self._OBS_FEATURE_DESC[i]
            # sparse-neighbor features have no fixed desc
            return f"Nachbar-Feature #{i - n_names}" if i >= n_names else ""

        def row(label, value, unit="", status="", note=""):
            """Fixed-width table row: label | value | unit | status | note"""
            return (f"  | {label:<22s} | {value:>12s} | {unit:<6s} | "
                    f"{status_text(status):<13s} | {note}")

        def hdr(title):
            print(f"\n  +{'─'*78}+")
            print(f"  | {color(title, C_CYAN):<87s}|")
            print(f"  +{'─'*22}+{'─'*14}+{'─'*8}+{'─'*6}+{'─'*26}+")
            print(f"  | {'Metric':<22s} | {'Value':>12s} | {'Unit':<6s} | "
                  f"{'St':<4s} | {'Interpretation':<24s} |")
            print(f"  +{'─'*22}+{'─'*14}+{'─'*8}+{'─'*6}+{'─'*26}+")

        def end_table():
            print(f"  +{'─'*22}+{'─'*14}+{'─'*8}+{'─'*6}+{'─'*26}+")

        print(f"\n{color(W, C_BLUE)}")
        print(f"  {color('MAPPO DIAGNOSTIC REPORT', C_BOLD + C_CYAN)}")
        print(f"  Episode : {self.episode_count}")
        print(f"  Interval: {self._obs_stat_interval} episodes")
        print(f"  Context : Flatland 5-agent rail scheduling, DecisionPoint obs {_BASE_OBS_DIM}D base + raw tree payload")
        print(W)

        # ── 1) Episoden-Kennzahlen ─────────────────────────────────────────────
        r_list = self._ep_stat_buf['reward']
        d_list = self._ep_stat_buf['done_frac']
        ep_reward_mean = 0.0
        ep_reward_std = 0.0
        ep_done_mean = 0.0
        ep_done_std = 0.0
        if r_list:
            r_arr = np.array(r_list, dtype=np.float32)
            d_arr = np.array(d_list, dtype=np.float32)
            ep_reward_mean = float(r_arr.mean())
            ep_reward_std = float(r_arr.std())
            ep_done_mean = float(d_arr.mean())
            ep_done_std = float(d_arr.std())

            hdr(f"SECTION 1 — EPISODE PERFORMANCE  (n={len(r_arr)} episodes)")
            def ep_row(lbl, val, unit, lo_ok, hi_ok, note_ok, note_warn):
                st = severity(val, lo_ok, hi_ok)
                note = note_ok if st == "OK" else note_warn
                print(row(lbl, f"{val:+.3f}", unit, st, note) + " |")
            ep_row("Reward mean",     float(r_arr.mean()),  "scaled", -200,  200,
                   "normal range", "check reward shaping")
            ep_row("Reward std",      float(r_arr.std()),   "scaled",    0,  150,
                   "acceptable",   "high variance → unstable")
            ep_row("Reward min",      float(r_arr.min()),   "scaled", -500,    0,
                   "ok",           "extreme penalty episodes")
            ep_row("Reward max",      float(r_arr.max()),   "scaled",    0,  500,
                   "ok",           "check if reached")
            ep_row("Done-Rate mean",  float(d_arr.mean()),  "frac",   0.3,  1.0,
                   "learning",     "too low → deadlocks dominate")
            ep_row("Done-Rate std",   float(d_arr.std()),   "frac",   0.0,  0.15,
                   "stable",       "high variance → unstable policy")
            ep_row("Done-Rate min",   float(d_arr.min()),   "frac",   0.0,  1.0,
                   "ok",           "zero-done episodes present")
            ep_row("Done-Rate max",   float(d_arr.max()),   "frac",   0.0,  1.0,
                   "ok",           "best episodes ok")
            end_table()

            issues = []
            if d_arr.mean() < 0.3:
                issues.append(f"Done-Rate {d_arr.mean():.3f} < 0.30: agents stuck in deadlocks")
            if d_arr.std() > 0.15:
                issues.append(f"Done-Rate std {d_arr.std():.3f} > 0.15: policy oscillates, "
                               "no stable equilibrium reached yet")
            if r_arr.std() > 150:
                issues.append(f"Reward std {r_arr.std():.1f} very high: "
                               "environment outcomes highly stochastic or policy random")
            if issues:
                print(f"\n  {color('EPISODE ISSUES:', C_YELLOW)}")
                for iss in issues:
                    print(f"    {color('WARN', C_YELLOW)}  {iss}")
            else:
                print(f"\n    {color('OK', C_GREEN)}  No episode-level issues.")

        self._ep_stat_buf['reward'].clear()
        self._ep_stat_buf['done_frac'].clear()

        # ── 2) Training-Kennzahlen ─────────────────────────────────────────────
        sb = self._stat_buf
        n_b = len(sb['v_loss'])
        issues = []
        # Safe defaults for summary sections when no PPO batch ran in interval.
        vl_m = 0.0
        pl_m = 0.0
        el_m = 0.0
        kl_m = 0.0
        rt_m = 1.0
        en_m = float(self.entropy_floor)
        am_m = 0.0
        as_m = 0.0
        gn_m = 0.0
        gn_post_m = 0.0
        adg_m = 0.0
        ret_min = 0.0
        ret_max = 0.0
        aux_m = 0.0
        forward_share_all = 0.0
        forward_share_decision = 0.0
        if n_b > 0:
            def _ms(key):
                a = np.array(sb[key], dtype=np.float32)
                return float(a.mean()), float(a.std())

            vl_m, vl_s = _ms('v_loss')
            pl_m, pl_s = _ms('p_loss')
            el_m, el_s = _ms('e_loss')
            kl_m, kl_s = _ms('kl')
            rt_m, rt_s = _ms('ratio')
            en_m, en_s = _ms('entropy')
            am_m, _    = _ms('adv_mean')
            as_m, _    = _ms('adv_std')
            gn_m, gn_s = _ms('grad_norm')
            gn_post_m, gn_post_s = _ms('grad_norm_post')
            adg_m, adg_s = _ms('action_div_gate_ratio')
            ret_min    = float(np.array(sb['ret_min']).min())
            ret_max    = float(np.array(sb['ret_max']).max())
            aux_m      = float(np.array(sb['aux_dl']).mean())
            ah = np.array(sb['action_hist'], dtype=np.float32)
            if ah.ndim == 2 and ah.shape[1] > 2:
                ah_sum = ah.sum(axis=0)
                ah_total = float(max(ah_sum.sum(), 1.0))
                forward_share_all = float(ah_sum[2] / ah_total)
            ah_dec = np.array(sb.get('action_hist_decision', []), dtype=np.float32)
            if ah_dec.ndim == 2 and ah_dec.shape[1] > 2:
                ah_dec_sum = ah_dec.sum(axis=0)
                ah_dec_total = float(max(ah_dec_sum.sum(), 1.0))
                forward_share_decision = float(ah_dec_sum[2] / ah_dec_total)

            hdr(f"SECTION 2 — PPO TRAINING METRICS  ({n_b} batches, "
                f"lr_factor={self.actor_lr_factor:.3f})")

            def tr_row(lbl, val, pm, unit, lo_ok, hi_ok, note_ok, note_warn):
                hi_alert = None
                if lbl == "GradNorm (pre-clip)":
                    hi_alert = self.grad_norm_hard
                elif lbl == "GradNorm (post-clip)":
                    hi_alert = self.max_grad_norm_single * 1.5
                elif lbl == "Entropy":
                    hi_alert = None
                elif lbl == "V_Loss (critic)":
                    hi_alert = 2.0
                st = severity(val, lo_ok, hi_ok, hi_alert=hi_alert)
                note = note_ok if st == "OK" else note_warn
                vs   = f"{val:.4f} ±{pm:.4f}"
                print(row(lbl, vs, unit, st, note) + " |")

            tr_row("V_Loss (critic)", vl_m, vl_s, "", 0.0, 0.30,
                   "critic fits returns",
                   "critic not converging")
            tr_row("P_Loss (policy)", pl_m, pl_s, "", -0.5, 0.5,
                   "policy improving",
                   "gradient signal weak")
            tr_row("Entropy", en_m, en_s, "nat", self.entropy_floor, 2.0,
                   "exploration ok",
                   f"below floor={self.entropy_floor:.2f}, collapse risk")
            tr_row("KL divergence", kl_m, kl_s, "", 0.0, self.ppo_target_kl * 1.5,
                   "trust region ok",
                   f"exceeds target={self.ppo_target_kl:.3f}")
            tr_row("PPO ratio mean", rt_m, rt_s, "", 0.90, 1.15,
                   "ratios stable",
                   "policy update too large/small")
            tr_row("Advantage mean", am_m, 0.0, "", -2.0, 2.0,
                   "centered ok",
                   "advantages heavily biased")
            tr_row("Advantage std", as_m, 0.0, "", 0.5, 5.0,
                   "signal present",
                   "signal too weak or exploding")
            tr_row("GradNorm (pre-clip)", gn_m, gn_s, "", 0.0, self.grad_norm_skip_step_hard,
                   "pre-clip gradients plausible",
                   "large raw gradients before clipping")
            tr_row("GradNorm (post-clip)", gn_post_m, gn_post_s, "", 0.0, self.max_grad_norm_single * 1.10,
                   "effective gradients stable",
                   "effective gradients unstable after clipping")
            tr_row("AuxDL loss", aux_m, 0.0, "", 0.0, 0.8,
                   "deadlock head ok",
                   "deadlock head not converging")
            tr_row("Adiv gate ratio", adg_m, adg_s, "frac", 0.05, 0.70,
                   "diversity shaped at decisions",
                   "too sparse/broad gating for diversity")
            tr_row("Forward share (all)", forward_share_all, 0.0, "frac", 0.0, 0.80,
                   "global action mix plausible for sparse decision env",
                   "global forward share unusually high")
            tr_row("Forward share (decision)", forward_share_decision, 0.0, "frac", 0.0, 0.65,
                   "decision-point action balance plausible",
                   "forward bias too high at decision contexts")
            print(row("Returns range", f"{ret_min:+.2f}…{ret_max:+.2f}", "", "", "") + " |")
            end_table()

            # Diagnose
            if vl_m > 0.30:
                issues.append(
                    f"V_Loss={vl_m:.4f} > 0.30: Critic slow to converge (value clamp removed, learning full range). "
                    f"Returns span [{ret_min:+.2f},{ret_max:+.2f}]. "
                    f"Expected to improve over 200+ episodes as critic adapts to broader range.")
            if abs(pl_m) < 0.005:
                issues.append(
                    f"P_Loss={pl_m:+.5f} ≈ 0: Policy gradient is nearly zero. "
                    f"Adv_mean={am_m:+.3f}, Adv_std={as_m:.3f}, Ratio={rt_m:.4f}. "
                    "Likely cause: advantages too small (critic overfit or reward variance "
                    "low) OR ratio stuck near 1 (policy not changing).")
            if en_m < self.entropy_floor:
                issues.append(
                    f"Entropy={en_m:.4f} below floor={self.entropy_floor:.2f}: "
                    "Policy is collapsing to a deterministic mode prematurely. "
                    "Increase weight_entropy or entropy_recovery_scale.")
            if kl_m > self.ppo_target_kl * 2.0:
                issues.append(
                    f"KL={kl_m:.4f} >> target={self.ppo_target_kl:.3f}: "
                    "Trust region violated. Reduce learning rate or k_epochs.")
            if abs(rt_m - 1.0) > 0.20:
                issues.append(
                    f"Ratio={rt_m:.4f} far from 1.0: "
                    "Either policy changes too fast (ratio>1.2) or old/new policy "
                    "diverge already at batch start (ratio<0.8). Check clip_eps.")
            if as_m < 0.3:
                issues.append(
                    f"Advantage std={as_m:.3f} very low: almost no gradient signal. "
                    "Critic may be over-fitting, or all returns are nearly identical "
                    "(deadlock-dominated constant negative rewards).")
            if gn_post_m > self.max_grad_norm_single * 1.20:
                issues.append(
                    f"GradNorm post-clip={gn_post_m:.3f} (> {self.max_grad_norm_single*1.20:.3f}): effective gradient instability. "
                    "Lower LR and/or reduce reward scale.")
            if adg_m < 0.01:
                issues.append(
                    f"Adiv gate ratio={adg_m:.3f}: diversity shaping is almost never active. "
                    "Check if decision-point/deadlock signals are present in observations.")
            if forward_share_decision > 0.70:
                issues.append(
                    f"Decision forward share={forward_share_decision:.3f} > 0.70: "
                    "action choice remains too forward-biased at true decision contexts. "
                    "Increase diversity pressure or reduce route-prior bias.")

            if issues:
                print(f"\n  {color(f'TRAINING ISSUES ({len(issues)}):', C_YELLOW)}")
                for iss in issues:
                    print(f"    {color('WARN', C_YELLOW)}  {iss}")
            else:
                print(f"\n    {color('OK', C_GREEN)}  No training-level issues.")

        # Log episode-level aggregated metrics to TensorBoard
        if n_b > 0:
            episode_metrics = {
                'v_loss_mean': vl_m,
                'p_loss_mean': pl_m,
                'e_loss_mean': el_m,
                'aux_dl_mean': aux_m,
                'kl_mean': kl_m,
                'entropy_mean': en_m,
                'ratio_mean': rt_m,
                'grad_norm_mean': gn_m,
                'grad_norm_post_mean': gn_post_m,
                'action_div_gate_ratio_mean': adg_m,
                'forward_share_all': forward_share_all,
                'forward_share_decision': forward_share_decision,
                'reward_mean': float(r_arr.mean()) if len(r_arr) > 0 else 0.0,
                'done_frac': float(d_arr.mean()) if len(d_arr) > 0 else 0.0,
            }
            self._log_episode_metrics(episode_metrics)

        # ── 3) Feature Importance ──────────────────────────────────────────────
        print(f"\n{color(W, C_BLUE)}")
        W_mat = self.encoder_actor.obs_encoder[0].weight.detach().cpu().numpy()
        sens = np.abs(W_mat).mean(axis=0)
        obs_dim = sens.shape[0]
        max_s = sens.max() if sens.max() > 0 else 1.0
        order = np.argsort(sens)[::-1]
        sens_norm = sens / max_s          # relative importance in [0,1]

        print(f"  SECTION 3 — FEATURE IMPORTANCE  "
              f"(mean |weight| first encoder layer, {obs_dim}D input)")
        print("  Interpretation guide:")
        print("    rel_imp = mean|W_col| / max(mean|W_col|)")
        print("    >0.50 = highly used  |  0.10-0.50 = moderate  |  <0.05 = nearly ignored  |  <0.01 = dead")
        print()
        print(f"  +{'─'*4}+{'─'*5}+{'─'*22}+{'─'*9}+{'─'*9}+{'─'*26}+{'─'*26}+{'─'*44}+")
        print(f"  | {'Rk':>2} | {'[i]':>3} | {'Name':<20s} | {'abs_sens':>7} "
              f"| {'rel_imp':>7} | {'Bar (24-wide)':<24s} | {'Interpretation':<24s} | {'Description':<42s} |")
        print(f"  +{'─'*4}+{'─'*5}+{'─'*22}+{'─'*9}+{'─'*9}+{'─'*26}+{'─'*26}+{'─'*44}+")

        def imp_note(rel):
            if rel > 0.50:
                return "highly used by network"
            if rel > 0.20:
                return "moderately used"
            if rel > 0.05:
                return "low usage"
            if rel > 0.01:
                return "nearly ignored"
            return "DEAD — no gradient signal"

        for rank in range(obs_dim):
            i = order[rank]
            s = sens[i]
            r = sens_norm[i]
            bar = ('█' * int(r * 24)).ljust(24)
            note = imp_note(r)
            desc = fdesc(i)[:42]
            print(f"  | {rank+1:2d} | [{i:2d}] | {fname(i):<20s} | "
                  f"{s:7.5f} | {r:7.4f} | {bar} | {note:<24s} | {desc:<42s} |")

        print(f"  +{'─'*4}+{'─'*5}+{'─'*22}+{'─'*9}+{'─'*9}+{'─'*26}+{'─'*26}+{'─'*44}+")

        dead_thresh = max_s * 0.01
        dead_feats = [i for i in range(obs_dim) if sens[i] < dead_thresh]
        top3 = [fname(order[k]) for k in range(min(3, obs_dim))]
        print(f"\n  Top-3 most used : {', '.join(top3)}")
        if dead_feats:
            print(f"  Dead features   : {len(dead_feats)} ignored "
                  f"({', '.join(fname(i) for i in dead_feats[:8])}"
                  f"{'…' if len(dead_feats) > 8 else ''})")
            print(f"  {color('WARN', C_YELLOW)}  Dead features waste network capacity and may indicate "
                  f"structural issues in the observation.")
        else:
            print(f"  {color('OK', C_GREEN)}  All features contribute to network (>=1% sensitivity).")

        # ── 3b) Tree Encoder Diagnostics (every _obs_stat_interval episodes) ──
        print(f"\n{color(W, C_BLUE)}")
        print("  SECTION 3B — TREE ENCODER DIAGNOSTICS")

        def _param_norm(module: nn.Module) -> float:
            sq = 0.0
            for p in module.parameters():
                d = p.detach()
                if torch.isnan(d).any() or torch.isinf(d).any():
                    continue
                sq += float(torch.sum(d * d).item())
            return float(math.sqrt(max(sq, 0.0)))

        def _has_nan_inf(module: nn.Module) -> int:
            for p in module.parameters():
                d = p.detach()
                if torch.isnan(d).any() or torch.isinf(d).any():
                    return 1
            return 0

        tree_issues = []

        actor_tree_norm = 0.0
        critic_tree_norm = 0.0
        actor_payload_norm = 0.0
        critic_payload_norm = 0.0
        actor_tree_bad = 0
        critic_tree_bad = 0
        actor_payload_bad = 0
        critic_payload_bad = 0
        has_actor_tree_encoder = hasattr(self.encoder_actor, 'tree_encoder')
        has_critic_tree_encoder = hasattr(self.encoder_critic, 'tree_encoder')

        if hasattr(self.encoder_actor, 'tree_encoder'):
            actor_tree_norm = _param_norm(self.encoder_actor.tree_encoder)
            actor_tree_bad = _has_nan_inf(self.encoder_actor.tree_encoder)
        if hasattr(self.encoder_critic, 'tree_encoder'):
            critic_tree_norm = _param_norm(self.encoder_critic.tree_encoder)
            critic_tree_bad = _has_nan_inf(self.encoder_critic.tree_encoder)
        if hasattr(self.encoder_actor, 'tree_payload_encoder'):
            actor_payload_norm = _param_norm(self.encoder_actor.tree_payload_encoder)
            actor_payload_bad = _has_nan_inf(self.encoder_actor.tree_payload_encoder)
        if hasattr(self.encoder_critic, 'tree_payload_encoder'):
            critic_payload_norm = _param_norm(self.encoder_critic.tree_payload_encoder)
            critic_payload_bad = _has_nan_inf(self.encoder_critic.tree_payload_encoder)

        tg_a = float(np.mean(sb['tree_grad_actor'])) if sb['tree_grad_actor'] else 0.0
        tg_c = float(np.mean(sb['tree_grad_critic'])) if sb['tree_grad_critic'] else 0.0
        tpg_a = float(np.mean(sb['tree_payload_grad_actor'])) if sb['tree_payload_grad_actor'] else 0.0
        tpg_c = float(np.mean(sb['tree_payload_grad_critic'])) if sb['tree_payload_grad_critic'] else 0.0

        tree_rows = [
            (
                "Actor tree param-norm",
                actor_tree_norm,
                "",
                "OK" if (not has_actor_tree_encoder or actor_tree_bad == 0) else "ALERT",
                "payload-only encoder (no tree module)" if not has_actor_tree_encoder else ("healthy" if actor_tree_bad == 0 else "NaN/Inf in params"),
            ),
            (
                "Critic tree param-norm",
                critic_tree_norm,
                "",
                "OK" if (not has_critic_tree_encoder or critic_tree_bad == 0) else "ALERT",
                "payload-only encoder (no tree module)" if not has_critic_tree_encoder else ("healthy" if critic_tree_bad == 0 else "NaN/Inf in params"),
            ),
            ("Actor payload param-norm", actor_payload_norm, "", "OK" if actor_payload_bad == 0 else "ALERT",
             "healthy" if actor_payload_bad == 0 else "NaN/Inf in params"),
            ("Critic payload param-norm", critic_payload_norm, "", "OK" if critic_payload_bad == 0 else "ALERT",
             "healthy" if critic_payload_bad == 0 else "NaN/Inf in params"),
            (
                "Actor tree grad-norm",
                tg_a,
                "",
                "OK" if (not has_actor_tree_encoder or tg_a > 1e-8) else "WARN",
                "payload-only encoder (no tree module)" if not has_actor_tree_encoder else ("learning signal" if tg_a > 1e-8 else "near-zero updates"),
            ),
            (
                "Critic tree grad-norm",
                tg_c,
                "",
                "OK" if (not has_critic_tree_encoder or tg_c > 1e-8) else "WARN",
                "payload-only encoder (no tree module)" if not has_critic_tree_encoder else ("learning signal" if tg_c > 1e-8 else "near-zero updates"),
            ),
            ("Actor payload grad-norm", tpg_a, "", "OK" if tpg_a > 1e-8 else "WARN",
             "learning signal" if tpg_a > 1e-8 else "near-zero updates"),
            ("Critic payload grad-norm", tpg_c, "", "OK" if tpg_c > 1e-8 else "WARN",
             "learning signal" if tpg_c > 1e-8 else "near-zero updates"),
        ]

        print(f"  +{'─'*28}+{'─'*14}+{'─'*8}+{'─'*6}+{'─'*26}+")
        print(f"  | {'Metric':<26s} | {'Value':>12s} | {'Unit':<6s} | {'St':<4s} | {'Interpretation':<24s} |")
        print(f"  +{'─'*28}+{'─'*14}+{'─'*8}+{'─'*6}+{'─'*26}+")
        for lbl, val, unit, st, note in tree_rows:
            print(row(lbl, f"{val:.5f}", unit, st, note) + " |")
            if st == "ALERT":
                tree_issues.append(f"{lbl}: NaN/Inf in parameters")
            elif "grad-norm" in lbl and st == "WARN":
                tree_issues.append(f"{lbl}: near-zero gradient flow")
        print(f"  +{'─'*28}+{'─'*14}+{'─'*8}+{'─'*6}+{'─'*26}+")

        def _print_encoded_feature_stats(title: str, names: List[str], means: np.ndarray, stds: np.ndarray):
            print(f"\n  {title}:")
            print(f"  +{'─'*4}+{'─'*34}+{'─'*12}+{'─'*12}+")
            print(f"  | {'#':>2s} | {'Feature':<32s} | {'mean':>10s} | {'std':>10s} |")
            print(f"  +{'─'*4}+{'─'*34}+{'─'*12}+{'─'*12}+")
            for i, n in enumerate(names):
                print(f"  | {i:2d} | {n:<32.32s} | {float(means[i]):10.4f} | {float(stds[i]):10.4f} |")
            print(f"  +{'─'*4}+{'─'*34}+{'─'*12}+{'─'*12}+")

        def _collect_usage_proxy(encoders: List[nn.Module], names: List[str], layer_path: str) -> Optional[np.ndarray]:
            vals = []
            for enc in encoders:
                mod = enc
                try:
                    for p in layer_path.split('.'):
                        if p.isdigit():
                            mod = mod[int(p)]
                        else:
                            mod = getattr(mod, p)
                    if isinstance(mod, nn.Linear):
                        w = mod.weight.detach().abs().mean(dim=0).cpu().numpy().astype(np.float64)
                        if w.shape[0] == len(names):
                            vals.append(w)
                except Exception:
                    continue
            if not vals:
                return None
            arr = np.mean(np.stack(vals, axis=0), axis=0)
            denom = float(np.sum(arr))
            if denom <= 1e-12:
                return np.zeros_like(arr)
            return arr / denom

        def _print_usage_top(title: str, names: List[str], rel: np.ndarray, top_k: int = 8):
            order = np.argsort(-rel)
            k = min(int(top_k), len(names))
            print(f"\n  {title}:")
            print(f"  +{'─'*4}+{'─'*34}+{'─'*14}+")
            print(f"  | {'#':>2s} | {'Feature':<32s} | {'share%':>12s} |")
            print(f"  +{'─'*4}+{'─'*34}+{'─'*14}+")
            for r in range(k):
                i = int(order[r])
                print(f"  | {r+1:2d} | {names[i]:<32.32s} | {100.0 * float(rel[i]):12.3f} |")
            print(f"  +{'─'*4}+{'─'*34}+{'─'*14}+")

        if self._tree_stat_buffer:
            tarr_nodes = np.array([x['nodes'] for x in self._tree_stat_buffer], dtype=np.float32)
            tarr_edges = np.array([x['edges'] for x in self._tree_stat_buffer], dtype=np.float32)
            tarr_inv   = np.array([x['invalid_edges'] for x in self._tree_stat_buffer], dtype=np.float32)
            tarr_unmap = np.array([x.get('unmapped_edges', 0) for x in self._tree_stat_buffer], dtype=np.float32)
            tarr_empty = np.array([x['empty_payload'] for x in self._tree_stat_buffer], dtype=np.float32)

            total_edges = float(np.maximum(tarr_edges.sum(), 1.0))
            invalid_ratio = float(tarr_inv.sum() / total_edges)
            unmapped_ratio = float(tarr_unmap.sum() / total_edges)
            empty_ratio = float(tarr_empty.mean())

            print(f"\n  Variable tree payload stats (N={len(self._tree_stat_buffer)} frames):")
            print(f"    nodes mean/std/min/max = {tarr_nodes.mean():.2f}/{tarr_nodes.std():.2f}/{int(tarr_nodes.min())}/{int(tarr_nodes.max())}")
            print(f"    edges mean/std/min/max = {tarr_edges.mean():.2f}/{tarr_edges.std():.2f}/{int(tarr_edges.min())}/{int(tarr_edges.max())}")
            print(f"    empty payload ratio     = {empty_ratio:.3f}")
            print(f"    invalid edge ratio      = {invalid_ratio:.4f}")
            print(f"    unmapped edge ratio     = {unmapped_ratio:.4f}")

            node_sum = np.zeros(TreePayloadEncoder.NODE_DIM, dtype=np.float64)
            node_sq_sum = np.zeros(TreePayloadEncoder.NODE_DIM, dtype=np.float64)
            edge_sum = np.zeros(TreePayloadEncoder.EDGE_DIM, dtype=np.float64)
            edge_sq_sum = np.zeros(TreePayloadEncoder.EDGE_DIM, dtype=np.float64)
            node_count = 0.0
            edge_count = 0.0
            for rec in self._tree_stat_buffer:
                ns = rec.get('node_feat_sum')
                nss = rec.get('node_feat_sq_sum')
                es = rec.get('edge_feat_sum')
                ess = rec.get('edge_feat_sq_sum')
                nc = rec.get('node_feat_count', 0)
                ec = rec.get('edge_feat_count', 0)
                if isinstance(ns, np.ndarray) and ns.shape[0] == TreePayloadEncoder.NODE_DIM:
                    node_sum += ns.astype(np.float64)
                if isinstance(nss, np.ndarray) and nss.shape[0] == TreePayloadEncoder.NODE_DIM:
                    node_sq_sum += nss.astype(np.float64)
                if isinstance(es, np.ndarray) and es.shape[0] == TreePayloadEncoder.EDGE_DIM:
                    edge_sum += es.astype(np.float64)
                if isinstance(ess, np.ndarray) and ess.shape[0] == TreePayloadEncoder.EDGE_DIM:
                    edge_sq_sum += ess.astype(np.float64)
                node_count += float(nc)
                edge_count += float(ec)

            if node_count > 0.5:
                node_mean = node_sum / node_count
                node_var = np.maximum(0.0, (node_sq_sum / node_count) - (node_mean * node_mean))
                node_std = np.sqrt(node_var)
                _print_encoded_feature_stats(
                    "Encoded TREE NODE feature stats",
                    TreePayloadEncoder.NODE_FEATURE_NAMES,
                    node_mean,
                    node_std,
                )
            else:
                print("\n  Encoded TREE NODE feature stats: no valid node features observed.")

            if edge_count > 0.5:
                edge_mean = edge_sum / edge_count
                edge_var = np.maximum(0.0, (edge_sq_sum / edge_count) - (edge_mean * edge_mean))
                edge_std = np.sqrt(edge_var)
                _print_encoded_feature_stats(
                    "Encoded TREE EDGE feature stats",
                    TreePayloadEncoder.EDGE_FEATURE_NAMES,
                    edge_mean,
                    edge_std,
                )
            else:
                print("\n  Encoded TREE EDGE feature stats: no valid edge features observed.")

            payload_encoders = []
            if hasattr(self.encoder_actor, 'tree_payload_encoder'):
                payload_encoders.append(self.encoder_actor.tree_payload_encoder)
            if hasattr(self.encoder_critic, 'tree_payload_encoder'):
                payload_encoders.append(self.encoder_critic.tree_payload_encoder)

            node_usage = _collect_usage_proxy(payload_encoders, TreePayloadEncoder.NODE_FEATURE_NAMES, "node_proj.0")
            if node_usage is not None:
                _print_usage_top("MAPPO tree usage proxy (node_proj input share)", TreePayloadEncoder.NODE_FEATURE_NAMES, node_usage)

            edge_usage = _collect_usage_proxy(payload_encoders, TreePayloadEncoder.EDGE_FEATURE_NAMES, "edge_gate.0")
            if edge_usage is not None:
                _print_usage_top("MAPPO tree usage proxy (edge_gate input share)", TreePayloadEncoder.EDGE_FEATURE_NAMES, edge_usage)

            if invalid_ratio > 0.02:
                tree_issues.append(f"Invalid edge ratio {invalid_ratio:.4f} > 0.02: malformed edge fields in payload")
            if unmapped_ratio > 0.25:
                tree_issues.append(f"Unmapped edge ratio {unmapped_ratio:.4f} > 0.25: many edges reference nodes outside retained local node set")
            if empty_ratio > 0.80:
                tree_issues.append(f"Empty payload ratio {empty_ratio:.3f} > 0.80: tree search may be missing coverage")

            self._tree_stat_buffer.clear()
        else:
            print("  No tree payload samples collected in this interval.")

        if tree_issues:
            print(f"\n  {color('TREE ENCODER ISSUES:', C_YELLOW)}")
            for iss in tree_issues:
                print(f"    {color('WARN', C_YELLOW)}  {iss}")
        else:
            print(f"\n    {color('OK', C_GREEN)}  Tree encoder diagnostics look healthy.")

        # ── 3c) Timing Profile (every _obs_stat_interval episodes) ──
        tbuf = self._time_profile_buf
        t_n = len(tbuf['total']) if 'total' in tbuf else 0
        if t_n > 0:
            t_total = np.array(tbuf['total'], dtype=np.float32)
            total_mean = float(np.mean(t_total)) if t_total.size > 0 else 0.0
            print(f"\n{color(W, C_BLUE)}")
            print(f"  SECTION 3C — TRAINING TIMING PROFILE  (n={t_n} updates)")
            print(f"  +{'─'*26}+{'─'*14}+{'─'*10}+{'─'*34}+")
            print(f"  | {'Block':<24s} | {'mean_s':>12s} | {'share%':>8s} | {'note':<32s} |")
            print(f"  +{'─'*26}+{'─'*14}+{'─'*10}+{'─'*34}+")
            for key, note in [
                ('pool_collect', 'episode pool assembly'),
                ('gae_prep', 'critic enc + gae preparation'),
                ('concat_sample', 'concat + sampling'),
                ('old_logprobs', 'precompute old logprobs'),
                ('encode_batch', 'encoder forward per batch'),
                ('forward_loss', 'actor/critic forward + losses'),
                ('backward_clip', 'backward + grad clipping'),
                ('optimizer_step', 'optimizer update'),
            ]:
                arr = np.array(tbuf.get(key, []), dtype=np.float32)
                mean_s = float(np.mean(arr)) if arr.size > 0 else 0.0
                share = (100.0 * mean_s / max(total_mean, 1e-9)) if total_mean > 0.0 else 0.0
                print(f"  | {key:<24s} | {mean_s:12.4f} | {share:8.2f} | {note:<32s} |")
            print(f"  +{'─'*26}+{'─'*14}+{'─'*10}+{'─'*34}+")
            print(f"  | {'total':<24s} | {total_mean:12.4f} | {100.0:8.2f} | {'per PPO update':<32s} |")
            print(f"  +{'─'*26}+{'─'*14}+{'─'*10}+{'─'*34}+")
            for k in tbuf:
                tbuf[k].clear()

        # ── 4) Obs-Sanity ──────────────────────────────────────────────────────
        obs_issues = []
        if self._obs_stat_buffer:
            data  = np.array(self._obs_stat_buffer, dtype=np.float32)
            self._obs_stat_buffer.clear()
            N, D  = data.shape
            mins  = data.min(axis=0)
            maxs  = data.max(axis=0)
            means = data.mean(axis=0)
            stds  = data.std(axis=0)
            RANGE_TOL = 0.05

            # Per-feature expected ranges for 24D DecisionPoint base vector.
            # Deltas [3..5] are intentionally in [-1, 1], others in [0, 1].
            exp_lo = np.zeros(D, dtype=np.float32)
            exp_hi = np.ones(D, dtype=np.float32)
            if D >= 6:
                exp_lo[3:6] = -1.0

            print(f"\n{color(W, C_BLUE)}")
            print(f"  SECTION 4 — OBSERVATION SANITY  (N={N} frames, D={D} features)")
            print()

            oor  = [
                (i, mins[i], maxs[i], exp_lo[i], exp_hi[i])
                for i in range(D)
                if mins[i] < (exp_lo[i] - RANGE_TOL) or maxs[i] > (exp_hi[i] + RANGE_TOL)
            ]
            dead = [(i, means[i]) for i in range(D) if stds[i] < 1e-4]

            valid_idx = [i for i in range(D) if stds[i] > 1e-6]
            dups = []
            if len(valid_idx) >= 2:
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr = np.corrcoef(data[:, valid_idx].T)
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                dups_raw = [(valid_idx[ii], valid_idx[jj], corr[ii, jj])
                            for ii in range(len(valid_idx))
                            for jj in range(ii + 1, len(valid_idx))
                            if abs(corr[ii, jj]) > 0.99]

                # Known intentional/derived pairs in the 24D contract:
                # st_6 = 1 - is_started, is_done mirrors st_4.
                expected_dup_pairs = {
                    tuple(sorted((10, 17))),
                    tuple(sorted((12, 16))),
                }
                dups = [
                    (a, b, c) for (a, b, c) in dups_raw
                    if tuple(sorted((a, b))) not in expected_dup_pairs
                ]

            obs_issues = []
            if oor:
                obs_issues.append(
                    f"{len(oor)} feature(s) outside [0,1]: "
                    + ", ".join(f"[{i}]{fname(i)}(min={lo:+.3f},max={hi:+.3f},exp=[{elo:+.1f},{ehi:+.1f}])"
                                for i, lo, hi, elo, ehi in oor[:5]))
            if dead:
                obs_issues.append(
                    f"{len(dead)} constant feature(s) (zero variance): "
                    + ", ".join(f"[{i}]{fname(i)}" for i, _ in dead[:8]))
            if dups:
                obs_issues.append(
                    f"{len(dups)} duplicate feature pair(s) (|corr|>0.99): "
                    + ", ".join(f"[{a}]{fname(a):<20s} <-> [{b}]{fname(b):<20s}  corr={c:+.4f}"
                                for a, b, c in dups[:4]))

            print(f"  +{'─'*30}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*6}+")
            print(f"  | {'Feature':<28s} | {'mean':>5} | {'std':>5} | "
                  f"{'min':>5} | {'max':>5} | {'oor':>5} | {'dead':>4} |")
            print(f"  +{'─'*30}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*6}+")
            for i in range(D):
                is_oor  = mins[i] < (exp_lo[i] - RANGE_TOL) or maxs[i] > (exp_hi[i] + RANGE_TOL)
                is_dead = stds[i] < 1e-4
                if not is_oor and not is_dead:
                    continue   # only print notable features
                oor_tag  = "YES" if is_oor  else ""
                dead_tag = "YES" if is_dead else ""
                print(f"  | [{i:2d}] {fname(i):<24s} | {means[i]:5.3f} | {stds[i]:5.3f} | "
                      f"{mins[i]:+5.3f} | {maxs[i]:5.3f} | {oor_tag:>5} | {dead_tag:>4} |")
            print(f"  +{'─'*30}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*6}+")

            if not oor and not dead:
                print(f"  {color('OK', C_GREEN)}  All features in [0,1], none constant.")
            if dups:
                print("\n  Duplicate pairs (|corr|>0.99):")
                for a, b, c in dups:
                    print(f"    [{a}]{fname(a):<20s} <-> [{b}]{fname(b):<20s}  corr={c:+.4f}")

            # Key feature means
            if D > 13:
                print("\n  Key feature means:")
                state_mass = float(np.sum(means[6:13]))
                print(f"    train_state_mass={state_mass:.3f}  priority_rank={means[13]:.3f}")

            if obs_issues:
                print(f"\n  {color('OBS ISSUES:', C_YELLOW)}")
                for iss in obs_issues:
                    print(f"    {color('WARN', C_YELLOW)}  {iss}")
            else:
                print(f"  {color('OK', C_GREEN)}  No observation sanity issues.")
        else:
            self._obs_stat_buffer.clear()

        # ── 5) Copilot summary block ───────────────────────────────────────────
        print(f"\n{color(W, C_BLUE)}")
        print(f"  {color('SECTION 5 — COPILOT INTERNAL TRIAGE', C_BOLD + C_CYAN)}")
        print("  Compact block for direct next-action decisions:")
        print(f"  {'─'*76}")

        all_issues = issues + obs_issues
        issue_text = (("\n".join(f"  - {x}" for x in all_issues))
                      if all_issues else "  - None detected.")
        overall = "OK"
        if ep_done_mean < 0.20 or en_m < self.entropy_floor or gn_m > 20.0:
            overall = "ALERT"
        elif all_issues:
            overall = "WARN"

        suggestions = []
        if ep_done_mean < 0.20:
            suggestions.append("1. DONE zu niedrig: `weight_entropy +0.02`, `decision_eps_floor +0.02`, `max_eps_random +0.02`")
        if en_m < self.entropy_floor:
            suggestions.append("2. Entropie-Kollaps: Idle/Stop weniger hart bestrafen und Exploration offen halten")
        if gn_m > 5.0:
            suggestions.append("3. Gradienten zu hoch: `learning_rate * 0.75` und `reward_scale -0.01`")
        if vl_m > 0.80:
            suggestions.append("4. Critic zu hoch: Reward-Skala klein halten und frischen Neustart statt Continue bevorzugen")
        if not suggestions:
            suggestions.append("1. Keine akute Anpassung noetig, weitertrainieren und DONE/Deadlocks beobachten")

        print(
            f"  PROFILE: MAPPO Flatland 5-agent | Episode={self.episode_count}\n"
            f"  OBS: DecisionPoint{_BASE_OBS_DIM} + raw-tree payload + temporal encoder + comm-attn\n"
            f"\n"
            f"  OVERALL: {status_text(overall)}\n"
            f"\n"
            f"  METRICS:\n"
            f"  done_mean={ep_done_mean:.3f} done_std={ep_done_std:.3f} "
            f"reward_mean={ep_reward_mean:+.1f} reward_std={ep_reward_std:.1f}\n"
            f"  v_loss={vl_m:.4f} p_loss={pl_m:+.4f} entropy={en_m:.4f} "
            f"kl={kl_m:.4f} ratio={rt_m:.4f} grad={gn_m:.4f}\n"
            f"  adv_mean={am_m:+.3f} adv_std={as_m:.3f} "
            f"returns=[{ret_min:+.2f},{ret_max:+.2f}] aux_dl={aux_m:.4f}\n"
            f"  forward_all={forward_share_all:.3f} forward_decision={forward_share_decision:.3f}\n"
            f"  actor_lr_factor={self.actor_lr_factor:.4f} clip_eps={self._effective_clip_eps():.3f}\n"
            f"\n"
            f"  DECISION_RULES:\n"
            f"  - if done_mean < 0.22 and v_loss > 0.80: increase critic pressure or reduce reward penalties\n"
            f"  - if entropy < {self.entropy_floor:.2f}: raise entropy weight / recovery scale\n"
            f"  - if decision-forward share > 0.65: increase action-diversity penalty\n"
            f"  - if grad > 5.0: lower actor lr or tighten clipping\n"
            f"\n"
            f"  ISSUES:\n{issue_text}\n"
            f"\n"
            f"  VORSCHLAG:\n  " + "\n  ".join(suggestions)
        )
        print(f"  {'─'*76}")
        print(f"{color(W, C_BLUE)}\n")

        # Clear training-batch diagnostics only after all report sections
        # consumed the buffered values (including tree diagnostics).
        for key in self._stat_buf:
            self._stat_buf[key].clear()

    def end_episode(self, train):
        if train:
            # Collect episode-level stats before buffer is reset
            ep_total_reward = 0.0
            ep_done_count   = 0
            ep_n_agents     = len(self.current_episode_memory.memory)
            ep_timeout_count = 0
            ep_len_list: List[int] = []
            ep_final_aux_list: List[float] = []
            ep_sp_match = 0
            ep_sp_total = 0
            for transitions in self.current_episode_memory.memory.values():
                if transitions:
                    ep_len_list.append(len(transitions))
                    ep_total_reward += sum(float(t[2]) for t in transitions)
                    last_t = transitions[-1]
                    if len(last_t) >= 6:
                        ep_final_aux_list.append(float(last_t[5]))

                    for t in transitions:
                        state, action, _, _, _, _ = t
                        best_action = self._extract_local_shortest_action_from_temporal_state(state)
                        if best_action is None:
                            continue
                        ep_sp_total += 1
                        if int(action) == int(best_action):
                            ep_sp_match += 1

                    # Count agents with done==True in their final transition
                    if transitions[-1][4]:   # done flag of last transition
                        ep_done_count += 1
                    else:
                        ep_timeout_count += 1
            # Keep logging aligned with rewards already shaped at transition time.
            self._ep_stat_buf['reward'].append(ep_total_reward)
            done_frac = ep_done_count / ep_n_agents if ep_n_agents > 0 else 0.0
            timeout_frac = ep_timeout_count / ep_n_agents if ep_n_agents > 0 else 0.0
            self._ep_stat_buf['done_frac'].append(done_frac)

            self._rollout_diag_buf['sp_match'].append(int(ep_sp_match))
            self._rollout_diag_buf['sp_total'].append(int(ep_sp_total))
            self._rollout_diag_buf['timeout_frac'].append(float(timeout_frac))
            self._rollout_diag_buf['ep_len'].append(float(np.mean(ep_len_list)) if ep_len_list else 0.0)
            self._rollout_diag_buf['final_aux'].append(float(np.mean(ep_final_aux_list)) if ep_final_aux_list else 0.0)
            self._rollout_diag_buf['done_frac'].append(float(done_frac))

            if self.time_profile_enabled:
                now_t = time.perf_counter()
                episode_wall = max(0.0, now_t - self._episode_wall_t0)
                self._episode_wall_t0 = now_t
                self._episode_perf_buf['episode_wall'].append(float(episode_wall))
                self._episode_perf_buf['act_time'].append(float(self._episode_act_time))
                self._episode_perf_buf['step_time'].append(float(self._episode_step_time))
                self._episode_perf_buf['act_calls'].append(int(self._episode_act_calls))
                self._episode_perf_buf['step_calls'].append(int(self._episode_step_calls))
                self._episode_act_time = 0.0
                self._episode_step_time = 0.0
                self._episode_act_calls = 0
                self._episode_step_calls = 0

                if (self.episode_count + 1) % self._perf_log_interval == 0 and self._episode_perf_buf['episode_wall']:
                    wall_mean = float(np.mean(self._episode_perf_buf['episode_wall']))
                    act_mean = float(np.mean(self._episode_perf_buf['act_time']))
                    step_mean = float(np.mean(self._episode_perf_buf['step_time']))
                    non_policy_mean = max(0.0, wall_mean - act_mean - step_mean)
                    act_calls_mean = float(np.mean(self._episode_perf_buf['act_calls']))
                    step_calls_mean = float(np.mean(self._episode_perf_buf['step_calls']))
                    print(
                        f"[PerfDiag] ep={self.episode_count + 1} interval={self._perf_log_interval} "
                        f"episode_wall={wall_mean:.3f}s act={act_mean:.3f}s step={step_mean:.3f}s "
                        f"non_policy={non_policy_mean:.3f}s act_calls={act_calls_mean:.1f} step_calls={step_calls_mean:.1f}"
                    )

            if self.show_pre_train_debug_msg and (self.episode_count + 1) % self._obs_stat_interval == 0:
                win_sp_total = int(sum(self._rollout_diag_buf['sp_total']))
                win_sp_match = int(sum(self._rollout_diag_buf['sp_match']))
                win_sp_acc = (win_sp_match / win_sp_total) if win_sp_total > 0 else 0.0
                win_done = float(np.mean(self._rollout_diag_buf['done_frac'])) if self._rollout_diag_buf['done_frac'] else 0.0
                win_timeout = float(np.mean(self._rollout_diag_buf['timeout_frac'])) if self._rollout_diag_buf['timeout_frac'] else 0.0
                win_len = float(np.mean(self._rollout_diag_buf['ep_len'])) if self._rollout_diag_buf['ep_len'] else 0.0
                win_aux = float(np.mean(self._rollout_diag_buf['final_aux'])) if self._rollout_diag_buf['final_aux'] else 0.0
                print(
                    f"[RolloutDiag] ep={self.episode_count + 1} "
                    f"done_win={win_done:.3f} timeout_win={win_timeout:.3f} "
                    f"sp_match_win={win_sp_acc:.3f} (n={win_sp_total}) "
                    f"ep_len_win={win_len:.1f} final_aux_win={win_aux:.3f}"
                )

            self.accumulated_episodes.append(self.current_episode_memory)
            self.current_episode_memory = EpisodeBuffers()

            if self.episode_count % self.train_frequency == 0:
                if self.show_pre_train_debug_msg:
                    print(f"\n🎯 Training with sliding window of {len(self.accumulated_episodes)} episodes...")
                self.train_net_accumulated()

            if (self.episode_count + 1) % self._obs_stat_interval == 0 and self._obs_stat_buffer:
                self._print_obs_statistics()

            self.episode_count += 1

    def save(self, filename):
        self.actor_critic_model.save(filename)
        self.encoder_actor.save(filename + "_actor")
        self.encoder_critic.save(filename + "_critic")
        torch.save(self.optimizer_actor.state_dict(), filename + ".optimizer_actor")
        torch.save(self.optimizer_critic.state_dict(), filename + ".optimizer_critic")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            obj.load_state_dict(torch.load(filename, map_location=self.device))
        return obj

    @staticmethod
    def _optimizer_state_compatible(optimizer) -> bool:
        """Validate that tensor-shaped optimizer states match current param shapes."""
        for group in optimizer.param_groups:
            for p in group['params']:
                if p not in optimizer.state:
                    continue
                st = optimizer.state[p]
                for _, v in st.items():
                    if torch.is_tensor(v) and tuple(v.shape) != tuple(p.shape):
                        return False
        return True

    def _reset_critic_optimizer(self):
        """Recreate critic-head optimizer after architecture changes."""
        self.optimizer_critic_head = optim.AdamW(
            self.actor_critic_model.critic.parameters(),
            lr=self.base_lr_critic_head
        )
        self.optimizer_critic = self.optimizer_critic_head

    def load(self, filename):
        self.actor_critic_model.load(filename)
        self.encoder_actor.load(filename + "_actor")
        self.encoder_critic.load(filename + "_critic")
        self.optimizer_actor = self._load(self.optimizer_actor, filename + ".optimizer_actor")
        self.optimizer_critic = self._load(self.optimizer_critic, filename + ".optimizer_critic")
        self.optimizer_actor_head = self.optimizer_actor
        self.optimizer_critic_head = self.optimizer_critic
        if not self._optimizer_state_compatible(self.optimizer_critic_head):
            print(" >> critic optimizer state incompatible with current critic shape; reinitializing critic optimizer")
            self._reset_critic_optimizer()
        self._set_actor_lr_factor(self.actor_lr_factor)
        self._set_critic_lr_defaults()
        print('{} -> load {} ok'.format(self.get_name(), filename))

    def clone(self):
        policy = MARL_ATTENTION_TEMPORAL_PPOPolicy(self.state_size, self.action_size, self.ppo_parameters)
        policy.actor_critic_model = copy.deepcopy(self.actor_critic_model)
        policy.encoder_actor = copy.deepcopy(self.encoder_actor)
        policy.encoder_critic = copy.deepcopy(self.encoder_critic)
        policy.optimizer_actor = copy.deepcopy(self.optimizer_actor)
        policy.optimizer_critic = copy.deepcopy(self.optimizer_critic)
        return policy

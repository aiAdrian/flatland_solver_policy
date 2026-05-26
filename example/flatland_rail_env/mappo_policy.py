# =============================================================================
# mappo_policy.py
# -----------------------------------------------------------------------------
# Minimal but solid MAPPO policy for Flatland decision-point MARL.
#
# Architecture:
#   - BaseFeatureEncoder (MLP over 22D base obs)
#   - TreePayloadEncoder (edge-aware GNN, 1 message-passing layer)
#   - ActorCriticHead (2x MLP, Critic with mean-pool over neighbors → CTDE)
#
# Training:
#   - Behavior Cloning warmstart from DLA expert demos
#   - PPO with GAE, action masking, gradient clipping
#   - Single AdamW optimizer (no separate actor/critic LRs — keep it simple)
#
# References:
#   [PPO]   Schulman et al. (2017), arXiv:1707.06347
#   [GAE]   Schulman et al. (2015), arXiv:1506.02438
#   [MAPPO] Yu et al. (2022), arXiv:2103.01955
#   [Mask]  Huang & Ontañón (2022), arXiv:2006.14171
#   [BC]    Pomerleau (1989); Ross & Bagnell (2010)
# =============================================================================

import os
import sys
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -----------------------------------------------------------------------------
# Path setup so this file can be imported regardless of CWD.
# -----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -----------------------------------------------------------------------------
# Flatland + project imports.
# -----------------------------------------------------------------------------
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.fast_methods import fast_count_nonzero
from flatland.envs.step_utils.states import TrainState

from policy.learning_policy.learning_policy import LearningPolicy
from environment.environment import Environment
from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils


# =============================================================================
# 1) BaseFeatureEncoder — 22D handcrafted obs → hidden vector.
# =============================================================================

class BaseFeatureEncoder(nn.Module):
    """MLP encoder for 22D base observation (DecisionPointObservation)."""

    def __init__(self, base_dim: int = 22, hidden: int = 64):
        super().__init__()
        self.base_dim = int(base_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(base_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 2) TreePayloadEncoder — variable-size graph (nodes + edges) → hidden vector.
#
# Uses one message-passing layer:
#     m_v   = Σ_u→v gate(e_uv) * h_u
#     h_v'  = update([h_v, m_v])
# Then masked mean over node embeddings.
# =============================================================================

class TreePayloadEncoder(nn.Module):

    NODE_DIM = 12  # See _encode_node().
    EDGE_DIM = 20  # See _encode_edge().
    MAX_NODES = 32  # Hard cap; trees larger than this are truncated.

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = int(hidden)

        self.node_proj = nn.Sequential(
            nn.Linear(self.NODE_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(self.EDGE_DIM, hidden // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden // 2, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------------------------------------------------------
    # Numpy feature encoders (one node / one edge → fixed-length float vector).
    # -------------------------------------------------------------------------
    @staticmethod
    def _encode_node(n: Dict[str, Any]) -> np.ndarray:
        depth = int(n.get("depth", 0))
        is_branch = bool(n.get("is_branch", False) or bool(n.get("is_switch", False)))
        alt = int(n.get("alternative_routes_count", 0))
        feat = np.array([
            float(n.get("deadlock_risk", 0.0)),
            min(1.0, float(n.get("num_transitions", 1)) / 3.0),
            1.0 if n.get("has_oncoming", False) else 0.0,
            min(1.0, float(n.get("backward_inflow_count", 0)) / 2.0),
            min(1.0, float(max(depth, 0)) / 12.0),
            1.0 if n.get("has_agents_encountered", False) else 0.0,
            min(1.0, float(n.get("incoming_agent_count", 0)) / 2.0),
            1.0 if is_branch else 0.0,
            float(n.get("deadlock_distance_norm", 0.0)),
            float(n.get("deadlock_hard_distance_norm", 0.0)),
            1.0 if n.get("deadlock_exists_within_probe", False) else 0.0,
            min(1.0, float(alt) / 3.0),
        ], dtype=np.float32)
        np.clip(feat, 0.0, 1.0, out=feat)
        return feat

    @staticmethod
    def _encode_edge(e: Dict[str, Any]) -> np.ndarray:
        def _f(v, d=0.0):
            try:
                x = float(v)
                return x if np.isfinite(x) else d
            except (TypeError, ValueError):
                return d

        def _norm_d(v):
            d = _f(v, 1.0)
            return float(d) if d <= 1.0 else float(d / (d + 32.0))

        rel = int(e.get("rel_dir_bin", 1))
        a_l = _f(e.get("action_left", 1.0 if rel == 0 else 0.0))
        a_f = _f(e.get("action_forward", 1.0 if rel == 1 else 0.0))
        a_r = _f(e.get("action_right", 1.0 if rel == 2 else 0.0))
        n_agents_edge = int(e.get("agents_on_edge_count", 0))
        if n_agents_edge <= 0:
            agents_on = e.get("agents_on_edge", [])
            if isinstance(agents_on, list):
                n_agents_edge = len(agents_on)

        # Deadlock-distance-delta is signed in [-1, +1] (positive = improving).
        # Remap to [0, 1] so the "no-info" default of 0.0 ends up at neutral 0.5.
        dl_delta = _f(e.get("deadlock_distance_delta", 0.0))
        dl_delta_unit = 0.5 * (max(-1.0, min(1.0, dl_delta)) + 1.0)

        feat = np.array([
            a_l, a_f, a_r,
            1.0 if n_agents_edge > 0 else 0.0,
            1.0 if e.get("has_oncoming_edge", False) else 0.0,
            min(1.0, _f(e.get("edge_len_cells", 1), 1.0) / 4.0),
            _norm_d(e.get("src_dist_to_target", 1.0)),
            _norm_d(e.get("dst_dist_to_target", 1.0)),
            _f(e.get("delta_from_root", 0.5), 0.5),
            _f(e.get("improves_over_current", 0.0)),
            1.0 if e.get("target_on_edge", False) else 0.0,
            _f(e.get("dst_deadlock_risk", 0.0)),
            _f(e.get("dst_deadlock_hard_block", 0.0)),
            _f(e.get("dst_deadlock_distance_norm", 0.0)),
            _f(e.get("dst_deadlock_hard_distance_norm", 0.0)),
            _f(e.get("src_deadlock_distance_norm", 0.0)),
            dl_delta_unit,
            _f(e.get("is_shortest_path_edge", 0.0)),
            _f(e.get("branch_choice_prob", 0.0)),
            min(1.0, _f(e.get("alternative_routes_count", 0)) / 3.0),
        ], dtype=np.float32)
        np.clip(feat, 0.0, 1.0, out=feat)
        return feat

    # -------------------------------------------------------------------------
    # Convert one payload dict → padded node array + edge list with src/dst idx.
    # -------------------------------------------------------------------------
    def _payload_to_graph(self, payload: Dict[str, Any], max_nodes: int):
        max_nodes = max(1, int(max_nodes))
        node_feats = np.zeros((max_nodes, self.NODE_DIM), dtype=np.float32)
        edge_list: List[Tuple[int, int, np.ndarray]] = []
        if not isinstance(payload, dict):
            return node_feats, edge_list, 0

        nodes = payload.get("nodes", []) or []
        edges = payload.get("edges", []) or []

        idx_simple: Dict[Tuple[int, int, int], int] = {}
        idx_pos: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        n_valid = min(len(nodes), max_nodes)
        for i in range(n_valid):
            node = nodes[i]
            pos = node.get("pos", (0, 0))
            d = int(node.get("dir", 0))
            depth = int(node.get("depth", 0))
            node_feats[i] = self._encode_node(node)
            idx_simple[(int(pos[0]), int(pos[1]), d)] = i
            pos_key = (int(pos[0]), int(pos[1]))
            idx_pos.setdefault(pos_key, []).append((depth, i))

        for plist in idx_pos.values():
            plist.sort(key=lambda x: x[0])

        def _resolve(key3: Tuple[int, int, int]) -> Optional[int]:
            i = idx_simple.get(key3)
            if i is not None:
                return i
            pos_list = idx_pos.get((key3[0], key3[1]))
            if pos_list:
                return pos_list[0][1]
            return None

        for edge in edges:
            if "src" in edge and "dst" in edge:
                s_i = int(edge["src"])
                d_i = int(edge["dst"])
            else:
                s_pos = edge.get("src_pos", (0, 0))
                d_pos = edge.get("dst_pos", (0, 0))
                s_dir = int(edge.get("src_dir", 0))
                d_dir = int(edge.get("dst_dir", 0))
                s_i = _resolve((int(s_pos[0]), int(s_pos[1]), s_dir))
                d_i = _resolve((int(d_pos[0]), int(d_pos[1]), d_dir))
                if s_i is None or d_i is None:
                    continue
            if s_i < 0 or d_i < 0 or s_i >= max_nodes or d_i >= max_nodes:
                continue
            edge_list.append((s_i, d_i, self._encode_edge(edge)))

        return node_feats, edge_list, n_valid

    # -------------------------------------------------------------------------
    # Forward pass for a batch of payloads → (B, hidden).
    # -------------------------------------------------------------------------
    def forward_batch(self, payload_list: List[Dict[str, Any]]) -> torch.Tensor:
        device = next(self.parameters()).device
        B = len(payload_list)
        if B == 0:
            return torch.empty(0, self.hidden, device=device)

        # Determine dynamic node padding for this batch.
        max_nodes = 1
        for p in payload_list:
            if isinstance(p, dict):
                nodes = p.get("nodes", []) or []
                max_nodes = max(max_nodes, len(nodes))
        max_nodes = min(self.MAX_NODES, max_nodes)

        node_arr = np.zeros((B, max_nodes, self.NODE_DIM), dtype=np.float32)
        node_mask = torch.zeros((B, max_nodes), dtype=torch.float32, device=device)
        edge_graphs: List[List[Tuple[int, int, np.ndarray]]] = []

        for b, payload in enumerate(payload_list):
            n_feat, e_list, n_valid = self._payload_to_graph(payload, max_nodes)
            node_arr[b] = n_feat
            edge_graphs.append(e_list)
            if n_valid > 0:
                node_mask[b, :n_valid] = 1.0

        nodes = torch.as_tensor(node_arr, device=device)
        h_node = self.node_proj(nodes)                   # (B, N, H)
        messages = torch.zeros_like(h_node)              # (B, N, H)

        for b, e_list in enumerate(edge_graphs):
            if not e_list:
                continue
            for s_i, d_i, e_feat_np in e_list:
                if s_i >= max_nodes or d_i >= max_nodes:
                    continue
                e_t = torch.as_tensor(e_feat_np, device=device)
                gate = torch.sigmoid(self.edge_gate(e_t)).squeeze(-1)
                messages[b, d_i] += gate * h_node[b, s_i]

        h_updated = self.update(torch.cat([h_node, messages], dim=-1))   # (B, N, H)

        # Masked mean pool over nodes (B, H).
        mask_exp = node_mask.unsqueeze(-1)
        denom = mask_exp.sum(dim=1).clamp(min=1.0)
        pooled = (h_updated * mask_exp).sum(dim=1) / denom
        return self.output(pooled)


# =============================================================================
# 3) ActorCriticHead — actor-MLP + centralized critic-MLP with neighbor pooling.
# =============================================================================

class ActorCriticHead(nn.Module):
    """Actor uses local embedding only.

    Critic also receives a mean-pool over neighbor base-features (CTDE,
    MAPPO §4.2 Agent-Specific Global State pattern).
    """

    def __init__(self, hidden: int, action_size: int, base_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_size),
        )
        # Critic receives concatenated [self_emb, neighbor_pool_emb].
        self.neighbor_proj = nn.Sequential(
            nn.Linear(base_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def actor_logits(self, self_emb: torch.Tensor) -> torch.Tensor:
        return self.actor(self_emb)

    def value(self, self_emb: torch.Tensor, neighbor_pool: torch.Tensor) -> torch.Tensor:
        n_emb = self.neighbor_proj(neighbor_pool)
        x = torch.cat([self_emb, n_emb], dim=-1)
        return self.critic(x).squeeze(-1)


# =============================================================================
# 4) RolloutBuffer — simple per-handle FIFO of transitions for one episode.
# =============================================================================

class RolloutBuffer:
    """Stores transitions per agent handle.

    Each transition: (base_obs, opp_pool_vec, payload, action, reward, done, finished).
    """

    def __init__(self):
        self.data: Dict[int, List[Tuple]] = {}

    def push(self, handle: int, transition: Tuple):
        self.data.setdefault(int(handle), []).append(transition)

    def get(self, handle: int) -> List[Tuple]:
        return self.data.get(int(handle), [])

    def reset(self):
        self.data = {}

    def all_transitions(self) -> List[Tuple]:
        out = []
        for ts in self.data.values():
            out.extend(ts)
        return out

    def __len__(self):
        return sum(len(v) for v in self.data.values())
# =============================================================================
# 5) MAPPOPolicy — main policy class implementing LearningPolicy interface.
# -----------------------------------------------------------------------------
# Public API (compatible with FlatlandSolver):
#   - reset(env)
#   - start_step(train)
#   - act(handle, state, eps)
#   - step(handle, state, action, reward, next_state, done, agent_finished)
#   - end_episode(train)
#   - save(filename), load(filename)
#
# Extra methods (called from train_marl.py, not from FlatlandSolver):
#   - train_bc(demos, n_epochs)  → behavior cloning warmstart
#   - get_name()
# =============================================================================

class MAPPOPolicy(LearningPolicy):

    BASE_DIM = 22       # DecisionPointObservation base vector size.
    ACTION_SIZE = 5     # DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP.

    def __init__(
        self,
        hidden: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.20,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)

        # ---- Networks ------------------------------------------------------
        self.base_encoder = BaseFeatureEncoder(self.BASE_DIM, self.hidden).to(self.device)
        self.tree_encoder = TreePayloadEncoder(self.hidden).to(self.device)
        self.fuse = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.LeakyReLU(0.01),
        ).to(self.device)
        self.head = ActorCriticHead(self.hidden, self.ACTION_SIZE, self.BASE_DIM).to(self.device)

        # ---- Optimizer (single shared, simple) -----------------------------
        all_params = (
            list(self.base_encoder.parameters())
            + list(self.tree_encoder.parameters())
            + list(self.fuse.parameters())
            + list(self.head.parameters())
        )
        self.optimizer = optim.AdamW(all_params, lr=self.lr)

        # ---- Episode state -------------------------------------------------
        self.env: Optional[Environment] = None
        self.buffer = RolloutBuffer()
        self.episode_count = 0

        # ---- Logging --------------------------------------------------------
        self.last_train_stats: Dict[str, float] = {}
        self.action_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.last_action_dist: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

    # -------------------------------------------------------------------------
    # Helpers: state unwrapping (supports both temporal and non-temporal obs).
    # -------------------------------------------------------------------------
    @staticmethod
    def _unwrap_state(state) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
        """Return (base_obs_22D, list_of_neighbor_base_obs, tree_payload_dict).

        Supports:
          - Pattern A: state = [(obs, opps, payload), ...] (temporal sequence)
          - Pattern B: state = (obs, opps, payload)        (single timestep)
          - Pattern C: state = obs                          (just base vector)
        """
        # Temporal sequence → take latest timestep.
        if isinstance(state, list) and len(state) > 0 and isinstance(state[0], (tuple, list)):
            state = state[-1]

        if isinstance(state, (tuple, list)):
            if len(state) >= 3:
                obs, opps, payload = state[0], state[1], state[2]
                if not isinstance(opps, list):
                    opps = []
                if not isinstance(payload, dict):
                    payload = {}
                return np.asarray(obs, dtype=np.float32).flatten(), opps, payload
            if len(state) >= 1:
                return np.asarray(state[0], dtype=np.float32).flatten(), [], {}

        # Plain vector.
        return np.asarray(state, dtype=np.float32).flatten(), [], {}

    @staticmethod
    def _neighbor_pool(opps: List[np.ndarray], base_dim: int) -> np.ndarray:
        """Mean-pool over neighbor base observations. Returns zero vector if none."""
        if not opps:
            return np.zeros(base_dim, dtype=np.float32)
        arrs = []
        for o in opps:
            v = np.asarray(o, dtype=np.float32).flatten()
            if v.shape[0] >= base_dim:
                arrs.append(v[:base_dim])
        if not arrs:
            return np.zeros(base_dim, dtype=np.float32)
        return np.mean(np.stack(arrs, axis=0), axis=0)

    # -------------------------------------------------------------------------
    # Action masking (Huang & Ontañón 2022).
    # Reads path bits from base_obs[0:3] = [path_left, path_forward, path_right].
    # -------------------------------------------------------------------------
    def _legal_action_mask(self, base_obs: np.ndarray, agent) -> np.ndarray:
        mask = np.zeros(self.ACTION_SIZE, dtype=np.float32)

        if agent.state == TrainState.DONE:
            mask[RailEnvActions.DO_NOTHING] = 1.0
            return mask
        if agent.state == TrainState.WAITING:
            mask[RailEnvActions.DO_NOTHING] = 1.0
            return mask
        if agent.state.is_off_map_state():
            mask[RailEnvActions.DO_NOTHING] = 1.0
            mask[RailEnvActions.STOP_MOVING] = 1.0
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
            return mask

        # On-map: STOP is always legal; movement actions follow path bits.
        mask[RailEnvActions.STOP_MOVING] = 1.0

        if base_obs.shape[0] >= 3:
            left_ok = float(base_obs[0]) > 0.5
            fwd_ok = float(base_obs[1]) > 0.5
            right_ok = float(base_obs[2]) > 0.5
            n_trans = int(left_ok) + int(fwd_ok) + int(right_ok)
            if n_trans == 0:
                # B1 FIX: defensive — no transition possible, only DO_NOTHING/STOP
                pass
            elif n_trans == 1:
                # 1-transition cells (curves, dead-ends) → forward is canonical.
                mask[RailEnvActions.MOVE_FORWARD] = 1.0
            else:
                mask[RailEnvActions.MOVE_LEFT] = 1.0 if left_ok else 0.0
                mask[RailEnvActions.MOVE_FORWARD] = 1.0 if fwd_ok else 0.0
                mask[RailEnvActions.MOVE_RIGHT] = 1.0 if right_ok else 0.0
        else:
            mask[RailEnvActions.MOVE_FORWARD] = 1.0

        return mask

    # -------------------------------------------------------------------------
    # Forward pass for one or many states → (logits, values, embeddings).
    # -------------------------------------------------------------------------
    def _forward(
        self,
        base_obs_t: torch.Tensor,
        payload_list: List[Dict[str, Any]],
        neigh_pool_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_base = self.base_encoder(base_obs_t)            # (B, H)
        emb_tree = self.tree_encoder.forward_batch(payload_list)  # (B, H)
        emb = self.fuse(torch.cat([emb_base, emb_tree], dim=-1))  # (B, H)
        logits = self.head.actor_logits(emb)                # (B, A)
        values = self.head.value(emb, neigh_pool_t)         # (B,)
        return logits, values, emb

    @staticmethod
    def _masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(masks < 0.5, -1e9)

    # -------------------------------------------------------------------------
    # LearningPolicy interface methods.
    # -------------------------------------------------------------------------
    def get_name(self) -> str:
        return "MAPPOPolicy"

    def _classify_cell_type(self, agent, raw_env):
        """Required by FlatlandSolver.run_policy_step() to enable
        state-machine reduction (Laurent et al. 2021).

        Skips FORWARD_ONLY->FORWARD_ONLY transitions in the solver
        layer so we get cleaner training samples in step().
        """
        try:
            return DecisionPointUtils.classify_cell_type(agent, raw_env)
        except Exception:
            return "OUTSIDE"

    def reset(self, env: Environment):
        self.env = env
        self.buffer = RolloutBuffer()

    def start_step(self, train: bool):
        # No-op (kept for FlatlandSolver compatibility).
        pass

    def act(self, handle: int, state, eps: float = 0.0) -> int:
        """Sample action from masked policy. Used at rollout time."""
        base_obs, opps, payload = self._unwrap_state(state)
        agent = self.env.raw_env.agents[handle]
        mask_np = self._legal_action_mask(base_obs, agent)
        legal = np.flatnonzero(mask_np > 0.5)

        if legal.size == 0:
            return int(RailEnvActions.DO_NOTHING)

        # Epsilon-greedy random exploration on legal actions.
        if eps > 0.0 and np.random.rand() < float(eps):
            action = int(np.random.choice(legal))
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            return action


        with torch.no_grad():
            base_t = torch.from_numpy(base_obs[:self.BASE_DIM]).float().unsqueeze(0).to(self.device)
            n_pool = self._neighbor_pool(opps, self.BASE_DIM)
            n_pool_t = torch.from_numpy(n_pool).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).to(self.device)

            logits, _, _ = self._forward(base_t, [payload], n_pool_t)
            logits = self._masked_logits(logits, mask_t)
            action = Categorical(logits=logits).sample().item()

        action = int(action)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        return action


    def step(self, handle, state, action, reward, next_state, done, agent_finished=None):
        """Store transition for later PPO update."""
        base_obs, opps, payload = self._unwrap_state(state)
        n_pool = self._neighbor_pool(opps, self.BASE_DIM)

        finished = bool(done) if agent_finished is None else bool(agent_finished)
        self.buffer.push(
            handle,
            (
                base_obs[:self.BASE_DIM].copy(),
                n_pool.copy(),
                payload,
                int(action),
                float(reward),
                bool(done),
                bool(
                    finished),
            ),
        )

    def end_episode(self, train: bool):
        """Run PPO update if training."""
        self.episode_count += 1

        # Snapshot action distribution for logging, then reset
        total = sum(self.action_counts.values())
        if total > 0:
            self.last_action_dist = {
                k: self.action_counts[k] / total for k in range(5)
            }
        else:
            self.last_action_dist = {k: 0.0 for k in range(5)}
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        if not train:
            self.buffer.reset()
            return

        if len(self.buffer) < self.batch_size:
            self.buffer.reset()
            return

        self._train_ppo()
        self.buffer.reset()

    # -------------------------------------------------------------------------
    # PPO update (single rollout buffer → multi-epoch minibatch updates).
    # -------------------------------------------------------------------------
    def _build_tensors_from_buffer(self):
        """Concatenate per-handle trajectories and compute GAE.

        IMPORTANT: GAE is computed PER HANDLE (per agent trajectory) because
        boundaries between agents are not consecutive in time. Then everything
        is concatenated into one big batch for SGD.
        """
        all_base, all_pool, all_payload = [], [], []
        all_act, all_old_lp = [], []
        all_adv, all_ret, all_mask = [], [], []

        for handle, transitions in self.buffer.data.items():
            if not transitions:
                continue

            T = len(transitions)
            base_arr = np.stack([t[0] for t in transitions], axis=0)
            pool_arr = np.stack([t[1] for t in transitions], axis=0)
            payloads = [t[2] for t in transitions]
            actions = np.array([t[3] for t in transitions], dtype=np.int64)
            rewards = np.array([t[4] for t in transitions], dtype=np.float32)
            dones = np.array([t[5] for t in transitions], dtype=np.float32)
            finished = np.array([t[6] for t in transitions], dtype=np.float32)

            # Build action masks from base_obs (path bits at indices 0..2).
            masks = np.zeros((T, self.ACTION_SIZE), dtype=np.float32)
            for i in range(T):
                v = base_arr[i]
                masks[i, RailEnvActions.STOP_MOVING] = 1.0
                masks[i, RailEnvActions.DO_NOTHING] = 1.0  # Always-legal superset.
                left_ok = float(v[0]) > 0.5 if v.shape[0] >= 1 else False
                fwd_ok = float(v[1]) > 0.5 if v.shape[0] >= 2 else False
                right_ok = float(v[2]) > 0.5 if v.shape[0] >= 3 else False
                n_trans = int(left_ok) + int(fwd_ok) + int(right_ok)
                if n_trans == 0:
                    pass
                elif n_trans == 1:
                    masks[i, RailEnvActions.MOVE_FORWARD] = 1.0
                else:
                    masks[i, RailEnvActions.MOVE_LEFT] = 1.0 if left_ok else 0.0
                    masks[i, RailEnvActions.MOVE_FORWARD] = 1.0 if fwd_ok else 0.0
                    masks[i, RailEnvActions.MOVE_RIGHT] = 1.0 if right_ok else 0.0


            # Forward pass to get values + old log-probs (no grad).
            with torch.no_grad():
                base_t = torch.from_numpy(base_arr).float().to(self.device)
                pool_t = torch.from_numpy(pool_arr).float().to(self.device)
                act_t = torch.from_numpy(actions).long().to(self.device)
                mask_t = torch.from_numpy(masks).float().to(self.device)
                logits, values, _ = self._forward(base_t, payloads, pool_t)
                logits = self._masked_logits(logits, mask_t)
                old_lp = Categorical(logits=logits).log_prob(act_t).cpu().numpy()
                values_np = values.cpu().numpy()

            # Compute GAE per trajectory.
            advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            # Bootstrap value of state after T (use last critic value if not finished).
            next_value = 0.0 if (T > 0 and finished[-1] > 0.5) else float(values_np[-1])
            for t in reversed(range(T)):
                next_v = next_value if t == T - 1 else values_np[t + 1]
                # Use 'finished' to cut bootstrap (true terminal), 'dones' to cut chain.
                delta = rewards[t] + self.gamma * next_v * (1.0 - finished[t]) - values_np[t]
                gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - dones[t])
                advantages[t] = gae
            returns = advantages + values_np

            all_base.append(base_arr)
            all_pool.append(pool_arr)
            all_payload.extend(payloads)
            all_act.append(actions)
            all_old_lp.append(old_lp)
            all_adv.append(advantages)
            all_ret.append(returns)
            all_mask.append(masks)

        if not all_base:
            return None

        base_arr = np.concatenate(all_base, axis=0)
        pool_arr = np.concatenate(all_pool, axis=0)
        actions = np.concatenate(all_act, axis=0)
        old_lp = np.concatenate(all_old_lp, axis=0)
        adv = np.concatenate(all_adv, axis=0)
        ret = np.concatenate(all_ret, axis=0)
        mask = np.concatenate(all_mask, axis=0)

        # Normalize advantages once per update (not per minibatch).
        adv_mean = float(np.mean(adv))
        adv_std = float(np.std(adv) + 1e-8)
        adv_norm = (adv - adv_mean) / adv_std
        adv_norm = np.clip(adv_norm, -5.0, 5.0)

        return {
            "base": torch.from_numpy(base_arr).float().to(self.device),
            "pool": torch.from_numpy(pool_arr).float().to(self.device),
            "payload": all_payload,
            "act": torch.from_numpy(actions).long().to(self.device),
            "old_lp": torch.from_numpy(old_lp).float().to(self.device),
            "adv": torch.from_numpy(adv_norm).float().to(self.device),
            "ret": torch.from_numpy(ret).float().to(self.device),
            "mask": torch.from_numpy(mask).float().to(self.device),
        }

    def _train_ppo(self):
        data = self._build_tensors_from_buffer()
        if data is None:
            return

        N = data["act"].shape[0]
        stats = {"v_loss": [], "p_loss": [], "ent": [], "kl": [], "ratio": [], "clip_frac": []}

        for _ in range(self.ppo_epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                mb = idx[start:start + self.batch_size]
                if len(mb) < 8:
                    continue

                b_base = data["base"][mb]
                b_pool = data["pool"][mb]
                b_payload = [data["payload"][i] for i in mb]
                b_act = data["act"][mb]
                b_old_lp = data["old_lp"][mb]
                b_adv = data["adv"][mb]
                b_ret = data["ret"][mb]
                b_mask = data["mask"][mb]

                logits, values, _ = self._forward(b_base, b_payload, b_pool)
                logits = self._masked_logits(logits, b_mask)
                dist = Categorical(logits=logits)
                lp = dist.log_prob(b_act)
                ent = dist.entropy().mean()

                ratio = torch.exp(lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((values - b_ret) ** 2).mean()
                loss = p_loss + self.value_coef * v_loss - self.entropy_coef * ent

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.base_encoder.parameters())
                    + list(self.tree_encoder.parameters())
                    + list(self.fuse.parameters())
                    + list(self.head.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    kl = (b_old_lp - lp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                stats["v_loss"].append(float(v_loss.item()))
                stats["p_loss"].append(float(p_loss.item()))
                stats["ent"].append(float(ent.item()))
                stats["kl"].append(float(kl))
                stats["ratio"].append(float(ratio.mean().item()))
                stats["clip_frac"].append(float(clip_frac))

        self.last_train_stats = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}

    # -------------------------------------------------------------------------
    # Behavior Cloning warmstart from expert (action) demonstrations.
    # demos: list of (base_obs, opps_list, payload, expert_action) tuples.
    # -------------------------------------------------------------------------
    def train_bc(self, demos: List[Tuple], n_epochs: int = 10, batch_size: int = 256):
        if len(demos) == 0:
            print("[BC] No demos provided.")
            return {"bc_acc": 0.0, "bc_loss": 0.0}

        # Pre-encode masks etc.
        N = len(demos)
        base_arr = np.stack([np.asarray(d[0], dtype=np.float32)[:self.BASE_DIM] for d in demos], axis=0)
        pool_arr = np.stack([self._neighbor_pool(d[1], self.BASE_DIM) for d in demos], axis=0)
        payloads = [d[2] if isinstance(d[2], dict) else {} for d in demos]
        actions = np.array([int(d[3]) for d in demos], dtype=np.int64)

        # Reconstruct legal masks from base obs.
        masks = np.zeros((N, self.ACTION_SIZE), dtype=np.float32)
        for i in range(N):
            v = base_arr[i]
            masks[i, RailEnvActions.STOP_MOVING] = 1.0
            masks[i, RailEnvActions.DO_NOTHING] = 1.0
            left_ok = float(v[0]) > 0.5
            fwd_ok = float(v[1]) > 0.5
            right_ok = float(v[2]) > 0.5
            n_trans = int(left_ok) + int(fwd_ok) + int(right_ok)
            if n_trans == 0:
                pass
            elif n_trans == 1:
                masks[i, RailEnvActions.MOVE_FORWARD] = 1.0
            else:
                masks[i, RailEnvActions.MOVE_LEFT] = 1.0 if left_ok else 0.0
                masks[i, RailEnvActions.MOVE_FORWARD] = 1.0 if fwd_ok else 0.0
                masks[i, RailEnvActions.MOVE_RIGHT] = 1.0 if right_ok else 0.0


        # Filter out demos where the expert action is masked illegal
        # (rare, but keeps cross-entropy well-defined).
        legal_for_expert = masks[np.arange(N), actions] > 0.5
        keep = np.flatnonzero(legal_for_expert)
        if keep.shape[0] < N:
            print(f"[BC] Filtered {N - keep.shape[0]}/{N} demos with illegal expert actions.")
        if keep.shape[0] == 0:
            print("[BC] No valid demos after filtering.")
            return {"bc_acc": 0.0, "bc_loss": 0.0}

        base_arr = base_arr[keep]
        pool_arr = pool_arr[keep]
        payloads = [payloads[i] for i in keep.tolist()]
        actions = actions[keep]
        masks = masks[keep]
        N = base_arr.shape[0]

        base_t = torch.from_numpy(base_arr).float().to(self.device)
        pool_t = torch.from_numpy(pool_arr).float().to(self.device)
        act_t = torch.from_numpy(actions).long().to(self.device)
        mask_t = torch.from_numpy(masks).float().to(self.device)

        loss_fn = nn.CrossEntropyLoss()
        last_loss, last_acc = 0.0, 0.0

        print(f"[BC] Training on {N} demos for {n_epochs} epochs (batch={batch_size})...")

        for epoch in range(n_epochs):
            perm = np.random.permutation(N)
            ep_losses, ep_correct, ep_total = [], 0, 0

            for start in range(0, N, batch_size):
                mb = perm[start:start + batch_size]
                if len(mb) < 4:
                    continue
                b_base = base_t[mb]
                b_pool = pool_t[mb]
                b_payload = [payloads[i] for i in mb]
                b_act = act_t[mb]
                b_mask = mask_t[mb]

                logits, _, _ = self._forward(b_base, b_payload, b_pool)
                logits = self._masked_logits(logits, b_mask)
                loss = loss_fn(logits, b_act)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.base_encoder.parameters())
                    + list(self.tree_encoder.parameters())
                    + list(self.fuse.parameters())
                    + list(self.head.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    ep_correct += int((pred == b_act).sum().item())
                    ep_total += int(b_act.shape[0])
                    ep_losses.append(float(loss.item()))

            last_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            last_acc = float(ep_correct / max(1, ep_total))
            print(f"[BC] Epoch {epoch+1}/{n_epochs}  loss={last_loss:.4f}  acc={last_acc:.3f}")

        return {"bc_acc": last_acc, "bc_loss": last_loss}

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------
    def save(self, filename: str):
        ckpt = {
            "base_encoder": self.base_encoder.state_dict(),
            "tree_encoder": self.tree_encoder.state_dict(),
            "fuse": self.fuse.state_dict(),
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hidden": self.hidden,
            "episode_count": self.episode_count,
        }
        torch.save(ckpt, filename)
        print(f"[Save] {filename}")

    def load(self, filename: str):
        if not os.path.exists(filename):
            print(f"[Load] File not found: {filename}")
            return False
        ckpt = torch.load(filename, map_location=self.device)
        try:
            self.base_encoder.load_state_dict(ckpt["base_encoder"])
            self.tree_encoder.load_state_dict(ckpt["tree_encoder"])
            self.fuse.load_state_dict(ckpt["fuse"])
            self.head.load_state_dict(ckpt["head"])
            if "optimizer" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                except (ValueError, KeyError, RuntimeError) as e:
                    print(f"[Load] Optimizer state skipped ({type(e).__name__}: {e})")
            self.episode_count = int(ckpt.get("episode_count", 0))
            print(f"[Load] {filename}  (episode_count={self.episode_count})")
            return True
        except (KeyError, RuntimeError) as e:
            print(f"[Load] Failed to load {filename}: {e}")
            return False

    def clone(self):
        """Required by LearningPolicy interface."""
        twin = MAPPOPolicy(
            hidden=self.hidden,
            learning_rate=self.lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_eps=self.clip_eps,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
            ppo_epochs=self.ppo_epochs,
            batch_size=self.batch_size,
            device=str(self.device),
        )
        twin.base_encoder.load_state_dict(self.base_encoder.state_dict())
        twin.tree_encoder.load_state_dict(self.tree_encoder.state_dict())
        twin.fuse.load_state_dict(self.fuse.state_dict())
        twin.head.load_state_dict(self.head.state_dict())
        return twin

"""
Hierarchical Decider Policy for Flatland MARL
==============================================

Implements the architecture described in
``HIERARCHICAL_DECIDER_ARCHITECTURE.md``:

* 5 specialist sub-modules (Routing / Merging / Deadlock / Comm / Tree) provide
  embeddings + confidence scores.
* A Decider (Actor-Critic MLP) fuses them and outputs a single
  Flatland-compatible action (5-way) plus a state value.
* Parameter sharing: ONE network for all `N` train agents.
* Communication: each agent attends sparsely (K=4) over its top-K nearest
  neighbours' obs vectors (provided by ``HierarchicalRoutesObservation``
  via the temporal wrapper).
* Auxiliary BCE loss on a 1-step deadlock label gives Deadlock + Comm a
  direct learning signal beyond PPO.

Compatible with the project's `Policy` API (see policy/policy.py):
    reset(env), start_episode(train), start_step(train),
    act(handle, state, eps), step(handle, state, action, reward, next_state, done),
    end_step(train), end_episode(train), save(filename), load(filename),
    get_name().
"""

from collections import deque
from time import perf_counter
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from flatland.envs.fast_methods import fast_count_nonzero
from flatland.envs.rail_env import RailEnvActions

from policy.learning_policy.learning_policy import LearningPolicy
from environment.environment import Environment

# Re-use EpisodeBuffers from the existing MAPPO module.
from marl_attention_temporal_mappo import EpisodeBuffers
from marl_attention_temporal_observation.decision_point_observation import (
    DecisionPointObservation,
)
from marl_attention_temporal_observation.hierarchical_routes_observation import (
    HierarchicalRoutesObservation,
    NEIGHBOR_K,
)


# ============================================================================
# Specialist sub-modules
# ============================================================================

class RoutingSpecialist(nn.Module):
    """Reads the 3 branch blocks (base[6-29]) plus decision_type/hint.

    Output: route_emb (32D) + soft route logits (3) + confidence (1)."""

    def __init__(self, hidden_dim: int = 64, emb_dim: int = 32):
        super().__init__()
        # 4 = decision_type + 3 hint + ... we use base[0:22] = 22 features.
        self.in_dim = 22
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.GELU(),
        )
        self.head_logits = nn.Linear(emb_dim, 3)
        self.head_conf = nn.Linear(emb_dim, 1)
        self.emb_dim = emb_dim

    def forward(self, base_obs: torch.Tensor):
        x = base_obs[..., 0:22]
        emb = self.net(x)
        logits = self.head_logits(emb)
        conf = torch.sigmoid(self.head_conf(emb))
        return emb, logits, conf


class MergingSpecialist(nn.Module):
    """Reads merge blocks (base[22-29]) + local_dl + coord soft-signals + comm.

    Output: merge_emb (32D) + wait_pressure (1) + priority (1)."""

    def __init__(self, comm_dim: int, hidden_dim: int = 64, emb_dim: int = 32):
        super().__init__()
        # 8 (merge fwd+bwd) + 1 (local_dl @42) + 5 (coord @43..47) = 14
        self.in_dim = 14 + comm_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.GELU(),
        )
        self.head_wait = nn.Linear(emb_dim, 1)
        self.head_prio = nn.Linear(emb_dim, 1)
        self.emb_dim = emb_dim

    def forward(self, base_obs: torch.Tensor, comm_emb: torch.Tensor):
        merge = base_obs[..., 22:30]
        local_dl = base_obs[..., 42:43]
        coord = base_obs[..., 43:48]
        x = torch.cat([merge, local_dl, coord, comm_emb], dim=-1)
        emb = self.net(x)
        wait = torch.sigmoid(self.head_wait(emb))
        prio = torch.sigmoid(self.head_prio(emb))
        return emb, wait, prio


class DeadlockSpecialist(nn.Module):
    """Reads all risk features + tree context + LSTM context + comm.

    Output: dl_emb (16D) + p_dl_1 (1) + p_dl_3 (1).

    The 1-step deadlock probability has a direct BCE auxiliary loss,
    giving Comm + LSTM a learning signal independent of PPO.
    """

    def __init__(self, ctx_dim: int, comm_dim: int, hidden_dim: int = 64, emb_dim: int = 16):
        super().__init__()
        # branch dl @5,11,17 + merge dl @22,26 + local @42 + coord pressure @46
        # + tree context @64 (from DecisionPointObservation._local_search/TreeLSTM)
        self.in_dim = 8 + ctx_dim + comm_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.GELU(),
        )
        self.head_p1 = nn.Linear(emb_dim, 1)
        self.head_p3 = nn.Linear(emb_dim, 1)
        self.emb_dim = emb_dim

    def forward(self, base_obs: torch.Tensor, ctx: torch.Tensor, comm_emb: torch.Tensor):
        risks = torch.stack(
            [
                base_obs[..., 5],
                base_obs[..., 11],
                base_obs[..., 17],
                base_obs[..., 22],
                base_obs[..., 26],
                base_obs[..., 42],
                base_obs[..., 46],
                base_obs[..., 64],
            ],
            dim=-1,
        )
        x = torch.cat([risks, ctx, comm_emb], dim=-1)
        emb = self.net(x)
        p1 = torch.sigmoid(self.head_p1(emb))
        p3 = torch.sigmoid(self.head_p3(emb))
        return emb, p1, p3


class TreeSpecialist(nn.Module):
    """Learns a compact representation of tree-derived context.

    Input uses tree_ctx from observation slot [64] plus a small safety context.
    Output: tree_emb (16D) + tree_conf (1).
    """

    def __init__(self, hidden_dim: int = 48, emb_dim: int = 16):
        super().__init__()
        # tree_ctx @64 + local_deadlock @5 + local_confirmed_deadlock @65 +
        # merge flag @4 + decision_required @30
        self.in_dim = 5
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.GELU(),
        )
        self.head_conf = nn.Linear(emb_dim, 1)
        self.emb_dim = emb_dim

    def forward(self, base_obs: torch.Tensor):
        x = torch.stack(
            [
                base_obs[..., 64],
                base_obs[..., 5],
                base_obs[..., 65],
                base_obs[..., 4],
                base_obs[..., 30],
            ],
            dim=-1,
        )
        emb = self.net(x)
        conf = torch.sigmoid(self.head_conf(emb))
        return emb, conf


class CommSpecialist(nn.Module):
    """Sparse multi-head attention over K=4 neighbor obs vectors + temporal.

    Input per timestep: list of K=NEIGHBOR_K neighbor 90D obs vectors.
    We project them, mask non-existing slots, and attend with self_query.
    """

    def __init__(self, obs_dim: int, query_dim: int, comm_dim: int = 32,
                 num_heads: int = 4):
        super().__init__()
        self.proj_kv = nn.Linear(obs_dim, comm_dim)
        self.q_proj = nn.Linear(query_dim, comm_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=comm_dim, num_heads=num_heads, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(comm_dim, 1), nn.Sigmoid()
        )
        self.comm_dim = comm_dim

    def forward(self, neighbors: torch.Tensor, neighbor_mask: torch.Tensor,
                self_query: torch.Tensor):
        """
        neighbors: [B, K, obs_dim]
        neighbor_mask: [B, K] (1 = valid, 0 = padding)
        self_query: [B, query_dim]
        returns: comm_emb [B, comm_dim], gate_score [B, 1]
        """
        kv = self.proj_kv(neighbors)
        q = self.q_proj(self_query).unsqueeze(1)  # [B, 1, comm_dim]
        # MultiheadAttention key_padding_mask: True = ignore.
        key_padding_mask = neighbor_mask < 0.5  # [B, K]
        # If a row is fully masked, attention would NaN. Fall back to zeros.
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)
        # Avoid all-True rows: temporarily unmask the first slot, gate it later.
        kpm = key_padding_mask.clone()
        kpm[:, 0] = torch.where(all_masked.squeeze(-1), torch.zeros_like(kpm[:, 0]), kpm[:, 0])
        attended, _ = self.attn(q, kv, kv, key_padding_mask=kpm)
        attended = attended.squeeze(1)  # [B, comm_dim]
        # Zero out comm where ALL neighbors absent.
        attended = attended * (~all_masked).float()
        gate = self.gate(attended)
        return attended, gate


# ============================================================================
# Decider Network
# ============================================================================

class DeciderNetwork(nn.Module):
    """The full shared network: encoder + 5 specialists + decider heads."""

    def __init__(
        self,
        obs_dim: int = HierarchicalRoutesObservation.OBS_SIZE,
        action_dim: int = 5,
        temporal_window: int = 3,
        ctx_dim: int = 64,
        comm_dim: int = 32,
        decider_hidden: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.temporal_window = temporal_window
        self.ctx_dim = ctx_dim
        self.comm_dim = comm_dim
        # base_dim used by specialists = DecisionPointObservation.OBS_SIZE = 48
        self.base_dim = DecisionPointObservation.OBS_SIZE

        # ---- LSTM over self obs history ----
        self.self_proj = nn.Linear(obs_dim, ctx_dim)
        self.lstm = nn.LSTM(ctx_dim, ctx_dim, batch_first=True)

        # ---- Comm specialist (operates on the LATEST timestep neighbors) ----
        self.comm = CommSpecialist(obs_dim=obs_dim, query_dim=ctx_dim,
                                   comm_dim=comm_dim, num_heads=4)

        # ---- Specialists ----
        self.routing = RoutingSpecialist(hidden_dim=64, emb_dim=32)
        self.merging = MergingSpecialist(comm_dim=comm_dim, hidden_dim=64, emb_dim=32)
        self.deadlock = DeadlockSpecialist(ctx_dim=ctx_dim, comm_dim=comm_dim,
                                           hidden_dim=64, emb_dim=16)
        self.tree = TreeSpecialist(hidden_dim=48, emb_dim=16)

        # ---- Decider fusion ----
        # Inputs: ctx + routing_emb + merging_emb + deadlock_emb + tree_emb + comm_emb +
        #         scores: route_logits(3) + route_conf(1) + wait(1) + prio(1)
        #                 + p_dl_1(1) + p_dl_3(1) + gate(1) + tree_conf(1) = 10 scalars
        fused_dim = ctx_dim + 32 + 32 + 16 + 16 + comm_dim + 10
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, decider_hidden),
            nn.LayerNorm(decider_hidden),
            nn.GELU(),
            nn.Linear(decider_hidden, decider_hidden),
            nn.GELU(),
        )
        self.actor = nn.Linear(decider_hidden, action_dim)
        self.critic = nn.Linear(decider_hidden, 1)

    # ------------------------------------------------------------------
    def encode_temporal(self, self_seq: torch.Tensor) -> torch.Tensor:
        """self_seq: [B, T, obs_dim] -> ctx [B, ctx_dim]."""
        proj = self.self_proj(self_seq)
        out, _ = self.lstm(proj)
        return out[:, -1, :]  # last timestep ctx

    def forward(
        self,
        self_seq: torch.Tensor,        # [B, T, obs_dim]
        neighbors_now: torch.Tensor,   # [B, K, obs_dim]
        neighbor_mask: torch.Tensor,   # [B, K]
    ):
        ctx = self.encode_temporal(self_seq)
        comm_emb, gate = self.comm(neighbors_now, neighbor_mask, ctx)

        route_emb, route_logits, route_conf = self.routing(self_seq[:, -1, :])
        merge_emb, wait_pres, prio = self.merging(self_seq[:, -1, :], comm_emb)
        dl_emb, p_dl_1, p_dl_3 = self.deadlock(self_seq[:, -1, :], ctx, comm_emb)
        tree_emb, tree_conf = self.tree(self_seq[:, -1, :])

        scalars = torch.cat(
            [
                torch.softmax(route_logits, dim=-1),  # 3
                route_conf,                            # 1
                wait_pres,                             # 1
                prio,                                  # 1
                p_dl_1,                                # 1
                p_dl_3,                                # 1
                gate,                                  # 1
                tree_conf,                             # 1
            ],
            dim=-1,
        )
        fused = torch.cat([ctx, route_emb, merge_emb, dl_emb, tree_emb, comm_emb, scalars], dim=-1)
        h = self.fuse(fused)
        action_logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        aux = {
            "p_dl_1": p_dl_1.squeeze(-1),
            "p_dl_3": p_dl_3.squeeze(-1),
            "gate": gate.squeeze(-1),
            "route_logits": route_logits,
            "route_conf": route_conf.squeeze(-1),
            "wait_pres": wait_pres.squeeze(-1),
            "prio": prio.squeeze(-1),
            "tree_conf": tree_conf.squeeze(-1),
        }
        return action_logits, value, aux


# ============================================================================
# Helpers: convert temporal_state list -> tensors
# ============================================================================

def _state_to_tensors(temporal_state, obs_dim: int, device: torch.device):
    """temporal_state = [(obs_t-2, [opp_vecs]), ..., (obs_t, [opp_vecs])]

    Returns:
        self_seq:        [1, T, obs_dim]
        neighbors_now:   [1, K, obs_dim]   (latest timestep only)
        neighbor_mask:   [1, K]
    """
    T = len(temporal_state)
    self_arr = np.zeros((T, obs_dim), dtype=np.float32)
    for t, (obs_self, _opps) in enumerate(temporal_state):
        v = np.asarray(obs_self, dtype=np.float32).reshape(-1)
        cap = min(v.shape[0], obs_dim)
        self_arr[t, :cap] = v[:cap]

    # Neighbors only from the LAST timestep (current step's view).
    _last_self, last_opps = temporal_state[-1]
    K = NEIGHBOR_K
    neigh_arr = np.zeros((K, obs_dim), dtype=np.float32)
    mask_arr = np.zeros(K, dtype=np.float32)
    if last_opps is not None:
        for k_idx, opp_vec in enumerate(last_opps[:K]):
            v = np.asarray(opp_vec, dtype=np.float32).reshape(-1)
            cap = min(v.shape[0], obs_dim)
            neigh_arr[k_idx, :cap] = v[:cap]
            mask_arr[k_idx] = 1.0

    self_seq = torch.from_numpy(self_arr).unsqueeze(0).to(device)
    neighbors_now = torch.from_numpy(neigh_arr).unsqueeze(0).to(device)
    neighbor_mask = torch.from_numpy(mask_arr).unsqueeze(0).to(device)
    return self_seq, neighbors_now, neighbor_mask


def _states_batch_to_tensors(states_list, obs_dim: int, device: torch.device):
    """Batch convert. states_list[i] is one temporal_state."""
    B = len(states_list)
    T = len(states_list[0]) if B > 0 else 0
    self_arr = np.zeros((B, T, obs_dim), dtype=np.float32)
    K = NEIGHBOR_K
    neigh_arr = np.zeros((B, K, obs_dim), dtype=np.float32)
    mask_arr = np.zeros((B, K), dtype=np.float32)
    for i, ts in enumerate(states_list):
        for t, (obs_self, _opps) in enumerate(ts):
            v = np.asarray(obs_self, dtype=np.float32).reshape(-1)
            cap = min(v.shape[0], obs_dim)
            self_arr[i, t, :cap] = v[:cap]
        _last_self, last_opps = ts[-1]
        if last_opps is not None:
            for k_idx, opp_vec in enumerate(last_opps[:K]):
                v = np.asarray(opp_vec, dtype=np.float32).reshape(-1)
                cap = min(v.shape[0], obs_dim)
                neigh_arr[i, k_idx, :cap] = v[:cap]
                mask_arr[i, k_idx] = 1.0
    return (
        torch.from_numpy(self_arr).to(device),
        torch.from_numpy(neigh_arr).to(device),
        torch.from_numpy(mask_arr).to(device),
    )


# ============================================================================
# DeciderPolicy (LearningPolicy)
# ============================================================================

class DeciderPPOPolicy(LearningPolicy):
    """ONE shared network for all train agents.

    PPO + 1-step deadlock BCE aux-loss. Action mask read from env.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1.0e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.12,
        k_epochs: int = 2,
        batch_size: int = 256,
        max_episodes_in_memory: int = 20,
        weight_entropy: float = 0.05,
        weight_value: float = 0.5,
        weight_aux_dl: float = 0.05,
        temporal_window: int = 3,
        train_frequency: int = 10,
        reward_scale: float = 0.01,
        aux_pos_weight: float = 4.0,
        target_kl: float = 0.07,
        max_eps_random: float = 0.0,
        clear_buffer_after_update: bool = True,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.weight_entropy = weight_entropy
        self.weight_value = weight_value
        self.weight_aux_dl = weight_aux_dl
        self.train_frequency = train_frequency
        self.temporal_window = temporal_window
        self.reward_scale = float(reward_scale)
        self.aux_pos_weight = float(aux_pos_weight)
        self.target_kl = float(target_kl)
        self.max_eps_random = float(max_eps_random)
        self.clear_buffer_after_update = bool(clear_buffer_after_update)
        self.max_episodes_in_memory = int(max_episodes_in_memory)

        self.device = torch.device("cpu")
        self.net = DeciderNetwork(
            obs_dim=state_size,
            action_dim=action_size,
            temporal_window=temporal_window,
        ).to(self.device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=learning_rate)

        self.current_episode_memory = EpisodeBuffers()
        self.accumulated_episodes: deque = deque(maxlen=max_episodes_in_memory)

        self._env: Union[Environment, None] = None
        self.episode_count = 0
        self._last_aux_label: dict = {}  # handle -> 1-step deadlock label
        self.loss = 0.0

        # ------------------------------------------------------------------
        # Startup banner: parameter counts per submodule + hyperparams
        # ------------------------------------------------------------------
        def _np(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        p_temporal = _np(self.net.self_proj) + _np(self.net.lstm)
        p_routing  = _np(self.net.routing)
        p_merging  = _np(self.net.merging)
        p_deadlock = _np(self.net.deadlock)
        p_tree     = _np(self.net.tree)
        p_comm     = _np(self.net.comm)
        p_fuse     = _np(self.net.fuse)
        p_actor    = _np(self.net.actor)
        p_critic   = _np(self.net.critic)
        p_total    = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        bar = "=" * 78
        sep = "-" * 78
        print("\n" + bar)
        print("🚂 DeciderPPOPolicy — Hierarchical Decider PPO (parameter sharing)")
        print(bar)
        print(f"  obs_dim={state_size}   action_dim={action_size}   "
              f"K={NEIGHBOR_K}   T={temporal_window}")
        print(f"  ctx_dim={self.net.ctx_dim}   comm_dim={self.net.comm_dim}   "
              f"base_dim={self.net.base_dim}")
        print(sep)
        print("  Submodule                          Params      Outputs")
        print(sep)
        print(f"  Temporal encoder (Linear+LSTM)  {p_temporal:>10,d}      ctx [{self.net.ctx_dim}]")
        print(f"  Routing specialist              {p_routing:>10,d}      route_emb[32] + 3 logits + conf")
        print(f"  Merging specialist              {p_merging:>10,d}      merge_emb[32] + wait + prio")
        print(f"  Deadlock specialist             {p_deadlock:>10,d}      dl_emb[16] + p_dl_1 + p_dl_3")
        print(f"  Tree specialist                 {p_tree:>10,d}      tree_emb[16] + tree_conf")
        print(f"  Comm specialist (K={NEIGHBOR_K}, heads=4)  {p_comm:>10,d}      comm_emb[{self.net.comm_dim}] + gate (sparse attn)")
        print(f"  Decider fuse MLP                {p_fuse:>10,d}      h[128]")
        print(f"  Actor head                      {p_actor:>10,d}      action_logits[{action_size}]")
        print(f"  Critic head                     {p_critic:>10,d}      value[1]")
        print(sep)
        print(f"  TOTAL TRAINABLE                 {p_total:>10,d}      (~{p_total/1e3:.1f}K)")
        print(sep)
        print(f"  optimizer=AdamW    lr={learning_rate:.2e}    gamma={gamma}    gae_lambda={gae_lambda}")
        print(f"  clip_eps={clip_eps}    k_epochs={k_epochs}    batch_size={batch_size}    train_freq={train_frequency}")
        print(f"  w_value={weight_value}    w_entropy={weight_entropy}    w_aux_dl={weight_aux_dl}")
        print(f"  reward_scale={self.reward_scale}    aux_pos_weight={self.aux_pos_weight}    "
              f"max_episodes_in_memory={max_episodes_in_memory}")
        print(f"  target_kl={self.target_kl}    max_eps_random={self.max_eps_random}")
        print(f"  clear_buffer_after_update={self.clear_buffer_after_update}")
        print(bar + "\n")

    # ------------------------------------------------------------------
    def get_name(self) -> str:
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Solver-API hooks
    # ------------------------------------------------------------------
    def reset(self, env: Environment):
        self._env = env

    def start_episode(self, train: bool):
        self.current_episode_memory = EpisodeBuffers()
        self._last_aux_label = {}

    def start_step(self, train: bool):
        pass

    def end_step(self, train: bool):
        pass

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def _legal_action_mask(self, handle: int) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[RailEnvActions.DO_NOTHING] = 1.0
        mask[RailEnvActions.STOP_MOVING] = 1.0
        if self._env is None:
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
            return mask
        agent = self._env.raw_env.agents[handle]
        if not agent.state.is_on_map_state():
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
            return mask
        pos = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        if pos is None or direction is None:
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
            return mask
        transitions = self._env.raw_env.rail.get_transitions(*pos, direction)
        rel_to_action = {
            direction: RailEnvActions.MOVE_FORWARD,
            (direction - 1) % 4: RailEnvActions.MOVE_LEFT,
            (direction + 1) % 4: RailEnvActions.MOVE_RIGHT,
        }
        any_movement_legal = False
        for new_dir in range(4):
            if transitions[new_dir] and new_dir in rel_to_action:
                mask[rel_to_action[new_dir]] = 1.0
                any_movement_legal = True
        # Dead-end -> Flatland forces MOVE_FORWARD as "back".
        if (
            fast_count_nonzero(transitions) == 1
            and transitions[(direction + 2) % 4] == 1
        ):
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
            any_movement_legal = True
        # Force commitment: when the agent CAN move, disallow BOTH passive
        # actions (DO_NOTHING and STOP_MOVING). Otherwise the policy collapses
        # on STOP_MOVING (entropy → log(2)≈0.69, done → 0) because waiting is
        # a low-variance local minimum: no progress, but also no deadlock
        # penalty if peers happen to clear the conflict. NeurIPS winners
        # (Mohanty 2020, JBR_HSE 2020) all used a movement-only action set.
        # Conflict avoidance must be learned through ROUTING (LEFT/RIGHT)
        # not through standstill.
        if any_movement_legal:
            mask[RailEnvActions.DO_NOTHING] = 0.0
            mask[RailEnvActions.STOP_MOVING] = 0.0
        return mask

    def act(self, handle: int, state, eps: float = 0.0) -> int:
        legal = self._legal_action_mask(handle)
        # Epsilon-greedy: keep exploration movement-oriented, but do not
        # over-bias it toward MOVE_FORWARD. The previous forward preference
        # caused the policy to under-explore lateral conflict resolution.
        eps_eff = min(float(eps) if eps is not None else 0.0, self.max_eps_random)
        if eps_eff > 0.0 and np.random.rand() < eps_eff:
            move_candidates = []
            for a in (RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT):
                if legal[a] > 0.5:
                    move_candidates.append(int(a))
            if len(move_candidates) > 0:
                if int(RailEnvActions.MOVE_FORWARD) in move_candidates and np.random.rand() < 0.35:
                    return int(RailEnvActions.MOVE_FORWARD)
                return int(np.random.choice(move_candidates))
            legal_idx = np.where(legal > 0.5)[0]
            if len(legal_idx) > 0:
                return int(np.random.choice(legal_idx))
        with torch.no_grad():
            self_seq, neigh, mask = _state_to_tensors(state, self.state_size, self.device)
            logits, _value, _aux = self.net(self_seq, neigh, mask)
            logits = logits.squeeze(0)
            legal_t = torch.from_numpy(legal).to(self.device)
            logits = logits.masked_fill(legal_t < 0.5, -1e9)

            # Gentle anti-forward bias: when lateral movement is legal, nudge
            # the policy to consider it instead of collapsing into straight
            # motion on every decision point.
            if (
                legal[RailEnvActions.MOVE_LEFT] > 0.5
                or legal[RailEnvActions.MOVE_RIGHT] > 0.5
            ):
                forward_penalty = float(getattr(self, 'forward_logit_penalty', 0.12))
                logits[RailEnvActions.MOVE_FORWARD] -= forward_penalty

            dist = Categorical(logits=logits)
            action = dist.sample().item()
        return int(action)

    # ------------------------------------------------------------------
    def step(self, handle, state, action, reward, next_state, done):
        # Aux label for current transition: 1-step deadlock signal at next_state.
        # next_state is a temporal sequence; the "current" deadlock is the
        # local_deadlock flag at index 42 of the LAST observation in the seq.
        try:
            last_obs_next = next_state[-1][0]
            dl_next = float(np.asarray(last_obs_next, dtype=np.float32).reshape(-1)[42])
        except Exception:
            dl_next = 0.0
        # Snapshot the legal-action mask at decision-time. Storing this in the
        # replay buffer is essential: act() samples from a MASKED logit dist,
        # so _update() must apply the SAME mask to compute logp_old/logp_new
        # consistently. Without this, ratio = exp(unmasked_logp - masked_logp)
        # becomes biased on transitions where some actions were masked out.
        action_mask = self._legal_action_mask(handle)
        transition = (
            state, int(action), float(reward) * self.reward_scale,
            next_state, bool(done), float(dl_next),
            action_mask.astype(np.float32),
        )
        self.current_episode_memory.push_transition(handle, transition)

    # ------------------------------------------------------------------
    def end_episode(self, train: bool):
        self.episode_count += 1
        if not train:
            return
        # Push current episode's per-handle transitions into accumulator.
        self.accumulated_episodes.append(self.current_episode_memory)
        if (self.episode_count % max(1, self.train_frequency)) != 0:
            return
        self._update()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def _flatten_episodes(self):
        """Return one flat list of transitions across all accumulated episodes
        and all handles, with per-trajectory boundaries respected by GAE."""
        trajectories = []  # list of list[transition]
        for ep_buf in self.accumulated_episodes:
            for h in list(ep_buf.memory.keys()):
                trans = ep_buf.get_transitions(h)
                if len(trans) > 0:
                    trajectories.append(trans)
        return trajectories

    def _compute_returns_advantages(self, rewards, values, dones, last_value):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else values[t + 1]
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_v * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            adv[t] = last_gae
        returns = adv + np.asarray(values, dtype=np.float32)
        return returns, adv

    def _update(self):
        t_update_start = perf_counter()
        trajectories = self._flatten_episodes()
        if len(trajectories) == 0:
            return

        # 1) Forward all states once to get values/logp_old/aux for advantages.
        all_states, all_actions, all_rewards, all_dones = [], [], [], []
        all_aux_labels = []
        all_masks = []
        traj_lens = []
        for traj in trajectories:
            traj_lens.append(len(traj))
            for trans in traj:
                # Backward-compat: old buffers may have 6-tuples (no mask).
                if len(trans) == 7:
                    s, a, r, ns, d, dl_next, mask = trans
                else:
                    s, a, r, ns, d, dl_next = trans
                    mask = np.ones(self.action_size, dtype=np.float32)
                all_states.append(s)
                all_actions.append(a)
                all_rewards.append(r)
                all_dones.append(d)
                all_aux_labels.append(dl_next)
                all_masks.append(mask)

        if len(all_states) == 0:
            return

        # Buffer/data diagnostics for update logs.
        episodes_in_buffer = len(self.accumulated_episodes)
        transitions_in_buffer = int(sum(len(t) for t in trajectories))
        avg_traj_len = float(np.mean(traj_lens)) if len(traj_lens) > 0 else 0.0
        max_traj_len = int(np.max(traj_lens)) if len(traj_lens) > 0 else 0

        self_seq, neigh, nmask = _states_batch_to_tensors(
            all_states, self.state_size, self.device
        )
        actions_t = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        rewards_arr = np.asarray(all_rewards, dtype=np.float32)
        dones_arr = np.asarray(all_dones, dtype=np.float32)
        aux_labels_t = torch.tensor(all_aux_labels, dtype=torch.float32, device=self.device)
        masks_t = torch.tensor(
            np.stack(all_masks, axis=0), dtype=torch.float32, device=self.device
        )  # [N, action_size], 1.0=legal, 0.0=illegal

        with torch.no_grad():
            old_logits, old_values, _aux = self.net(self_seq, neigh, nmask)
            # Apply legal-action mask BEFORE computing log_prob so old_logp
            # matches what act() actually sampled from.
            old_logits_masked = old_logits.masked_fill(masks_t < 0.5, -1e9)
            old_dist = Categorical(logits=old_logits_masked)
            old_logp = old_dist.log_prob(actions_t).cpu().numpy()
            old_values_np = old_values.cpu().numpy()

        # 2) Per-trajectory GAE.
        # Bootstrap rule: at trajectory end, V(s_T) = 0. This treats
        # truncated episodes as terminal, avoiding the common bug of
        # using V(s_{T-1}) which yields delta = r + (gamma-1)*V instead of
        # r + gamma*V(s_T) - V(s_{T-1}). For Flatland episodes that end via
        # max_step truncation this is a slightly pessimistic but unbiased
        # bootstrap (Schulman et al. 2017, App. A; OpenAI baselines PPO).
        returns_all = np.zeros_like(rewards_arr)
        adv_all = np.zeros_like(rewards_arr)
        cur = 0
        for tl in traj_lens:
            r_seg = rewards_arr[cur : cur + tl]
            v_seg = old_values_np[cur : cur + tl]
            d_seg = dones_arr[cur : cur + tl]
            last_v = 0.0
            ret_seg, adv_seg = self._compute_returns_advantages(r_seg, v_seg, d_seg, last_v)
            returns_all[cur : cur + tl] = ret_seg
            adv_all[cur : cur + tl] = adv_seg
            cur += tl

        # Normalise advantages per-minibatch later; here just clip outliers
        # that cause gradient spikes (> 5 sigma from mean).
        if adv_all.std() > 1e-6:
            adv_mean = adv_all.mean()
            adv_std  = adv_all.std() + 1e-8
            adv_all  = np.clip(adv_all, adv_mean - 5 * adv_std, adv_mean + 5 * adv_std)
            adv_all  = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
        returns_t = torch.tensor(returns_all, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv_all, dtype=torch.float32, device=self.device)
        old_logp_t = torch.tensor(old_logp, dtype=torch.float32, device=self.device)

        # 3) PPO epochs with mini-batches with live progress bar.
        N = self_seq.shape[0]
        idx = np.arange(N)
        bs = min(self.batch_size, N)
        num_batches = max(1, (N + bs - 1) // bs)
        total_iters = self.k_epochs * num_batches
        cur_iter = 0
        epoch_loss = 0.0
        n_minibatches = 0
        kl_early_stop = False

        print(
            f"\n[Decider] ep={self.episode_count:>5d}  upd_samples={N:>5d}  "
            f"batches={num_batches}×epochs={self.k_epochs}={total_iters}  "
            f"buffer_eps={episodes_in_buffer:>2d}/{self.max_episodes_in_memory}  "
            f"traj={len(trajectories):>3d}  avg_len={avg_traj_len:5.1f}  max_len={max_traj_len:>3d}"
        )

        for k_loop in range(self.k_epochs):
            if kl_early_stop:
                break
            np.random.shuffle(idx)
            for b_idx, start in enumerate(range(0, N, bs)):
                mb = idx[start : start + bs]
                mb_t = torch.tensor(mb, dtype=torch.long, device=self.device)
                logits, values, aux = self.net(
                    self_seq.index_select(0, mb_t),
                    neigh.index_select(0, mb_t),
                    nmask.index_select(0, mb_t),
                )
                # Apply the SAME action mask used at decision-time so the
                # entropy and log-probability are computed over the legal
                # action support only. This both fixes the policy-mismatch
                # bias and prevents entropy from being inflated by mass on
                # impossible actions (which the masked behavioral policy
                # never selected).
                mb_mask = masks_t.index_select(0, mb_t)
                logits_masked = logits.masked_fill(mb_mask < 0.5, -1e9)
                dist = Categorical(logits=logits_masked)
                new_logp = dist.log_prob(actions_t.index_select(0, mb_t))
                old_logp_mb = old_logp_t.index_select(0, mb_t)
                # Clip log-ratio to ±5 BEFORE exp to keep updates finite even
                # under rare near-impossible old actions. PPO clip_eps still
                # does the trust-region bounding; this is purely an overflow
                # guard. exp(±5) = [0.0067, 148] which is far beyond clip_eps.
                log_ratio = torch.clamp(new_logp - old_logp_mb, -5.0, 5.0)
                ratio = torch.exp(log_ratio)
                a_mb = adv_t.index_select(0, mb_t)
                surr1 = ratio * a_mb
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * a_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                # Clipped + Huber value loss (Schulman PPO §6, "PPO2 trick").
                # Robust against reward-scale shifts across curriculum phases:
                # the critic update is bounded to ±clip_eps around the old
                # value prediction, preventing critic over-fit to rare
                # DONE_BONUS spikes when more agents complete simultaneously.
                value_pred   = values
                value_target = returns_t.index_select(0, mb_t)
                old_value_mb = torch.tensor(
                    old_values_np[mb], dtype=torch.float32, device=self.device
                )
                v_clipped = old_value_mb + torch.clamp(
                    value_pred - old_value_mb, -self.clip_eps, self.clip_eps
                )
                v_loss_unclipped = nn.functional.smooth_l1_loss(
                    value_pred, value_target, reduction='none'
                )
                v_loss_clipped = nn.functional.smooth_l1_loss(
                    v_clipped, value_target, reduction='none'
                )
                value_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                entropy = dist.entropy().mean()

                # Aux: 1-step deadlock BCE with positive class up-weighting
                # (label==1 is rare, so vanilla BCE collapses to predicting 0).
                p_dl_clipped = aux["p_dl_1"].clamp(1e-6, 1 - 1e-6)
                lbl = aux_labels_t.index_select(0, mb_t).clamp(0.0, 1.0)
                pos_w = torch.full_like(lbl, self.aux_pos_weight)
                bce = -(
                    pos_w * lbl * torch.log(p_dl_clipped)
                    + (1.0 - lbl) * torch.log(1.0 - p_dl_clipped)
                )
                aux_dl_loss = bce.mean()

                loss = (
                    policy_loss
                    + self.weight_value * value_loss
                    - self.weight_entropy * entropy
                    + self.weight_aux_dl * aux_dl_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                # Tight grad clip to counter gradient explosions seen at
                # late training (g_norm 6-9 → entropy collapse). 0.5 keeps
                # individual specialist branches from dominating the update.
                grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()

                epoch_loss += float(loss.item())
                n_minibatches += 1

                # ---- live per-batch progress line ------------------------
                cur_iter += 1
                bar_len = 30
                prog = cur_iter / max(1, total_iters)
                filled = int(bar_len * prog)
                pbar = "█" * filled + "░" * (bar_len - filled)
                approx_kl = float((old_logp_mb - new_logp.detach()).mean().abs().item())
                pos_frac = float(lbl.mean().item())
                p_dl_mean = float(p_dl_clipped.detach().mean().item())
                gn = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
                print(
                    f"\r  [{pbar}] E{k_loop+1}/{self.k_epochs} "
                    f"B{b_idx+1}/{num_batches} ({prog*100:5.1f}%) "
                    f"| L={float(loss.item()):+7.3f} "
                    f"| P={float(policy_loss.item()):+6.3f} "
                    f"| V={float(value_loss.item()):6.3f} "
                    f"| Ent={float(entropy.item()):.3f} "
                    f"| Aux={float(aux_dl_loss.item()):.3f} "
                    f"| ratio={float(ratio.mean().item()):.3f} "
                    f"| KL={approx_kl:.4f} "
                    f"| g_norm={gn:5.2f} "
                    f"| dl(pos={pos_frac:.2f},pred={p_dl_mean:.2f})",
                    end="",
                    flush=True,
                )

                # PPO early-stop on KL divergence spike (Schulman et al.).
                if approx_kl > self.target_kl:
                    kl_early_stop = True
                    print(
                        f"\n  ⚠️ KL early-stop: KL={approx_kl:.4f} > target_kl={self.target_kl:.4f} "
                        f"(epoch {k_loop + 1}, batch {b_idx + 1})"
                    )
                    break
        # newline after the in-place progress bar finishes
        print("")

        if n_minibatches > 0:
            self.loss = epoch_loss / n_minibatches
        update_sec = max(1e-9, perf_counter() - t_update_start)
        samp_per_sec = float(N) / update_sec
        # Final summary line per update.
        try:
            print(
                f"  → ep={self.episode_count:>5d}  N={N:>5d}  "
                f"avg_loss={self.loss:+.4f}  policy={float(policy_loss.item()):+.4f}  "
                f"value={float(value_loss.item()):.4f}  ent={float(entropy.item()):.3f}  "
                f"aux_dl={float(aux_dl_loss.item()):.3f}  ratio={float(ratio.mean().item()):.3f}  "
                f"buffer_trans={transitions_in_buffer}  t={update_sec:.2f}s  "
                f"throughput={samp_per_sec:.0f} samp/s  "
                f"kl_stop={int(kl_early_stop)}\n"
            )
        except Exception:
            pass

        if self.clear_buffer_after_update:
            self.accumulated_episodes.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filename: str):
        torch.save(self.net.state_dict(), filename)

    def load(self, filename: str):
        try:
            self.net.load_state_dict(torch.load(filename, map_location=self.device))
        except Exception as exc:
            print(f"[DeciderPPOPolicy] load failed: {exc}")

    def clone(self):
        return self

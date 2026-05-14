import copy
import math
import os
from collections import namedtuple, deque
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence

from policy.learning_policy.learning_policy import LearningPolicy


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

        # 🎯 CRITICAL: Wenn letzter Eintrag bereits done==True, keine neue Transition!
        # Pro Agent: maximal EINE done==True transition (die letzte)
        if len(transitions) > 0:
            last_transition = transitions[-1]
            # Struktur: (state, action, reward, next_state, done, aux_deadlock)
            done_flag = last_transition[4]
            if done_flag:
                # Agent ist fertig - ignoriere neue Steps
                return

        transitions.append(transition)
        self.memory.update({handle: transitions})


# =============================================================================
# LOCAL TREE ENCODER (LSTM über DFS-Sequenz — topologie-sensitiv)
# -----------------------------------------------------------------------------
# Encodiert den lokalen Suchbaum aus DecisionPointObservation._local_search()
# als geordnete Sequenz von Knoten in DFS-Pre-Order.
#
# Node features (8D per node, serialized in obs[35:155], 15 nodes × 8D = 120D):
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

    Input:  (MAX_NODES * NODE_DIM,) = (120,) DFS-ordered flat sequence from obs[35:155]
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
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tree_flat: torch.Tensor) -> torch.Tensor:
        """Single-agent: (MAX_NODES * NODE_DIM,) = (120,) → (hidden_dim,)"""
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


# Constants shared across encoder classes
_BASE_OBS_DIM = 35   # Handcrafted features [0-34] from DecisionPointObservation
_TREE_BLOCK_DIM = LocalTreeEncoder.MAX_NODES * LocalTreeEncoder.NODE_DIM  # = 120


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
    1. Observation Encoder: Maps 35D (Modus A Pure) or 94D (HierarchicalRoutes) obs → 128D embedding
    2. TEMPORAL Attention: Links t-2, t-1, t → learns movement patterns
    3. SPATIAL Attention: Links self + opponents → learns interactions
    4. Output Projection: Final 128D context embedding
    
    Advantages over simple MAEncoderWithAttention:
    ✅ Sees movement (not just snapshot)
    ✅ Predicts collisions (velocity awareness)
    ✅ Smoother trajectories (temporal context)
    """
    
    def __init__(self, 
                 obs_dim: int,           # 35D DecisionPoint Modus A Pure or 94D HierarchicalRoutes
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
        # obs_encoder processes only the BASE_OBS_DIM (35D) handcrafted features.
        # Tree node features obs[35:155] are handled separately by tree_encoder.
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

        # LocalTreeEncoder: DeepSets-style encoder for tree nodes from obs[35:155].
        # Replaces old slot-based tree_proj (which referenced non-existent indices).
        self.tree_encoder = LocalTreeEncoder(hidden_dim)
        self.tree_norm = nn.LayerNorm(hidden_dim)

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
        # tree_encoder is initialized with near-zero weights inside LocalTreeEncoder.__init__()
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
            print(f"⚠️ WARNING: Input contains NaN/Inf, replacing with 0")
            t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # ⚠️ PROTECTION: Clamp extreme values
        t = torch.clamp(t, min=-10.0, max=10.0)

        return t

    def _encode_tree_block(self, obs_1d: torch.Tensor) -> torch.Tensor:
        """Extract obs[35:155] tree node block and encode via LocalTreeEncoder."""
        end = _BASE_OBS_DIM + _TREE_BLOCK_DIM
        if obs_1d.shape[0] > _BASE_OBS_DIM:
            tree_flat = obs_1d[_BASE_OBS_DIM:min(end, obs_1d.shape[0])]
            if tree_flat.shape[0] < _TREE_BLOCK_DIM:
                pad = torch.zeros(_TREE_BLOCK_DIM - tree_flat.shape[0], device=self.device)
                tree_flat = torch.cat([tree_flat, pad], dim=0)
        else:
            tree_flat = torch.zeros(_TREE_BLOCK_DIM, device=self.device)
        return self.tree_encoder(tree_flat)

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
                         Each obs is a 35D or 94D numpy array
            handle: Agent handle (for debugging)
        
        Returns:
            context_embedding: (hidden_dim,) - Temporal + Spatial Context
        """
        # ========================================================================
        # STEP 1: Encode all observations across time
        # ========================================================================
        
        # Extract self observations over time
        self_obs_sequence = []  # Will be [emb_t-2, emb_t-1, emb_t]
        for obs_self, _ in temporal_seq:
            obs_t = self._to_1d_tensor(obs_self)
            base_obs = obs_t[:self.base_obs_dim]  # First 35D handcrafted features only
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

        # LocalTreeEncoder: encode node features from obs[35:155] at timestep t.
        last_obs_t = self._to_1d_tensor(temporal_seq[-1][0])
        tree_emb = self._encode_tree_block(last_obs_t)
        self_temporal_context = self.tree_norm(self_temporal_context + tree_emb)
        
        # ========================================================================
        # STEP 3: SPATIAL ATTENTION - Multi-Agent Interaction
        # ========================================================================
        
        # Get current opponent observations (only from t, not entire history)
        _, current_opponents = temporal_seq[-1]  # Last timestep
        
        # Encode opponents (current timestep only, base 35D features)
        opp_embeddings = []
        for opp_obs in current_opponents:
            opp_t = self._to_1d_tensor(opp_obs)
            opp_base = opp_t[:self.base_obs_dim]  # First 35D only
            opp_emb = self.obs_encoder(opp_base)
            opp_embeddings.append(opp_emb)
        
        # Combine self + opponents
        if len(opp_embeddings) > 0:
            all_agents = [self_temporal_context] + opp_embeddings
            all_agents_tensor = torch.stack(all_agents, dim=0)  # (N_agents, 128)
            all_agents_batched = all_agents_tensor.unsqueeze(0)  # (1, N_agents, 128)
            
            # Query = self (only own representation)
            query = self_temporal_context.unsqueeze(0).unsqueeze(0)  # (1, 1, 128)
            
            # Spatial Attention: Self attends to all (self + opponents)
            spatial_output, _ = self.spatial_attention(
                query=query,
                key=all_agents_batched,
                value=all_agents_batched
            )
            
            context = spatial_output.squeeze(0).squeeze(0)  # (128,)
            
            # Residual connection
            context = context + self_temporal_context
            context, comm_reg, gate_mean, intent_mean = self._apply_communication(context, opp_embeddings)
            self.last_comm_reg = comm_reg
            self.last_comm_gate_mean = float(gate_mean.detach().cpu().item())
            self.last_comm_intent_mean = [float(x) for x in intent_mean.detach().cpu().tolist()]
            self.last_comm_valid_count = 1
        else:
            # No opponents → only own temporal context
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
        
        for temp_seq in temporal_sequences:
            self_seq = [obs_self for obs_self, _ in temp_seq]
            all_self_obs.append(torch.stack([self._to_1d_tensor(obs) for obs in self_seq]))
            _, current_opps = temp_seq[-1]
            all_opponents.append(current_opps)
        
        # Stack: (batch_size, temporal_window, 33)
        all_self_obs_tensor = torch.stack(all_self_obs, dim=0)
        
        # Encode base obs features only (first 35D); tree block encoded separately
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

        # LocalTreeEncoder (batch): encode tree block from obs[35:155] at timestep t.
        last_obs_b = all_self_obs_tensor[:, -1, :]  # (B, obs_dim)
        if last_obs_b.shape[1] > _BASE_OBS_DIM:
            tree_flat_b = last_obs_b[:, _BASE_OBS_DIM:_BASE_OBS_DIM + _TREE_BLOCK_DIM]
            if tree_flat_b.shape[1] < _TREE_BLOCK_DIM:
                pad = torch.zeros(batch_size, _TREE_BLOCK_DIM - tree_flat_b.shape[1], device=self.device)
                tree_flat_b = torch.cat([tree_flat_b, pad], dim=1)
        else:
            tree_flat_b = torch.zeros(batch_size, _TREE_BLOCK_DIM, device=self.device)
        tree_emb_b = self.tree_encoder.forward_batch(tree_flat_b)
        self_temporal_contexts = self.tree_norm(self_temporal_contexts + tree_emb_b)
        
        # Spatial attention (process per agent due to varying opponent counts)
        final_embeddings = []
        comm_regs = []
        comm_gate_means = []
        comm_intents = []
        for i in range(batch_size):
            self_ctx = self_temporal_contexts[i]
            opps = all_opponents[i]
            
            if len(opps) > 0:
                opp_embs = []
                for opp_obs in opps:
                    opp_t = self._to_1d_tensor(opp_obs)
                    opp_base = opp_t[:self.base_obs_dim]  # First 35D only
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
            self.load_state_dict(torch.load(state_file, map_location=self.device))


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

        # obs_encoder processes only the BASE_OBS_DIM (35D) handcrafted features.
        # Tree node features obs[35:155] are handled by tree_encoder (LocalTreeEncoder).
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

        # LocalTreeEncoder: DeepSets-style encoder for tree nodes from obs[35:155].
        self._tree_slots = None  # Unused; kept for checkpoint backward-compatibility stub.
        self.tree_encoder = LocalTreeEncoder(hidden_dim)
        self.tree_norm = nn.LayerNorm(hidden_dim)

        self.last_comm_reg = torch.tensor(0.0, device=self.device)
        self.last_comm_gate_mean = 0.0
        self.last_comm_intent_mean = [0.0, 0.0, 0.0]
        self.last_comm_valid_count = 0

        self._init_weights()
        # tree_encoder is initialized with near-zero weights inside LocalTreeEncoder.__init__()
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

    def _encode_tree_block(self, obs_1d: torch.Tensor) -> torch.Tensor:
        """Extract obs[35:155] tree node block and encode via LocalTreeEncoder."""
        end = _BASE_OBS_DIM + _TREE_BLOCK_DIM
        if obs_1d.shape[0] > _BASE_OBS_DIM:
            tree_flat = obs_1d[_BASE_OBS_DIM:min(end, obs_1d.shape[0])]
            if tree_flat.shape[0] < _TREE_BLOCK_DIM:
                pad = torch.zeros(_TREE_BLOCK_DIM - tree_flat.shape[0], device=self.device)
                tree_flat = torch.cat([tree_flat, pad], dim=0)
        else:
            tree_flat = torch.zeros(_TREE_BLOCK_DIM, device=self.device)
        return self.tree_encoder(tree_flat)

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
        for obs_self, _ in temporal_seq:
            obs_t = self._to_1d_tensor(obs_self)
            base_obs = obs_t[:self.base_obs_dim]  # First 35D only
            emb_t = self.obs_encoder(base_obs)
            self_obs_sequence.append(emb_t)

        self_seq_tensor = torch.stack(self_obs_sequence, dim=0).unsqueeze(0)
        temporal_output, _ = self.temporal_lstm(self_seq_tensor)
        self_temporal_context = temporal_output[0, -1, :]

        # LocalTreeEncoder: encode node features from obs[35:155] at timestep t.
        last_obs_t = self._to_1d_tensor(temporal_seq[-1][0])
        tree_emb = self._encode_tree_block(last_obs_t)
        self_temporal_context = self.tree_norm(self_temporal_context + tree_emb)

        _, current_opponents = temporal_seq[-1]
        opp_embeddings = []
        for opp_obs in current_opponents:
            opp_t = self._to_1d_tensor(opp_obs)
            opp_base = opp_t[:self.base_obs_dim]  # First 35D only
            opp_embeddings.append(self.obs_encoder(opp_base))

        if len(opp_embeddings) > 0:
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

        for temp_seq in temporal_sequences:
            self_seq = [obs_self for obs_self, _ in temp_seq]
            all_self_obs.append(torch.stack([self._to_1d_tensor(obs) for obs in self_seq]))
            _, current_opps = temp_seq[-1]
            all_opponents.append(current_opps)

        all_self_obs_tensor = torch.stack(all_self_obs, dim=0)
        flat_obs = all_self_obs_tensor.view(-1, all_self_obs_tensor.shape[-1])
        flat_base_obs = flat_obs[:, :self.base_obs_dim]  # (B*T, 35)
        flat_embeddings = self.obs_encoder(flat_base_obs)  # (B*T, H)
        self_embeddings = flat_embeddings.view(batch_size, self.temporal_window, self.hidden_dim)

        temporal_output, _ = self.temporal_lstm(self_embeddings)
        self_temporal_contexts = temporal_output[:, -1, :]

        # LocalTreeEncoder (batch): encode tree block from obs[35:155] at timestep t.
        last_obs_b = all_self_obs_tensor[:, -1, :]  # (B, obs_dim)
        if last_obs_b.shape[1] > _BASE_OBS_DIM:
            tree_flat_b = last_obs_b[:, _BASE_OBS_DIM:_BASE_OBS_DIM + _TREE_BLOCK_DIM]
            if tree_flat_b.shape[1] < _TREE_BLOCK_DIM:
                pad = torch.zeros(batch_size, _TREE_BLOCK_DIM - tree_flat_b.shape[1], device=self.device)
                tree_flat_b = torch.cat([tree_flat_b, pad], dim=1)
        else:
            tree_flat_b = torch.zeros(batch_size, _TREE_BLOCK_DIM, device=self.device)
        tree_emb_b = self.tree_encoder.forward_batch(tree_flat_b)
        self_temporal_contexts = self.tree_norm(self_temporal_contexts + tree_emb_b)

        final_embeddings = []
        comm_regs = []
        comm_gate_means = []
        comm_intents = []
        for i in range(batch_size):
            self_ctx = self_temporal_contexts[i]
            opps = all_opponents[i]

            if len(opps) > 0:
                opp_embs = []
                for opp_obs in opps:
                    opp_t = self._to_1d_tensor(opp_obs)
                    opp_base = opp_t[:self.base_obs_dim]  # First 35D only
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
            missing, unexpected = self.load_state_dict(sd, strict=False)
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
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.actor = self._load(self.actor, filename + ".actor")
        self.critic = self._load(self.critic, filename + ".value")
        self.deadlock_head = self._load(self.deadlock_head, filename + ".deadlock")


# =============================================================================
# PPO PARAMETERS (extended with temporal_window)
# =============================================================================

MARL_ATTENTION_TEMPORAL_MAPPO_Param = namedtuple('MARL_ATTENTION_TEMPORAL_MAPPO_Param',
                            ['hidden_size', 'batch_size', 'learning_rate',
                             'discount', 'gae_lambda', 'use_gpu',
                             'max_episodes_in_training_memory', 'batch_fraction', 'k_epochs',
                             'max_batches_per_training', 'temporal_window', 'encoder_type'])


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
        self.optimizer_mode = optimizer_mode.lower()  # 'single' or 'multiple'

        self.state_size = state_size  # temporal obs size per timestep (35D/94D)
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
        else:
            self.hidden_size = 256
            self.learning_rate = 5.0e-3
            self.discount = 0.99
            self.batch_size = 128  # Back to baseline
            self.temporal_window = 3
            self.encoder_type = 'transformer'

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
            
        self.surrogate_eps_clip = 0.12  # tighter trust region to reduce KL spikes
        self.weight_loss = 2.0  # Prioritize critic fitting to stabilize actor guidance
        self.weight_entropy = 0.03  # Increased to counter entropy collapse observed after ~800 episodes.
        self.weight_policy = 1.0
        self.weight_aux_deadlock = 0.08
        # Sparse-switch maps: keep forward dominant and avoid forcing turn frequency.
        self.weight_action_diversity = 0.06
        self.forward_prob_soft_max = 0.72
        self.lr_prob_soft_min = 0.05
        self.idle_prob_soft_max = 0.30
        # Apply action-diversity shaping only at meaningful conflict/decision contexts.
        self.action_diversity_gate_enabled = True
        self.action_diversity_decision_idx = 30
        self.action_diversity_local_deadlock_idx = 5
        self.action_diversity_gate_threshold = 0.5
        self.aux_deadlock_pos_weight = 4.0
        self.weight_comm = 3.0e-4  # weak communication sparsity regularizer
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
        self.actor_lr_decay_on_instability = 0.92
        self.actor_lr_recover_rate = 1.02
        self.actor_lr_min_factor = 0.50
        self.actor_lr_max_factor = 1.00
        self.gae_lambda = self.ppo_parameters.gae_lambda if self.ppo_parameters else 0.95

        # Reward scaling: raw per-step rewards are O(-0.5), giving discounted returns
        # in [-30, +3]. With SmoothL1(beta=1) in the linear regime, V_Loss stays ≈ 10
        # and advantages remain noisy. Scaling rewards to [-3, +0.3] drives V_Loss to
        # <0.1, enabling the critic to converge. Policy gradient is unaffected because
        # advantages are normalized per mini-batch regardless of absolute scale.
        self.reward_scale = 0.1

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
            
            all_params = list(self.encoder_actor.parameters()) + \
                        list(self.actor_critic_model.actor.parameters()) + \
                        list(self.encoder_critic.parameters()) + \
                        list(self.actor_critic_model.critic.parameters())
            
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
        self.entropy_floor = 0.60       # Higher floor to avoid premature deterministic collapse.
        self.entropy_recovery_scale = 5.0  # Stronger entropy rescue when floor is violated.

        self.loss_function = nn.SmoothL1Loss(beta=1.0)
        self.training_step_count = 0
        
        # TensorBoard writer (optional, can be set via set_tensorboard_writer)
        self.writer = None
        self._tensorboard_batch_counter = 0

        # Observation sanity statistics (gesammelt über 100 Episoden)
        self._obs_stat_buffer: list = []   # rohe Feature-Vektoren der letzten 100 Ep.
        self._obs_stat_interval = 100

        # Training-Kennzahlen je Batch (alle _obs_stat_interval Episoden geleert)
        self._stat_buf: dict = {
            'v_loss': [], 'p_loss': [], 'e_loss': [], 'aux_dl': [],
            'kl': [], 'ratio': [], 'entropy': [],
            'adv_mean': [], 'adv_std': [], 'grad_norm': [],
            'ret_min': [], 'ret_max': [],
            'comm_loss': [], 'action_div_loss': [], 'action_div_gate_ratio': [], 'total_loss': [],
            'action_hist': [],
        }
        # Episode-Kennzahlen (Reward + Done-Rate)
        self._ep_stat_buf: dict = {'reward': [], 'done_frac': []}

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
            return min(0.12, float(self.surrogate_eps_clip))
        if self.episode_count < self.stability_guard_start_episode:
            return float(self.surrogate_eps_clip)
        if self.episode_count >= self.stability_guard_hard_episode:
            return max(0.08, self.surrogate_eps_clip * 0.60)
        return max(0.10, self.surrogate_eps_clip * 0.75)

    def _effective_k_epochs(self) -> int:
        if self.episode_count < 200:
            return 1
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
        with torch.no_grad():
            emb = self.encoder_actor.forward_agent(temporal_state, handle)
            emb_batch = emb.unsqueeze(0)  # (1, hidden_dim)
            dist = self.actor_critic_model.get_actor_dist(emb_batch)
            action = dist.sample()

        return action.item()

    def step(self, handle, state, action, reward, next_state, done):
        """Store transition - state is now temporal sequence!"""
        aux_deadlock = self._extract_deadlock_label_from_temporal_state(next_state)
        transition = (state, action, reward, next_state, done, aux_deadlock)
        self.current_episode_memory.push_transition(handle, transition)
        # Observation-Statistik: letzten Frame des temporalen Fensters sammeln
        try:
            last_frame = np.asarray(state[-1][0], dtype=np.float32).reshape(-1)
            if last_frame.shape[0] > 0:
                self._obs_stat_buffer.append(last_frame)
        except Exception:
            pass

    @staticmethod
    def _extract_deadlock_label_from_temporal_state(temporal_state) -> float:
        """Build a robust [0,1] deadlock-risk label from next-state features.

        Works with both 35D DecisionPointObservation (Modus A Pure) and 94D
        HierarchicalRoutesObservation (35D base + sparse neighbors).
        
        Modus A Pure feature indices:
          [5]      = local_deadlock (confirmed)
          [30]     = confirmed_deadlock (tree-based)
          [31]     = max_deadlock (peak in search)
          [32]     = conflict_density (normalized)
        """
        try:
            last_obs = np.asarray(temporal_state[-1][0], dtype=np.float32).reshape(-1)
        except Exception:
            return 0.0

        if last_obs.shape[0] == 0:
            return 0.0

        def _safe(idx: int) -> float:
            if 0 <= idx < last_obs.shape[0]:
                return float(last_obs[idx])
            return 0.0

        # Modus A Pure deadlock signals
        local_deadlock  = _safe(5)       # Immediate context signal
        tree_confirmed  = _safe(30)      # Tree-based confirmation
        tree_max_risk   = _safe(31)      # Worst case in subtree
        conflict_density = _safe(32)     # Conflict frequency (0 to 1)
        
        # Optional extra cues in hierarchical sparse-neighbor block.
        # Per-neighbor 6D block: index +3 == local conflict flag.
        # Current supported layouts:
        #   - 35D base -> no sparse block
        #   - 94D base -> sparse block starts at 35
        sparse_local = 0.0
        if last_obs.shape[0] >= 94:
            sparse_start = 35
            sparse_local = max(
                _safe(sparse_start + 3),
                _safe(sparse_start + 9),
                _safe(sparse_start + 15),
                _safe(sparse_start + 21),
            )

        # Combine signals: local direct signal + tree-based estimates
        risk = max(
            local_deadlock,
            tree_confirmed,
            min(1.0, 0.5 * tree_max_risk),
            min(1.0, 0.3 * conflict_density),
            sparse_local,
        )
        return float(np.clip(risk, 0.0, 1.0))

    def _extract_action_diversity_gate_from_temporal_state(self, temporal_state) -> float:
        """Return gate in [0,1] for where diversity shaping should be active.

        Gate is active only at decision points and local deadlock contexts.
        This avoids penalizing unavoidable forward moves on straight corridors.
        """
        try:
            last_obs = np.asarray(temporal_state[-1][0], dtype=np.float32).reshape(-1)
        except Exception:
            return 0.0

        if last_obs.shape[0] == 0:
            return 0.0

        def _safe(idx: int) -> float:
            if 0 <= idx < last_obs.shape[0]:
                return float(last_obs[idx])
            return 0.0

        decision_required = _safe(int(getattr(self, 'action_diversity_decision_idx', 30)))
        local_deadlock = _safe(int(getattr(self, 'action_diversity_local_deadlock_idx', 5)))
        thr = float(getattr(self, 'action_diversity_gate_threshold', 0.5))

        gate = 1.0 if (decision_required >= thr or local_deadlock >= thr) else 0.0
        return gate

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

        episode_data = []
        trajectory_count = 0
        num_episodes = len(self.accumulated_episodes)
        
        if num_episodes < self.max_episodes_in_training_memory:
            if self.show_pre_train_debug_msg:
                print(f"\n🔍 Collect episodes {num_episodes}/{self.max_episodes_in_training_memory}")
            return
        
        # ⚡ OPTIMIZATION: Process ALL episodes with batched encoding
        for episode_memory in self.accumulated_episodes:
            episode_state_tuples = []
            episode_actions = []
            episode_advantages = []
            episode_returns = []
            episode_aux_deadlock = []
            
            # Collect all trajectories first (avoid repeated handle iteration)
            all_trajectories = []
            for handle in range(len(episode_memory)):
                agent_episode_history = episode_memory.get_transitions(handle)
                if len(agent_episode_history) > 0:
                    all_trajectories.append(agent_episode_history)
            
            if len(all_trajectories) == 0:
                continue
                
            trajectory_count += len(all_trajectories)
            
            # Process all trajectories in parallel batches
            for agent_episode_history in all_trajectories:
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
                
                episode_state_tuples.extend(state_tuples)
                episode_actions.append(actions)
                episode_advantages.append(traj_gae_advantages)
                episode_returns.append(traj_gae_returns)
                episode_aux_deadlock.append(aux_deadlock)
            
            if len(episode_state_tuples) > 0:
                episode_data.append((
                    episode_state_tuples,
                    torch.cat(episode_actions, dim=0),
                    torch.cat(episode_advantages, dim=0),
                    torch.cat(episode_returns, dim=0),
                    torch.cat(episode_aux_deadlock, dim=0)
                ))
        
        if len(episode_data) == 0:
            print("⚠️ No transitions to train on!")
            return
        
        # Concatenate all episode data
        all_state_tuples = []
        all_actions = []
        all_gae_advantages = []
        all_gae_returns = []
        all_aux_deadlock = []
        
        # ⚡ NEW: Track episode indices for recency-based sampling
        episode_sample_weights = []
        
        for ep_idx, (ep_states, ep_actions, ep_advantages, ep_returns, ep_aux_deadlock) in enumerate(episode_data):
            all_state_tuples.extend(ep_states)
            all_actions.append(ep_actions)
            all_gae_advantages.append(ep_advantages)
            all_gae_returns.append(ep_returns)
            all_aux_deadlock.append(ep_aux_deadlock)
            
            # Uniform sampling across the sliding window avoids forgetting older
            # but still relevant traffic patterns and stabilizes long runs.
            combined_weight = 1.0
            
            episode_sample_weights.extend([combined_weight] * len(ep_states))
        
        all_actions = torch.cat(all_actions, dim=0)
        all_gae_advantages = torch.cat(all_gae_advantages, dim=0)
        all_gae_returns = torch.cat(all_gae_returns, dim=0)
        all_aux_deadlock = torch.cat(all_aux_deadlock, dim=0)
        episode_sample_weights = torch.tensor(episode_sample_weights, dtype=torch.float32)
        
        total_samples = len(all_state_tuples)
        
        # Determine batch configuration
        total_possible_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        if self.max_batches_per_training is not None:
            num_batches = min(self.max_batches_per_training, total_possible_batches)
            samples_to_use = min(total_samples, num_batches * self.batch_size)
        else:
            num_batches = max(1, int(total_possible_batches * self.batch_fraction))
            samples_to_use = min(total_samples, num_batches * self.batch_size)
        
        # ⚡ Weighted sampling: newer episodes have higher probability
        # Use multinomial sampling WITHOUT replacement to avoid overfitting
        if samples_to_use >= total_samples:
            # Use all samples
            sampled_indices = torch.arange(total_samples)
        else:
            # Sample without replacement (each sample max 1×)
            sampled_indices = torch.multinomial(
                episode_sample_weights, 
                num_samples=samples_to_use, 
                replacement=False  # ⚡ Prevents overfitting on new episodes
            )
        all_state_tuples = [all_state_tuples[i] for i in sampled_indices]
        all_actions = all_actions[sampled_indices]
        all_gae_advantages = all_gae_advantages[sampled_indices]
        all_gae_returns = all_gae_returns[sampled_indices]
        all_aux_deadlock = all_aux_deadlock[sampled_indices]
        
        if self.show_pre_train_debug_msg:
            print(f"📦 Using {samples_to_use}/{total_samples} samples ({samples_to_use/total_samples*100:.1f}%)")
            k_epochs_eff = self._effective_k_epochs()
            print(f"📦 Batch Config: {num_batches} batches (batch_size={self.batch_size}) over {k_epochs_eff} epochs")
        
        # 🎯 KRITISCH: Berechne old_logprobs VOR dem Training
        # Dies ist die EINZIGE korrekte Methode für PPO!
        batch_size_encoding = 256  # ⚡ WICHTIG: Außerhalb definieren!
        
        if self.show_pre_train_debug_msg:
            print(f"\n🔍 Computing initial old_logprobs for {len(all_state_tuples)} samples...")
        
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
        
        if self.show_pre_train_debug_msg:
            print(f"✅ Initial old_logprobs computed (mean={all_old_logprobs.mean().item():.4f})")
        
        k_epochs_eff = self._effective_k_epochs()
        total_iterations = k_epochs_eff * num_batches
        current_iteration = 0
        hard_spike_batches_total = 0
        hard_spike_streak = 0
        action_hist_total = torch.zeros(self.action_size, dtype=torch.long)
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
                
                batch_state_tuples = [all_state_tuples[i] for i in batch_indices]
                batch_actions = all_actions[batch_indices]
                batch_action_hist = torch.bincount(batch_actions.detach().cpu(), minlength=self.action_size)
                action_hist_total += batch_action_hist
                batch_gae_advantages = all_gae_advantages[batch_indices]
                batch_gae_returns = all_gae_returns[batch_indices]
                batch_aux_deadlock = all_aux_deadlock[batch_indices]
                batch_old_logprobs = all_old_logprobs[batch_indices]  # 🎯 Pre-computed!

                comm_progress = self._apply_comm_schedule()

                # Encode states
                states_actor = self.encoder_actor.forward_batch(batch_state_tuples)
                states_critic = self.encoder_critic.forward_batch(batch_state_tuples)

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
                advantages = advantages_normalized

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
                        adiv_gate = torch.tensor(gate_vals, dtype=probs.dtype, device=self.device)
                    else:
                        adiv_gate = torch.ones(probs.shape[0], dtype=probs.dtype, device=self.device)

                    gate_sum = float(adiv_gate.sum().item())
                    action_diversity_gate_ratio = float(adiv_gate.mean().item())

                    # Soft constraints via squared hinge losses:
                    # 1) keep forward probability below a soft cap
                    # 2) keep combined left+right probability above a soft floor
                    # 3) keep idle actions (DO_NOTHING + STOP) below a soft cap
                    if gate_sum >= 1.0:
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
                deadlock_logits = torch.squeeze(self.actor_critic_model.deadlock_head(states_actor), dim=-1)
                aux_targets = torch.clamp(batch_aux_deadlock, 0.0, 1.0)
                pos_weight = torch.full_like(aux_targets, self.aux_deadlock_pos_weight)
                aux_deadlock_loss_component = nn.functional.binary_cross_entropy_with_logits(
                    deadlock_logits,
                    aux_targets,
                    pos_weight=pos_weight,
                )
                comm_loss_component = self.encoder_actor.last_comm_reg + self.encoder_critic.last_comm_reg
                ratio_mean = ratios.mean().item()
                approx_kl = torch.abs((batch_old_logprobs - logprobs).mean()).item()

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
                    policy_weight_eff *= 0.90
                elif value_loss_scalar < 0.45:
                    value_weight_eff *= 0.90
                    policy_weight_eff *= 1.05

                loss = \
                    policy_weight_eff * policy_loss_component \
                    + value_weight_eff * value_loss_component \
                    + entropy_weight_eff * entropy_loss_component \
                    + self.weight_action_diversity * action_diversity_loss_component \
                    + self.weight_aux_deadlock * aux_deadlock_loss_component \
                    + comm_weight_eff * comm_loss_component

                # Backward pass
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
                
                # Gradient clipping (Standard PPO: max_norm=0.5..1.0 ist üblich).
                grad_norm_actor = torch.nn.utils.clip_grad_norm_(
                    list(self.encoder_actor.parameters()) + 
                    list(self.actor_critic_model.actor.parameters()),
                    max_norm=0.35
                )
                
                grad_norm_critic = torch.nn.utils.clip_grad_norm_(
                    list(self.encoder_critic.parameters()) + 
                    list(self.actor_critic_model.critic.parameters()),
                    max_norm=0.35
                )
                
                grad_norm = max(grad_norm_actor.item(), grad_norm_critic.item())
                
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
                self._stat_buf['comm_loss'].append(comm_loss_component.item())
                self._stat_buf['action_div_loss'].append(action_diversity_loss_component.item())
                self._stat_buf['action_div_gate_ratio'].append(action_diversity_gate_ratio)
                self._stat_buf['total_loss'].append(loss.item())
                self._stat_buf['action_hist'].append(batch_action_hist.detach().cpu().numpy())
                
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
                }
                self._log_batch_metrics(batch_metrics)
                self._stat_buf['ret_min'].append(batch_gae_returns.min().item())
                self._stat_buf['ret_max'].append(batch_gae_returns.max().item())

                # 📊 Log metrics for this iteration
                adv_mean = raw_adv_mean  # ⚡ RAW advantage mean (BEFORE normalization)
                adv_std = raw_adv_std    # ⚡ RAW advantage std (BEFORE normalization)
                
                if self.show_progress_bar:
                    progress = current_iteration / total_iterations
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r  [{bar}] Epoch {k_loop+1}/{k_epochs_eff}, Batch {batch_idx+1}/{num_batches} ({progress*100:3.1f}%)", end='')
                    print(f"\t", end='')
                    print(f"| Loss: {loss.item():.4f}", end='')
                    print(f"| P_Loss: {policy_loss_component.item():.4f}", end='')
                    print(f"| V_Loss: {value_loss_component.item():.4f}", end='')
                    print(f"| E_Loss: {entropy_loss_component.item():.4f}", end='')
                    print(f"| Adiv: {action_diversity_loss_component.item():.4f}", end='')
                    print(f"| AdivMask: {action_diversity_gate_ratio:.2f}", end='')
                    print(f"| AuxDL: {aux_deadlock_loss_component.item():.4f}", end='')
                    print(f"| C_Loss: {comm_loss_component.item():.4f}", end='')
                    print(f"| Adv: {adv_mean:.2f}±{adv_std:.2f}", end='')  # ⚡ RAW advantage (mean±std BEFORE norm)
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
            print("\n")  # New line after progress bar
        
        if not hasattr(self, 'training_step'):
            self.training_step = 0
        self.training_step += 1

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

    # Feature-Namen für das feste DecisionPointObservation-Layout (64D)
    _OBS_FEATURE_NAMES = [
        "is_switch",   "hint_L",      "hint_F",      "hint_R",      "is_merge",
        "local_dl",
        "swL_progress","swL_dl_sig",  "swL_switches","swL_dist",
        "swL_target",  "swL_abort",   "swL_dl_ahead","valid_left",
        "swF_progress","swF_dl_sig",  "swF_switches","swF_dist",
        "swF_target",  "swF_abort",   "swF_dl_ahead","valid_forward",
        "swR_progress","swR_dl_sig",  "swR_switches","swR_dist",
        "swR_target",  "swR_abort",   "swR_dl_ahead","valid_right",
        "decision_required",
        "mgF_dl_sig",  "mgF_switches","mgF_target",  "mgF_abort",  "mgF_dl_ahead",
        "mgB_dl_sig",  "mgB_switches","mgB_target",  "mgB_abort",  "mgB_dl_ahead",
        "st_0","st_1","st_2","st_3","st_4","st_5","st_6",
        "act_DN","act_L","act_F","act_R","act_S",
        "priority_rank",
        "ct_OUTSIDE","ct_FORWARD_ONLY","ct_MERGING","ct_SWITCH","ct_DONE",
        "tr_FWD_FWD","tr_FWD_MRG","tr_FWD_SWI","tr_SWI_FWD","tr_MRG_FWD",
    ]

    # Short prose descriptions for each feature — same order as _OBS_FEATURE_NAMES
    _OBS_FEATURE_DESC = [
        # 0-5 Base
        "Switch present (branching possible)",        # is_switch
        "Shortest path goes Left",                    # hint_L
        "Shortest path goes Forward",                 # hint_F
        "Shortest path goes Right",                   # hint_R
        "Merge point (track join) ahead",             # is_merge
        "Deadlock detected at current position",      # local_dl
        # 6-13 Left branch
        "Left branch: progress toward target",        # swL_progress
        "Left branch: deadlock risk signal",          # swL_dl_sig
        "Left branch: number of switches on path",    # swL_switches
        "Left branch: distance to target",            # swL_dist
        "Left branch: target reachable via this arm", # swL_target
        "Left branch: dead-end / abort detected",     # swL_abort
        "Left branch: deadlock ahead",                # swL_dl_ahead
        "Left branch: valid action from current cell",# valid_left
        # 14-21 Forward branch
        "Fwd branch: progress toward target",         # swF_progress
        "Fwd branch: deadlock risk signal",           # swF_dl_sig
        "Fwd branch: number of switches on path",     # swF_switches
        "Fwd branch: distance to target",             # swF_dist
        "Fwd branch: target reachable",               # swF_target
        "Fwd branch: dead-end / abort detected",      # swF_abort
        "Fwd branch: deadlock ahead",                 # swF_dl_ahead
        "Fwd branch: valid action from current cell", # valid_forward
        # 22-29 Right branch
        "Right branch: progress toward target",       # swR_progress
        "Right branch: deadlock risk signal",         # swR_dl_sig
        "Right branch: number of switches on path",   # swR_switches
        "Right branch: distance to target",           # swR_dist
        "Right branch: target reachable",             # swR_target
        "Right branch: dead-end / abort detected",    # swR_abort
        "Right branch: deadlock ahead",               # swR_dl_ahead
        "Right branch: valid action from current cell", # valid_right
        # 30 Decision
        "Decision required at current cell",          # decision_required
        # 31-35 Merge-Forward
        "Merge-Fwd: deadlock risk signal",            # mgF_dl_sig
        "Merge-Fwd: number of switches on path",      # mgF_switches
        "Merge-Fwd: target reachable",                # mgF_target
        "Merge-Fwd: dead-end / abort detected",       # mgF_abort
        "Merge-Fwd: deadlock ahead",                  # mgF_dl_ahead
        # 36-40 Merge-Backward
        "Merge-Back: deadlock risk signal",           # mgB_dl_sig
        "Merge-Back: number of switches on path",     # mgB_switches
        "Merge-Back: target reachable",               # mgB_target
        "Merge-Back: dead-end / abort detected",      # mgB_abort
        "Merge-Back: deadlock ahead",                 # mgB_dl_ahead
        # 41-47 TrainState one-hot
        "TrainState: READY_TO_DEPART",                # st_0
        "TrainState: MALFUNCTION_OFF_MAP",            # st_1
        "TrainState: MOVING",                         # st_2
        "TrainState: STOPPED",                        # st_3
        "TrainState: MALFUNCTION (on map)",           # st_4
        "TrainState: WAITING",                        # st_5
        "TrainState: DONE (reached target)",          # st_6
        # 48-52 Last action one-hot
        "Last action: DO_NOTHING",                    # act_DN
        "Last action: turn Left",                     # act_L
        "Last action: move Forward",                  # act_F
        "Last action: turn Right",                    # act_R
        "Last action: STOP",                          # act_S
        # 53 Priority
        "Priority rank: 0=nearest target, 1=farthest",  # priority_rank
        # 54-58 Cell-type one-hot
        "Cell type: OUTSIDE (off network)",           # ct_OUTSIDE
        "Cell type: FORWARD_ONLY (straight track)",   # ct_FORWARD_ONLY
        "Cell type: MERGING (track join)",            # ct_MERGING
        "Cell type: SWITCH (branching point)",        # ct_SWITCH
        "Cell type: DONE (target cell)",              # ct_DONE
        # 59-63 Selected transitions (non-null in practice)
        "Transition: FORWARD→FORWARD (straight run)",  # tr_FWD_FWD
        "Transition: FORWARD→MERGE (approaching join)", # tr_FWD_MRG
        "Transition: FORWARD→SWITCH (approaching branch)", # tr_FWD_SWI
        "Transition: SWITCH→FORWARD (leaving branch)",  # tr_SWI_FWD
        "Transition: MERGE→FORWARD (leaving join)",     # tr_MRG_FWD
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
        print(f"  Context : Flatland 5-agent rail scheduling, DecisionPoint obs 64D")
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
            adg_m, adg_s = _ms('action_div_gate_ratio')
            ret_min    = float(np.array(sb['ret_min']).min())
            ret_max    = float(np.array(sb['ret_max']).max())
            aux_m      = float(np.array(sb['aux_dl']).mean())

            hdr(f"SECTION 2 — PPO TRAINING METRICS  ({n_b} batches, "
                f"lr_factor={self.actor_lr_factor:.3f})")

            def tr_row(lbl, val, pm, unit, lo_ok, hi_ok, note_ok, note_warn):
                hi_alert = None
                if lbl == "GradNorm":
                    hi_alert = 20.0
                elif lbl == "Entropy":
                    hi_alert = None
                elif lbl == "V_Loss (critic)":
                    hi_alert = 2.0
                st = severity(val, lo_ok, hi_ok, hi_alert=hi_alert)
                note = note_ok if st == "OK" else note_warn
                vs   = f"{val:.4f} ±{pm:.4f}"
                print(row(lbl, vs, unit, st, note) + " |")

            tr_row("V_Loss (critic)",    vl_m, vl_s, "",    0.0, 0.30,
                   "critic fits returns",
                   "critic not converging")
            tr_row("P_Loss (policy)",    pl_m, pl_s, "",   -0.5, 0.5,
                   "policy improving",
                   "gradient signal weak")
            tr_row("Entropy",            en_m, en_s, "nat", self.entropy_floor, 2.0,
                   "exploration ok",
                   f"below floor={self.entropy_floor:.2f}, collapse risk")
            tr_row("KL divergence",      kl_m, kl_s, "",    0.0, self.ppo_target_kl * 1.5,
                   "trust region ok",
                   f"exceeds target={self.ppo_target_kl:.3f}")
            tr_row("PPO ratio mean",     rt_m, rt_s, "",    0.90, 1.15,
                   "ratios stable",
                   "policy update too large/small")
            tr_row("Advantage mean",     am_m, 0.0,  "",   -2.0, 2.0,
                   "centered ok",
                   "advantages heavily biased")
            tr_row("Advantage std",      as_m, 0.0,  "",    0.5, 5.0,
                   "signal present",
                   "signal too weak or exploding")
            tr_row("GradNorm",           gn_m, gn_s, "",    0.0, 4.0,
                   "gradients stable",
                   "gradient instability")
            tr_row("AuxDL loss",         aux_m, 0.0, "",    0.0, 0.8,
                   "deadlock head ok",
                   "deadlock head not converging")
                 tr_row("Adiv gate ratio",    adg_m, adg_s, "frac", 0.05, 0.70,
                     "diversity shaped at decisions",
                     "too sparse/broad gating for diversity")
            print(row("Returns range",
                       f"{ret_min:+.2f}…{ret_max:+.2f}", "", "", "") + " |")
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
            if gn_m > 5.0:
                issues.append(
                    f"GradNorm={gn_m:.2f}: gradient explosion. "
                    "Reduce LR or add gradient clipping.")
            if adg_m < 0.01:
                issues.append(
                    f"Adiv gate ratio={adg_m:.3f}: diversity shaping is almost never active. "
                    "Check if decision-point/deadlock signals are present in observations.")

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
                'action_div_gate_ratio_mean': adg_m,
                'reward_mean': float(r_arr.mean()) if len(r_arr) > 0 else 0.0,
                'done_frac': float(d_arr.mean()) if len(d_arr) > 0 else 0.0,
            }
            self._log_episode_metrics(episode_metrics)

        for key in self._stat_buf:
            self._stat_buf[key].clear()

        # ── 3) Feature Importance ──────────────────────────────────────────────
        print(f"\n{color(W, C_BLUE)}")
        try:
            W_mat = self.encoder_actor.obs_encoder[0].weight.detach().cpu().numpy()
            sens   = np.abs(W_mat).mean(axis=0)
            obs_dim   = sens.shape[0]
            max_s     = sens.max() if sens.max() > 0 else 1.0
            order     = np.argsort(sens)[::-1]
            sens_norm = sens / max_s          # relative importance in [0,1]

            print(f"  SECTION 3 — FEATURE IMPORTANCE  "
                  f"(mean |weight| first encoder layer, {obs_dim}D input)")
            print(f"  Interpretation guide:")
            print(f"    rel_imp = mean|W_col| / max(mean|W_col|)")
            print(f"    >0.50 = highly used  |  0.10-0.50 = moderate  "
                  f"|  <0.05 = nearly ignored  |  <0.01 = dead")
            print()
            print(f"  +{'─'*4}+{'─'*5}+{'─'*22}+{'─'*9}+{'─'*9}+{'─'*26}+{'─'*26}+{'─'*44}+")
            print(f"  | {'Rk':>2} | {'[i]':>3} | {'Name':<20s} | {'abs_sens':>7} "
                  f"| {'rel_imp':>7} | {'Bar (24-wide)':<24s} | {'Interpretation':<24s} | {'Description':<42s} |")
            print(f"  +{'─'*4}+{'─'*5}+{'─'*22}+{'─'*9}+{'─'*9}+{'─'*26}+{'─'*26}+{'─'*44}+")

            def imp_note(rel):
                if rel > 0.50: return "highly used by network"
                if rel > 0.20: return "moderately used"
                if rel > 0.05: return "low usage"
                if rel > 0.01: return "nearly ignored"
                return "DEAD — no gradient signal"

            for rank in range(obs_dim):
                i    = order[rank]
                s    = sens[i]
                r    = sens_norm[i]
                bar  = ('█' * int(r * 24)).ljust(24)
                note = imp_note(r)
                desc = fdesc(i)[:42]
                print(f"  | {rank+1:2d} | [{i:2d}] | {fname(i):<20s} | "
                      f"{s:7.5f} | {r:7.4f} | {bar} | {note:<24s} | {desc:<42s} |")

            print(f"  +{'─'*4}+{'─'*5}+{'─'*22}+{'─'*9}+{'─'*9}+{'─'*26}+{'─'*26}+{'─'*44}+")

            dead_thresh = max_s * 0.01
            dead_feats  = [i for i in range(obs_dim) if sens[i] < dead_thresh]
            top3        = [fname(order[k]) for k in range(min(3, obs_dim))]
            print(f"\n  Top-3 most used : {', '.join(top3)}")
            if dead_feats:
                print(f"  Dead features   : {len(dead_feats)} ignored "
                      f"({', '.join(fname(i) for i in dead_feats[:8])}"
                      f"{'…' if len(dead_feats) > 8 else ''})")
                print(f"  {color('WARN', C_YELLOW)}  Dead features waste network capacity and may indicate "
                      f"structural issues in the observation.")
            else:
                print(f"  {color('OK', C_GREEN)}  All features contribute to network (>=1% sensitivity).")

        except Exception as exc:
            print(f"  (Feature importance not available: {exc})")

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

            print(f"\n{color(W, C_BLUE)}")
            print(f"  SECTION 4 — OBSERVATION SANITY  (N={N} frames, D={D} features)")
            print()

            oor  = [(i, mins[i], maxs[i]) for i in range(D)
                    if mins[i] < -RANGE_TOL or maxs[i] > 1.0 + RANGE_TOL]
            dead = [(i, means[i]) for i in range(D) if stds[i] < 1e-4]

            valid_idx = [i for i in range(D) if stds[i] > 1e-6]
            dups = []
            if len(valid_idx) >= 2:
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr = np.corrcoef(data[:, valid_idx].T)
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                dups = [(valid_idx[ii], valid_idx[jj], corr[ii, jj])
                        for ii in range(len(valid_idx))
                        for jj in range(ii + 1, len(valid_idx))
                        if abs(corr[ii, jj]) > 0.99]

            obs_issues = []
            if oor:
                obs_issues.append(
                    f"{len(oor)} feature(s) outside [0,1]: "
                    + ", ".join(f"[{i}]{fname(i)}(min={lo:+.3f},max={hi:+.3f})"
                                for i, lo, hi in oor[:5]))
            if dead:
                obs_issues.append(
                    f"{len(dead)} constant feature(s) (zero variance): "
                    + ", ".join(f"[{i}]{fname(i)}" for i, _ in dead[:8]))
            if dups:
                obs_issues.append(
                    f"{len(dups)} duplicate feature pair(s) (|corr|>0.99): "
                    + ", ".join(f"[{a}]{fname(a)}↔[{b}]{fname(b)}"
                                for a, b, _ in dups[:4]))

            print(f"  +{'─'*30}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*6}+")
            print(f"  | {'Feature':<28s} | {'mean':>5} | {'std':>5} | "
                  f"{'min':>5} | {'max':>5} | {'oor':>5} | {'dead':>4} |")
            print(f"  +{'─'*30}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*7}+{'─'*6}+")
            for i in range(D):
                is_oor  = mins[i] < -RANGE_TOL or maxs[i] > 1.0 + RANGE_TOL
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
                print(f"\n  Duplicate pairs (|corr|>0.99):")
                for a, b, c in dups:
                    print(f"    [{a}]{fname(a):<20s} <-> [{b}]{fname(b):<20s}  corr={c:+.4f}")

            # Key feature means
            if D > 34:  # 35D Modus A Pure
                print(f"\n  Key feature means:")
                print(f"    is_switch={means[0]:.3f}  is_merge={means[4]:.3f}  "
                      f"local_dl={means[5]:.3f}  priority_rank={means[18]:.3f}")

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
        print(f"  Compact block for direct next-action decisions:")
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
            f"  OBS: DecisionPoint54 + temporal LSTM + comm-attn\n"
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
            f"  actor_lr_factor={self.actor_lr_factor:.4f} clip_eps={self._effective_clip_eps():.3f}\n"
            f"\n"
            f"  DECISION_RULES:\n"
            f"  - if done_mean < 0.22 and v_loss > 0.80: increase critic pressure or reduce reward penalties\n"
            f"  - if entropy < {self.entropy_floor:.2f}: raise entropy weight / recovery scale\n"
            f"  - if forward-action share > 0.55: increase action-diversity penalty\n"
            f"  - if grad > 5.0: lower actor lr or tighten clipping\n"
            f"\n"
            f"  ISSUES:\n{issue_text}\n"
            f"\n"
            f"  VORSCHLAG:\n  " + "\n  ".join(suggestions)
        )
        print(f"  {'─'*76}")
        print(f"{color(W, C_BLUE)}\n")

    def _apply_episode_end_bonus(self, episode_memory, done_count, num_agents):
        """Apply cooperative episode-end shaping — PER AGENT ONLY.

        🎯 CRITICAL: Bonus applied ONLY to agents where final transition has done==True
        - Each agent (handle) has at most ONE done==True transition (the last one)
        - Multiple agents CAN be done in the same episode
        - Bonus applied exactly ONCE per agent, to their final transition only
        
        Mechanism:
        - Progress reward grows non-linearly with team completion ratio
        - Deadlock-risk on terminal states is penalized
        - Total shaping is normalized and capped for stability across 1..n agents

        Returns:
            Total scalar delta applied over all modified final transitions.
        """
        if num_agents <= 0:
            return 0.0
    
        # Cooperative progress bonus is computed as an EPISODE budget, then distributed
        # over done agents. This keeps scale stable for variable team sizes.
        completion_ratio = float(done_count) / float(num_agents)
        TEAM_BONUS_EPISODE_CAP = 100.0
        TEAM_BONUS_EXP = 1.8
        total_bonus_budget = TEAM_BONUS_EPISODE_CAP * (completion_ratio ** TEAM_BONUS_EXP)
        per_done_agent_bonus = total_bonus_budget / max(1, done_count)

        # Penalize agents that finish in high deadlock-risk states (aux_dl in [0, 1]).
        DEADLOCK_PENALTY_PER_AGENT_MAX = 45.0

        # Safety clamps to prevent reward/cost explosion.
        PER_AGENT_DELTA_MIN = -40.0
        PER_AGENT_DELTA_MAX = 45.0
        EPISODE_DELTA_ABS_CAP = 120.0

        done_updates = []
        for handle in episode_memory.memory:
            transitions = episode_memory.memory[handle]
            if not transitions:
                continue

            # Get final transition for this agent
            state, action, reward, next_state, done, aux_dl = transitions[-1]
            # 🎯 ONLY process agents with done==True in their final transition
            if not done:
                continue
            
            # Index [30] = confirmed_deadlock in 35D Modus A Pure features
            deadlock_risk = max(state[0][0][30], next_state[0][0][30])
            delta = per_done_agent_bonus - (DEADLOCK_PENALTY_PER_AGENT_MAX * deadlock_risk)
            delta = float(np.clip(delta, PER_AGENT_DELTA_MIN, PER_AGENT_DELTA_MAX))
            # Store: (handle, ..., delta) — exactly ONE entry per done agent
            done_updates.append((handle, state, action, reward, next_state, done, aux_dl, delta))

        if not done_updates:
            return 0.0

        total_delta = float(sum(item[7] for item in done_updates))

        # Final episode-level cap independent of number of agents.
        if abs(total_delta) > EPISODE_DELTA_ABS_CAP:
            scale = EPISODE_DELTA_ABS_CAP / (abs(total_delta) + 1e-8)
            done_updates = [
                (h, s, a, r, ns, d, adl, delta * scale)
                for (h, s, a, r, ns, d, adl, delta) in done_updates
            ]
            total_delta = float(sum(item[7] for item in done_updates))

        # 🎯 Apply bonus to each done agent — exactly ONCE to their final transition
        for handle, state, action, reward, next_state, done, aux_dl, delta in done_updates:
            transitions = episode_memory.memory[handle]
            # Modify only the final transition reward (done flag stays True)
            transitions[-1] = (state, action, reward + delta, next_state, done, aux_dl)

        return float(total_delta)

    def end_episode(self, train):
        if train:
            # Collect episode-level stats before buffer is reset
            ep_total_reward = 0.0
            ep_done_count   = 0
            ep_n_agents     = len(self.current_episode_memory.memory)
            for transitions in self.current_episode_memory.memory.values():
                if transitions:
                    ep_total_reward += sum(float(t[2]) for t in transitions)
                    # Count agents with done==True in their final transition
                    if transitions[-1][4]:   # done flag of last transition
                        ep_done_count += 1
            # ═══════════════════════════════════════════════════════════════
            # POST-EPISODE SHAPING: cooperative progress bonus + deadlock penalty
            # Applied PER AGENT to agents with done==True only
            # ═══════════════════════════════════════════════════════════════
            team_bonus = self._apply_episode_end_bonus(
                self.current_episode_memory, ep_done_count, ep_n_agents
            )

            # Keep logging aligned with the actual rewards used for training.
            ep_total_reward += team_bonus
            self._ep_stat_buf['reward'].append(ep_total_reward)
            self._ep_stat_buf['done_frac'].append(
                ep_done_count / ep_n_agents if ep_n_agents > 0 else 0.0)

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
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except Exception as e:
                print(f" >> failed to load: {e}")
        return obj

    @staticmethod
    def _optimizer_state_compatible(optimizer) -> bool:
        """Validate that tensor-shaped optimizer states match current param shapes."""
        try:
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p not in optimizer.state:
                        continue
                    st = optimizer.state[p]
                    for _, v in st.items():
                        if torch.is_tensor(v) and tuple(v.shape) != tuple(p.shape):
                            return False
            return True
        except Exception:
            return False

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

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

        # -------- not yet working -------
        #    if len(transitions) > 0:
        #        el = transitions[len(transitions)-1]
        #        (_,_,_,_, done) = el
        #        if done:
        #            return
        # -------------------------------

        transitions.append(transition)
        self.memory.update({handle: transitions})


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
    1. Observation Encoder: Maps 33D obs → 128D embedding
    2. TEMPORAL Attention: Links t-2, t-1, t → learns movement patterns
    3. SPATIAL Attention: Links self + opponents → learns interactions
    4. Output Projection: Final 128D context embedding
    
    Advantages over simple MAEncoderWithAttention:
    ✅ Sees movement (not just snapshot)
    ✅ Predicts collisions (velocity awareness)
    ✅ Smoother trajectories (temporal context)
    """
    
    def __init__(self, 
                 obs_dim: int,           # 33D (30 base + 3 velocity)
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
        # ========================================================================
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
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
            print(f"⚠️ WARNING: Input contains NaN/Inf, replacing with 0")
            t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # ⚠️ PROTECTION: Clamp extreme values
        t = torch.clamp(t, min=-10.0, max=10.0)
        
        return t

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
                         Each obs is 33D numpy array
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
            obs_t = self._to_1d_tensor(obs_self)  # (33,)
            emb_t = self.obs_encoder(obs_t)       # (128,)
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
        
        # ========================================================================
        # STEP 3: SPATIAL ATTENTION - Multi-Agent Interaction
        # ========================================================================
        
        # Get current opponent observations (only from t, not entire history)
        _, current_opponents = temporal_seq[-1]  # Last timestep
        
        # Encode opponents (current timestep only)
        opp_embeddings = []
        for opp_obs in current_opponents:
            opp_t = self._to_1d_tensor(opp_obs)  # (33,)
            # Take only base observation (first 33D)
            if opp_t.shape[0] > self.obs_dim:
                opp_base = opp_t[:self.obs_dim]
            else:
                opp_base = opp_t
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
        
        # Reshape for parallel encoding: (batch_size * temporal_window, 33)
        flat_obs = all_self_obs_tensor.view(-1, self.obs_dim)
        
        # Encode all at once: (batch_size * temporal_window, hidden_dim)
        flat_embeddings = self.obs_encoder(flat_obs)
        
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
                    if opp_t.shape[0] > self.obs_dim:
                        opp_t = opp_t[:self.obs_dim]
                    opp_embs.append(self.obs_encoder(opp_t))
                
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

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
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
            emb_t = self.obs_encoder(obs_t)
            self_obs_sequence.append(emb_t)

        self_seq_tensor = torch.stack(self_obs_sequence, dim=0).unsqueeze(0)
        temporal_output, _ = self.temporal_lstm(self_seq_tensor)
        self_temporal_context = temporal_output[0, -1, :]

        _, current_opponents = temporal_seq[-1]
        opp_embeddings = []
        for opp_obs in current_opponents:
            opp_t = self._to_1d_tensor(opp_obs)
            if opp_t.shape[0] > self.obs_dim:
                opp_t = opp_t[:self.obs_dim]
            opp_embeddings.append(self.obs_encoder(opp_t))

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
        flat_obs = all_self_obs_tensor.view(-1, self.obs_dim)
        flat_embeddings = self.obs_encoder(flat_obs)
        self_embeddings = flat_embeddings.view(batch_size, self.temporal_window, self.hidden_dim)

        temporal_output, _ = self.temporal_lstm(self_embeddings)
        self_temporal_contexts = temporal_output[:, -1, :]

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
                    if opp_t.shape[0] > self.obs_dim:
                        opp_t = opp_t[:self.obs_dim]
                    opp_embs.append(self.obs_encoder(opp_t))

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


# =============================================================================
# ACTOR-CRITIC MODEL  
# =============================================================================

class ActorCriticModel(nn.Module):
    def __init__(self, state_size, action_size, device, hidsize1=512, hidsize2=256):
        super(ActorCriticModel, self).__init__()
        self.device = device
        
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, action_size)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, 1)
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
                 train_frequency = 1):
        super(MARL_ATTENTION_TEMPORAL_PPOPolicy, self).__init__()

        self.show_debug_msg = False
        self.show_pre_train_debug_msg = show_pre_train_debug_msg
        self.show_progress_bar = show_progress_bar
        self.train_frequency = train_frequency
        self.episode_count = 0

        self.state_size = state_size  # 33D per timestep
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
            
        self.surrogate_eps_clip = 0.15  # slightly tighter trust region
        self.weight_loss = 0.5  # Standard
        self.weight_entropy = 0.01  # Lower entropy pressure for late-stage policy stability
        self.weight_policy = 1.0
        self.weight_aux_deadlock = 0.08
        self.aux_deadlock_pos_weight = 4.0
        self.weight_comm = 3.0e-4  # weak communication sparsity regularizer
        self.comm_reg_start_episode = 300
        self.comm_reg_full_episode = 600
        self.comm_dropout_early = 0.00
        self.comm_dropout_late = 0.05
        self.stability_guard_start_episode = 900
        self.stability_guard_hard_episode = 1300
        self.ppo_target_kl = 0.020
        self.ppo_max_kl = 0.045
        self.ratio_guard_soft = 1.12
        self.ratio_guard_hard = 1.22
        self.ratio_guard_soft_low = 0.88
        self.ratio_guard_hard_low = 0.75
        self.ppo_emergency_kl = 0.12
        self.ppo_emergency_kl_hard = 0.25
        self.comm_gate_target = 0.032
        self.max_hard_batches_before_lr_decay = 4
        self.hard_spike_streak_limit = 2
        self.actor_lr_decay_on_instability = 0.85
        self.actor_lr_recover_rate = 1.01
        self.actor_lr_min_factor = 0.15
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
        self.actor_critic_model = ActorCriticModel(
            self.hidden_size, action_size, self.device,
            hidsize1=self.hidden_size,
            hidsize2=self.hidden_size
        )

        # Adaptive Learning Rates
        base_lr = self.learning_rate
        
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
            lr=base_lr * 1.0
        )
        
        self.optimizer_critic_head = optim.AdamW(
            self.actor_critic_model.critic.parameters(),
            lr=base_lr * 0.5
        )
        
        self.optimizer_actor = self.optimizer_actor_head
        self.optimizer_critic = self.optimizer_critic_head
        self.optimizer = self.optimizer_actor_head
        self.base_lr_encoder_actor = base_lr * 1.2
        self.base_lr_actor_head = base_lr * 1.0
        self.actor_lr_factor = 1.0
        # Entropy rescue prevents late deterministic collapse around local minima.
        self.entropy_rescue_start_episode = 350      # ⬆️ Activate VERY early (was 700, now immediately!)
        self.entropy_floor = 0.55       # ⬆️ Raise threshold (34% of max)
        self.entropy_recovery_scale = 2.5  # ⬆️ Stronger recovery kick

        self.loss_function = nn.SmoothL1Loss(beta=1.0)
        self.training_step_count = 0

    def _comm_progress(self) -> float:
        start_ep = int(self.comm_reg_start_episode)
        full_ep = int(self.comm_reg_full_episode)
        if full_ep <= start_ep:
            return 1.0
        if self.episode_count <= start_ep:
            return 0.0
        return float(np.clip((self.episode_count - start_ep) / float(full_ep - start_ep), 0.0, 1.0))

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
        self.actor_lr_factor = float(np.clip(factor, self.actor_lr_min_factor, self.actor_lr_max_factor))
        for param_group in self.optimizer_encoder_actor.param_groups:
            param_group['lr'] = self.base_lr_encoder_actor * self.actor_lr_factor
        for param_group in self.optimizer_actor_head.param_groups:
            param_group['lr'] = self.base_lr_actor_head * self.actor_lr_factor

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

    @staticmethod
    def _extract_deadlock_label_from_temporal_state(temporal_state) -> float:
        """Build a robust [0,1] deadlock-risk label from next-state features.

        Works with both 48D DecisionPointObservation and 72D
        HierarchicalRoutesObservation (base 48D + sparse neighbors).
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

        # Base deadlock cues from DecisionPointObservation layout.
        branch_deadlock = _safe(5) + _safe(11) + _safe(17)
        merge_deadlock = _safe(22) + _safe(26)
        local_deadlock = _safe(42)

        # Optional extra cues in 72D hierarchical sparse-neighbor block.
        # Per-neighbor 6D block starts at 48; index +3 == local conflict flag.
        sparse_local = 0.0
        if last_obs.shape[0] >= 72:
            sparse_local = max(_safe(51), _safe(57), _safe(63), _safe(69))

        risk = max(
            local_deadlock,
            min(1.0, 0.5 * branch_deadlock),
            min(1.0, 0.5 * merge_deadlock),
            sparse_local,
        )
        return float(np.clip(risk, 0.0, 1.0))

    def _convert_transitions_to_torch_tensors(self, transitions_array):
        """Convert episode transitions to tensors"""
        state_list, action_list, reward_list, state_next_list, done_list, aux_deadlock_list = [], [], [], [], [], []

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
            done_list.append(1 if done_i else 0)
            aux_deadlock_list.append(float(aux_deadlock_i))

        actions = torch.tensor(action_list, dtype=torch.long).to(self.device)
        rewards = torch.tensor(reward_list, dtype=torch.float).to(self.device) * self.reward_scale
        dones = torch.tensor(done_list, dtype=torch.float).to(self.device)
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
                    values = torch.clamp(values, -5, 5)  # scaled range: reward_scale=0.1 → returns ≈ [-3, +0.3]
                    
                    next_states_critic = self.encoder_critic.forward_batch(state_next_tuples)
                    next_values = torch.squeeze(self.actor_critic_model.critic(next_states_critic), dim=-1)
                    next_values = torch.clamp(next_values, -5, 5)
                    
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
                
                # 📊 Speichere RAW advantage stats BEFORE normalization
                raw_adv_mean = advantages.mean().item()
                raw_adv_std = advantages.std().item()
                
                advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + eps)
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
                    policy_weight_eff = max(policy_weight_eff * 0.35, 0.15)
                    entropy_weight_eff *= 0.4
                    comm_weight_eff *= 1.35
                    hard_spike_batches_total += 1
                    hard_spike_streak += 1
                elif approx_kl > self.ppo_emergency_kl or ratio_soft_viol:
                    hard_spike_streak = max(hard_spike_streak, 1)
                else:
                    hard_spike_streak = 0

                if approx_kl > self.ppo_emergency_kl_hard:
                    policy_weight_eff = max(policy_weight_eff * 0.25, 0.10)
                    entropy_weight_eff *= 0.25
                    comm_weight_eff *= 1.45
                    hard_spike_batches_total += 1
                    hard_spike_streak += 1
                
                loss = \
                    policy_weight_eff * policy_loss_component \
                    + self.weight_loss * value_loss_component \
                    + entropy_weight_eff * entropy_loss_component \
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
                    max_norm=1.0
                )
                
                grad_norm_critic = torch.nn.utils.clip_grad_norm_(
                    list(self.encoder_critic.parameters()) + 
                    list(self.actor_critic_model.critic.parameters()),
                    max_norm=1.0
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
                
                # Update optimizers
                if policy_weight_eff > 0.0:
                    self.optimizer_encoder_actor.step()
                    self.optimizer_actor_head.step()
                self.optimizer_encoder_critic.step()
                self.optimizer_critic_head.step()
                
                self.loss = loss.detach().cpu().numpy()
                
                # 📊 Log metrics for this iteration
                adv_mean = raw_adv_mean  # ⚡ RAW advantage mean (BEFORE normalization)
                adv_std = raw_adv_std    # ⚡ RAW advantage std (BEFORE normalization)
                
                if self.show_progress_bar:
                    progress = current_iteration / total_iterations
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    actor_int = self.encoder_actor.last_comm_intent_mean
                    critic_int = self.encoder_critic.last_comm_intent_mean
                    valid_cnt = self.encoder_actor.last_comm_valid_count + self.encoder_critic.last_comm_valid_count
                    if valid_cnt > 0:
                        wait_mean = 0.5 * (actor_int[0] + critic_int[0])
                        go_mean = 0.5 * (actor_int[1] + critic_int[1])
                        yield_mean = 0.5 * (actor_int[2] + critic_int[2])
                    else:
                        wait_mean, go_mean, yield_mean = 0.0, 0.0, 0.0
                    print(f"\r  [{bar}] Epoch {k_loop+1}/{k_epochs_eff}, Batch {batch_idx+1}/{num_batches} ({progress*100:3.1f}%)", end='')
                    print(f"\t", end='')
                    print(f"| Loss: {loss.item():.4f}", end='')
                    print(f"| P_Loss: {policy_loss_component.item():.4f}", end='')
                    print(f"| V_Loss: {value_loss_component.item():.4f}", end='')
                    print(f"| E_Loss: {entropy_loss_component.item():.4f}", end='')
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

    def end_episode(self, train):
        if train:
            self.accumulated_episodes.append(self.current_episode_memory)
            self.current_episode_memory = EpisodeBuffers()
            
            if self.episode_count % self.train_frequency == 0:
                if self.show_pre_train_debug_msg:
                    print(f"\n🎯 Training with sliding window of {len(self.accumulated_episodes)} episodes...")
                self.train_net_accumulated()
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

    def load(self, filename):
        self.actor_critic_model.load(filename)
        self.encoder_actor.load(filename + "_actor")
        self.encoder_critic.load(filename + "_critic")
        self.optimizer_actor = self._load(self.optimizer_actor, filename + ".optimizer_actor")
        self.optimizer_critic = self._load(self.optimizer_critic, filename + ".optimizer_critic")
        print('{} -> load {} ok'.format(self.get_name(), filename))

    def clone(self):
        policy = MARL_ATTENTION_TEMPORAL_PPOPolicy(self.state_size, self.action_size, self.ppo_parameters)
        policy.actor_critic_model = copy.deepcopy(self.actor_critic_model)
        policy.encoder_actor = copy.deepcopy(self.encoder_actor)
        policy.encoder_critic = copy.deepcopy(self.encoder_critic)
        policy.optimizer_actor = copy.deepcopy(self.optimizer_actor)
        policy.optimizer_critic = copy.deepcopy(self.optimizer_critic)
        return policy

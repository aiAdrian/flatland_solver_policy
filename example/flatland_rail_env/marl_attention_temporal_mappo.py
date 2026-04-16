import copy
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
        
        self._init_weights()
        self.to(self.device)
    
    def _init_weights(self):
        """Kaiming initialization for LeakyReLU"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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
        else:
            # No opponents → only own temporal context
            context = self_temporal_context
        
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
            else:
                context = self_ctx
            
            final_embeddings.append(self.output_proj(context))
        
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


# =============================================================================
# PPO PARAMETERS (extended with temporal_window)
# =============================================================================

MARL_ATTENTION_TEMPORAL_MAPPO_Param = namedtuple('MARL_ATTENTION_TEMPORAL_MAPPO_Param',
                            ['hidden_size', 'batch_size', 'learning_rate',
                             'discount', 'gae_lambda', 'use_gpu',
                             'max_episodes_in_training_memory', 'batch_fraction', 'k_epochs',
                             'max_batches_per_training', 'temporal_window'])


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
        else:
            self.hidden_size = 256
            self.learning_rate = 5.0e-3
            self.discount = 0.99
            self.batch_size = 128  # Back to baseline
            self.temporal_window = 3

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
            
        self.surrogate_eps_clip = 0.15  # Etwas mehr Flexibilität für Policy-Updates
        self.weight_loss = 0.4  # Weniger Fokus auf Value Loss
        self.weight_entropy = 0.02  # Mehr Exploration
        self.weight_policy = 1.0
        self.gae_lambda = self.ppo_parameters.gae_lambda if self.ppo_parameters else 0.95 

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
        print("\n🚀 Creating Temporal Transformer Encoders:")
        print(f"   - obs_dim: {state_size}")
        print(f"   - hidden_dim: {self.hidden_size}")
        print(f"   - temporal_window: {self.temporal_window}")
        print(f"   - num_heads: {self.num_heads}")
        
        self.encoder_actor = TemporalTransformerEncoder(
            obs_dim=state_size,
            hidden_dim=self.hidden_size,
            num_heads=self.num_heads,
            temporal_window=self.temporal_window,
            device=self.device
        )
        
        self.encoder_critic = TemporalTransformerEncoder(
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
            lr=base_lr * 2.5  # ⚡ Reduced from 3.0 for smoother updates
        )
        
        self.optimizer_actor_head = optim.AdamW(
            self.actor_critic_model.actor.parameters(),
            lr=base_lr * 2.5  # ⚡ Reduced from 3.0
        )
        
        self.optimizer_encoder_critic = optim.AdamW(
            self.encoder_critic.parameters(),
            lr=base_lr * 1.25  # ⚡ Reduced from 1.5
        )
        
        self.optimizer_critic_head = optim.AdamW(
            self.actor_critic_model.critic.parameters(),
            lr=base_lr * 0.65  # ⚡ Reduced from 0.75
        )
        
        self.optimizer_actor = self.optimizer_actor_head
        self.optimizer_critic = self.optimizer_critic_head
        self.optimizer = self.optimizer_actor_head

        self.loss_function = nn.MSELoss()
        self.training_step_count = 0

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
        transition = (state, action, reward, next_state, done)
        self.current_episode_memory.push_transition(handle, transition)

    def _convert_transitions_to_torch_tensors(self, transitions_array):
        """Convert episode transitions to tensors"""
        state_list, action_list, reward_list, state_next_list, done_list = [], [], [], [], []

        for transition in transitions_array:
            state_i, action_i, reward_i, state_next_i, done_i = transition

            state_list.append(state_i)
            action_list.append(action_i)
            reward_list.append(reward_i)
            state_next_list.append(state_next_i)
            done_list.append(1 if done_i else 0)

        actions = torch.tensor(action_list, dtype=torch.long).to(self.device)
        rewards = torch.tensor(reward_list, dtype=torch.float).to(self.device)
        dones = torch.tensor(done_list, dtype=torch.float).to(self.device)

        return state_list, actions, rewards, state_next_list, dones
    
    def _compute_gae(self, rewards, values, dones, next_values):
        """Compute Generalized Advantage Estimation (GAE)"""
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
                state_tuples, actions, rewards, state_next_tuples, dones = \
                    self._convert_transitions_to_torch_tensors(agent_episode_history)
                
                # Compute GAE (encoder calls are batched internally)
                with torch.no_grad():
                    states_critic = self.encoder_critic.forward_batch(state_tuples)
                    values = torch.squeeze(self.actor_critic_model.critic(states_critic), dim=-1)
                    values = torch.clamp(values, -10, 10)  # ⚡ Clip values!
                    
                    next_states_critic = self.encoder_critic.forward_batch(state_next_tuples)
                    next_values = torch.squeeze(self.actor_critic_model.critic(next_states_critic), dim=-1)
                    next_values = torch.clamp(next_values, -10, 10)  # ⚡ Clip values!
                    
                    traj_gae_advantages, traj_gae_returns = self._compute_gae(
                        rewards, values, dones, next_values
                    )
                
                episode_state_tuples.extend(state_tuples)
                episode_actions.append(actions)
                episode_advantages.append(traj_gae_advantages)
                episode_returns.append(traj_gae_returns)
            
            if len(episode_state_tuples) > 0:
                episode_data.append((
                    episode_state_tuples,
                    torch.cat(episode_actions, dim=0),
                    torch.cat(episode_advantages, dim=0),
                    torch.cat(episode_returns, dim=0)
                ))
        
        if len(episode_data) == 0:
            print("⚠️ No transitions to train on!")
            return
        
        # Concatenate all episode data
        all_state_tuples = []
        all_actions = []
        all_gae_advantages = []
        all_gae_returns = []
        
        # ⚡ NEW: Track episode indices for recency-based sampling
        episode_sample_weights = []
        
        for ep_idx, (ep_states, ep_actions, ep_advantages, ep_returns) in enumerate(episode_data):
            # Get episode memory to check done status
            episode_memory = self.accumulated_episodes[ep_idx]
            
            # Check if episode has successful completion (any agent done=True)
            episode_has_success = False
            for handle in range(len(episode_memory)):
                agent_history = episode_memory.get_transitions(handle)
                if len(agent_history) > 0:
                    # Check last transition
                    _, _, _, _, done = agent_history[-1]
                    if done:
                        episode_has_success = True
                        break
            
            all_state_tuples.extend(ep_states)
            all_actions.append(ep_actions)
            all_gae_advantages.append(ep_advantages)
            all_gae_returns.append(ep_returns)
            
            # Combined weight: recency + success bonus
            recency_weight = (ep_idx + 1) / len(episode_data)
            success_bonus = 5.0 if episode_has_success else 1.0  # 5× weight for successful episodes!
            combined_weight = recency_weight * success_bonus
            
            episode_sample_weights.extend([combined_weight] * len(ep_states))
        
        all_actions = torch.cat(all_actions, dim=0)
        all_gae_advantages = torch.cat(all_gae_advantages, dim=0)
        all_gae_returns = torch.cat(all_gae_returns, dim=0)
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
        
        if self.show_pre_train_debug_msg:
            print(f"📦 Using {samples_to_use}/{total_samples} samples ({samples_to_use/total_samples*100:.1f}%)")
            print(f"📦 Batch Config: {num_batches} batches (batch_size={self.batch_size}) over {int(self.K_epoch)} epochs")
        
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
        
        total_iterations = int(self.K_epoch) * num_batches
        current_iteration = 0
        
        if self.show_progress_bar:
            print("")
        
        # Training epochs
        for k_loop in range(int(self.K_epoch)):
            # 🎯 STANDARD PPO: Update old_logprobs nach jedem Epoch (außer dem ersten)
            if k_loop > 0:
                if self.show_pre_train_debug_msg:
                    print(f"\n🔄 Re-computing old_logprobs for Epoch {k_loop+1}...")
                
                with torch.no_grad():
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
                    print(f"✅ Updated (mean={all_old_logprobs.mean().item():.4f})")
            
            # Shuffle data each epoch for better training
            indices = torch.randperm(samples_to_use)
            
            for batch_idx in range(num_batches):
                current_iteration += 1
                
               
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, samples_to_use)
                batch_indices = indices[start_idx:end_idx]
                
                batch_state_tuples = [all_state_tuples[i] for i in batch_indices]
                batch_actions = all_actions[batch_indices]
                batch_gae_advantages = all_gae_advantages[batch_indices]
                batch_gae_returns = all_gae_returns[batch_indices]
                batch_old_logprobs = all_old_logprobs[batch_indices]  # 🎯 Pre-computed!

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
                
                # 🔍 DEBUG: Verify old_logprobs are different from new ones (NUR Epoch 1)
                if self.show_debug_msg:
                    if k_loop == 0 and batch_idx < 3:  # Erste Epoch, erste 3 Batches
                        diff_mean = (logprobs - batch_old_logprobs).abs().mean().item()
                        ratio_raw = torch.exp(logprobs - batch_old_logprobs).mean().item()
                        print(f"🔍 E1 B{batch_idx}: Diff={diff_mean:.6f} Ratio_raw={ratio_raw:.4f}", flush=True)
                
                state_values = torch.squeeze(self.actor_critic_model.critic(states_critic), dim=-1)
                
                # PPO ratios with AGGRESSIVE clipping to prevent explosion
                ratios = torch.exp(logprobs - batch_old_logprobs)
                ratios = torch.clamp(ratios, 0.5, 2.0)  # ⚡ Limit ratio range BEFORE advantage multiplication
                
                # Normalize advantages
                advantages = batch_gae_advantages
                eps = 1e-8
                
                # 📊 Speichere RAW advantage stats BEFORE normalization
                raw_adv_mean = advantages.mean().item()
                raw_adv_std = advantages.std().item()
                
                advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + eps)
                advantage_scale_factor = 0.3  # ⚡ REDUZIERT: Sanftere Updates → verhindert Catastrophic Forgetting!
                advantages = advantages_normalized * advantage_scale_factor

                # PPO loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.surrogate_eps_clip, 
                                   1.0 + self.surrogate_eps_clip) * advantages

                policy_loss_component = -torch.min(surr1, surr2).mean()
                value_loss_component = self.loss_function(state_values, batch_gae_returns)
                entropy_loss_component = -dist_entropy.mean()
                
                loss = \
                    self.weight_policy * policy_loss_component \
                    + self.weight_loss * value_loss_component \
                    + self.weight_entropy * entropy_loss_component

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
                
                # Gradient clipping - ausbalanciert für Stabilität und Lernen
                grad_norm_actor = torch.nn.utils.clip_grad_norm_(
                    list(self.encoder_actor.parameters()) + 
                    list(self.actor_critic_model.actor.parameters()),
                    max_norm=0.5  # ⚡ Erhöht: Erlaubt größere Updates ohne Explosion
                )
                
                grad_norm_critic = torch.nn.utils.clip_grad_norm_(
                    list(self.encoder_critic.parameters()) + 
                    list(self.actor_critic_model.critic.parameters()),
                    max_norm=0.5  # ⚡ Erhöht: Erlaubt größere Updates ohne Explosion
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
                self.optimizer_encoder_actor.step()
                self.optimizer_actor_head.step()
                self.optimizer_encoder_critic.step()
                self.optimizer_critic_head.step()
                
                self.loss = loss.detach().cpu().numpy()
                
                # 📊 Log metrics for this iteration
                ratio_mean = ratios.mean().item()
                adv_mean = raw_adv_mean  # ⚡ RAW advantage mean (BEFORE normalization)
                adv_std = raw_adv_std    # ⚡ RAW advantage std (BEFORE normalization)
                
                if self.show_progress_bar:
                    progress = current_iteration / total_iterations
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r  [{bar}] Epoch {k_loop+1}/{int(self.K_epoch)}, Batch {batch_idx+1}/{num_batches} ({progress*100:3.1f}%)", end='')
                    print(f"\t", end='')
                    print(f"| Loss: {loss.item():.4f}", end='')
                    print(f"| P_Loss: {policy_loss_component.item():.4f}", end='')
                    print(f"| V_Loss: {value_loss_component.item():.4f}", end='')
                    print(f"| E_Loss: {entropy_loss_component.item():.4f}", end='')
                    print(f"| Adv: {adv_mean:.2f}±{adv_std:.2f}", end='')  # ⚡ RAW advantage (mean±std BEFORE norm)
                    print(f"| Ratio: {ratio_mean:.4f}", end='')
                    print(f"| Grad_Norm: {grad_norm:.4f}", end='')
                    print("", end='', flush=True)

        if self.show_progress_bar:
            print("\n")  # New line after progress bar
        
        if not hasattr(self, 'training_step'):
            self.training_step = 0
        self.training_step += 1

        # Clear data
        del all_state_tuples
        del all_actions
        del all_gae_advantages
        del all_gae_returns
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

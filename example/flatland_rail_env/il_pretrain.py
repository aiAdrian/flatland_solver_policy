"""
IL — Phase B: Behavior Cloning of the DeciderNetwork
====================================================

Loads the demonstrations produced by `il_collect_demos.py` and trains the
DeciderNetwork to imitate the teacher actions via masked cross-entropy.

The trained `state_dict` is saved to `il_bc_checkpoint.pt`. The MARL training
script (`marl_attention_temporal_il.py`) then loads it via
`DeciderPPOPolicy.load(...)` before starting PPO fine-tuning.

Why masked CE
-------------
The PPO update applies an action mask to the logits (illegal actions get
-1e9). The behavioral target distribution must be expressed in the SAME
masked-logit space, otherwise BC and PPO operate on different probability
manifolds. We mask the logits before computing log-softmax for the
cross-entropy loss.

Training scheme
---------------
* Standard mini-batch supervised learning over (state, teacher_action, mask)
  triplets.
* AdamW optimizer with cosine LR decay.
* Gradient clipping at 0.5 (same as PPO update for consistency).
* No entropy regularisation — the mask alone provides enough action diversity.
* Validation split: last 10% of transitions, never seen during weight updates.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from decider_policy import (
    DeciderPPOPolicy,
    _states_batch_to_tensors,
)


# ----------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------
DEMO_INPUT_FILE = 'il_demos.pkl'
CHECKPOINT_OUTPUT = 'il_bc_checkpoint.pt'

NUM_EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP_NORM = 0.5
VALIDATION_FRACTION = 0.10


def _build_minibatch_tensors(
    states: List, actions: np.ndarray, masks: np.ndarray, idx: np.ndarray,
    state_size: int, device: torch.device,
):
    """Build the same (self_seq, neigh, nmask) tensors that the PPO update
    pipeline uses, so the network input format is identical."""
    sub_states = [states[i] for i in idx]
    self_seq, neigh, nmask = _states_batch_to_tensors(
        sub_states, state_size, device
    )
    actions_t = torch.tensor(actions[idx], dtype=torch.long, device=device)
    masks_t = torch.tensor(masks[idx], dtype=torch.float32, device=device)
    return self_seq, neigh, nmask, actions_t, masks_t


def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with -inf on illegal actions (Huang & Ontañón 2022)."""
    masked_logits = logits.masked_fill(mask < 0.5, -1e9)
    return nn.functional.cross_entropy(masked_logits, targets)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor,
              mask: torch.Tensor) -> float:
    masked = logits.masked_fill(mask < 0.5, -1e9)
    pred = masked.argmax(dim=-1)
    return float((pred == targets).float().mean().item())


def pretrain() -> str:
    if not os.path.exists(DEMO_INPUT_FILE):
        raise FileNotFoundError(
            f"{DEMO_INPUT_FILE} not found. Run il_collect_demos.py first."
        )

    print("=" * 80)
    print("IL — Behavior Cloning of DeciderNetwork")
    print("=" * 80)
    with open(DEMO_INPUT_FILE, 'rb') as f:
        payload = pickle.load(f)

    transitions = payload['transitions']
    state_size = int(payload['state_size'])
    action_size = int(payload['action_size'])
    temporal_window = int(payload['temporal_window'])
    print(f"Loaded {len(transitions):,} transitions  "
          f"state_size={state_size}  action_size={action_size}  T={temporal_window}")
    for s in payload.get('stats', []):
        print(f"  {s['phase']:<14s}  done_rate={s['done_rate']:.2%}  "
              f"transitions={s['transitions']:>6d}")

    # Unpack into parallel arrays. We keep `states` as a Python list because
    # each entry is a temporal sequence of variable internal structure.
    states = [t[0] for t in transitions]
    actions = np.array([t[1] for t in transitions], dtype=np.int64)
    masks = np.stack([t[2] for t in transitions], axis=0).astype(np.float32)

    # Action distribution diagnostic.
    counts = np.bincount(actions, minlength=action_size)
    total = max(1, len(actions))
    print("\nTeacher action distribution:")
    action_names = ['DO_NOTHING', 'MOVE_LEFT', 'MOVE_FORWARD', 'MOVE_RIGHT', 'STOP_MOVING']
    for a in range(action_size):
        name = action_names[a] if a < len(action_names) else f'action_{a}'
        print(f"  {a} {name:<14s}  {counts[a]:>7,d}  ({counts[a]/total:.1%})")

    # Train / validation split (deterministic).
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(len(states))
    n_val = max(1, int(VALIDATION_FRACTION * len(states)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"\nSplit: {len(train_idx):,} train  /  {len(val_idx):,} val")

    # Build a fresh DeciderPPOPolicy. Its network is what we train.
    device = torch.device('cpu')
    policy = DeciderPPOPolicy(
        state_size=state_size,
        action_size=action_size,
        learning_rate=LEARNING_RATE,
        max_episodes_in_memory=1,
        train_frequency=10 ** 9,
        temporal_window=temporal_window,
    )
    net = policy.net

    # Use a fresh AdamW for BC (the policy's optimizer is configured for PPO
    # with a smaller LR; we want a faster supervised LR here).
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.1
    )

    print(f"\nStarting BC: {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"lr={LEARNING_RATE}")
    print("-" * 80)
    for epoch in range(1, NUM_EPOCHS + 1):
        net.train()
        rng.shuffle(train_idx)
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_mb = 0
        t_epoch = time.perf_counter()

        for start in range(0, len(train_idx), BATCH_SIZE):
            mb_idx = train_idx[start:start + BATCH_SIZE]
            self_seq, neigh, nmask, actions_t, masks_t = _build_minibatch_tensors(
                states, actions, masks, mb_idx, state_size, device
            )
            logits, _values, _aux = net(self_seq, neigh, nmask)
            loss = _masked_cross_entropy(logits, actions_t, masks_t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_acc += _accuracy(logits.detach(), actions_t, masks_t)
            n_mb += 1

        scheduler.step()
        train_loss = epoch_loss / max(1, n_mb)
        train_acc = epoch_acc / max(1, n_mb)

        # Validation pass.
        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            n_val_mb = 0
            for start in range(0, len(val_idx), BATCH_SIZE):
                mb_idx = val_idx[start:start + BATCH_SIZE]
                self_seq, neigh, nmask, actions_t, masks_t = _build_minibatch_tensors(
                    states, actions, masks, mb_idx, state_size, device
                )
                logits, _v, _aux = net(self_seq, neigh, nmask)
                val_loss += float(_masked_cross_entropy(logits, actions_t, masks_t).item())
                val_acc += _accuracy(logits, actions_t, masks_t)
                n_val_mb += 1
            val_loss /= max(1, n_val_mb)
            val_acc /= max(1, n_val_mb)

        elapsed = time.perf_counter() - t_epoch
        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"epoch {epoch:>3d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"lr={lr_now:.2e}  t={elapsed:5.1f}s"
        )

    # Save the BC checkpoint using the policy's own save() method so the
    # PPO trainer can later load it via solver.load_policy() / policy.load().
    policy.save(CHECKPOINT_OUTPUT)
    size_kb = os.path.getsize(CHECKPOINT_OUTPUT) / 1024
    print(f"\nSaved {CHECKPOINT_OUTPUT} ({size_kb:.1f} KB).")
    return CHECKPOINT_OUTPUT


if __name__ == "__main__":
    pretrain()

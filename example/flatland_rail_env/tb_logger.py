# =============================================================================
# tb_logger.py
# -----------------------------------------------------------------------------
# Unified TensorBoard logging for the MARL training/eval pipeline.
#
# Design:
#   - All runs go under runs/<timestamp>_<mode>_<policy>/
#   - Consistent tag scheme so eval/train/bc are directly comparable:
#       eval/done_rate, eval/deadlock_rate, eval/episode_len, eval/total_reward
#       train/done_rate_50, train/reward_50, train/eps
#       ppo/v_loss, ppo/p_loss, ppo/entropy, ppo/kl, ppo/ratio, ppo/clip_frac
#       bc/loss, bc/accuracy
#
# Usage:
#   logger = TBLogger(run_name="eval_dla")
#   logger.log_eval_episode(ep_idx, done_rate, deadlock_rate, ep_len, reward)
#   logger.close()
# =============================================================================

import os
import time
from typing import Dict, Any, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    SummaryWriter = None


def _runs_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "runs")


class TBLogger:
    """Thin wrapper around SummaryWriter with consistent tag scheme."""

    def __init__(self, run_name: str, enabled: bool = True):
        self.run_name = run_name
        self.enabled = enabled and _TB_AVAILABLE
        self.writer: Optional[SummaryWriter] = None
        self.run_dir = ""

        if self.enabled:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(_runs_dir(), f"{ts}_{run_name}")
            os.makedirs(self.run_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.run_dir)
            print(f"[TBLogger] Logging to {self.run_dir}")
        else:
            print(f"[TBLogger] Disabled (TB available={_TB_AVAILABLE}, enabled={enabled})")

    def log_eval_episode(
        self,
        step: int,
        done_rate: float,
        deadlock_rate: float,
        episode_len: float,
        total_reward: float,
    ):
        if not self.enabled:
            return
        self.writer.add_scalar("eval/done_rate", float(done_rate), step)
        self.writer.add_scalar("eval/deadlock_rate", float(deadlock_rate), step)
        self.writer.add_scalar("eval/episode_len", float(episode_len), step)
        self.writer.add_scalar("eval/total_reward", float(total_reward), step)

    def log_eval_summary(
        self,
        done_rate: float,
        deadlock_rate: float,
        episode_len: float,
        total_reward: float,
        n_episodes: int,
    ):
        """Log final mean values once at the end of an eval run."""
        if not self.enabled:
            return
        self.writer.add_scalar("eval_summary/done_rate", float(done_rate), n_episodes)
        self.writer.add_scalar("eval_summary/deadlock_rate", float(deadlock_rate), n_episodes)
        self.writer.add_scalar("eval_summary/episode_len", float(episode_len), n_episodes)
        self.writer.add_scalar("eval_summary/total_reward", float(total_reward), n_episodes)

    def log_train_episode(
        self,
        episode: int,
        done_50: float,
        reward_50: float,
        eps: float,
        ppo_stats: Dict[str, float],
    ):
        if not self.enabled:
            return
        self.writer.add_scalar("train/done_rate_50", float(done_50), episode)
        self.writer.add_scalar("train/reward_50", float(reward_50), episode)
        self.writer.add_scalar("train/eps", float(eps), episode)

        if ppo_stats:
            self.writer.add_scalar("ppo/v_loss", float(ppo_stats.get("v_loss", 0.0)), episode)
            self.writer.add_scalar("ppo/p_loss", float(ppo_stats.get("p_loss", 0.0)), episode)
            self.writer.add_scalar("ppo/entropy", float(ppo_stats.get("ent", 0.0)), episode)
            self.writer.add_scalar("ppo/kl", float(ppo_stats.get("kl", 0.0)), episode)
            self.writer.add_scalar("ppo/ratio", float(ppo_stats.get("ratio", 1.0)), episode)
            self.writer.add_scalar("ppo/clip_frac", float(ppo_stats.get("clip_frac", 0.0)), episode)

    def log_bc_epoch(self, epoch: int, loss: float, accuracy: float):
        if not self.enabled:
            return
        self.writer.add_scalar("bc/loss", float(loss), epoch)
        self.writer.add_scalar("bc/accuracy", float(accuracy), epoch)

    def log_scalar(self, tag: str, value: float, step: int):
        """Generic escape hatch for ad-hoc logging."""
        if not self.enabled:
            return
        self.writer.add_scalar(tag, float(value), step)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

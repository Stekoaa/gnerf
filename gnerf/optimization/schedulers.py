import torch

from dataclasses import dataclass, field


@dataclass
class SchedulerConfig:
    milestones: list = field(default_factory=lambda: [0.5, 0.75, 0.9])
    """Milestones for the learning rate scheduler."""
    gamma: float = 0.33
    """Gamma for the learning rate scheduler."""


def initialize_scheduler(config, optimizer):
    return torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=100),
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(m * config.max_steps) for m in config.scheduler.milestones], gamma=config.scheduler.gamma),
        ]
    )
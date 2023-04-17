# Flatland Solver

## Solving the Flatland problem

The policy abstraction allows to solve the Flatland problem
in a very accurate way. The user mustn't care about the Flatland integration.

The goal of this library is that user just extend the abstract policy, create
a environment and enable rendering if required. If no reinforcement learning is
used there observation can be replaced by a dummy observation.

### Example

Environments which can be used and are tested:

- openAI cartpool
    - Learning policy
        - DDDQNPolicy
        - PPOPolicy
- Flatland
    - Learning policy
        - DDDQNPolicy
        - PPOPolicy
    - Heuristic policy
        - DeadLockAvoidancePolicy

Training / quality logging is done with tensorboard. Navigate to the example folder
and call ``tensorboard --logdir runs``

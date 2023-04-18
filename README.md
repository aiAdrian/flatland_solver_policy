# Flatland Solver

## Solving the Flatland problem

Policy abstraction makes it possible to solve the flatland problem in a very accurate way. The user does not have to care about flatland integration.

The goal of this library is to allow the user to easily extend the abstract policy to solve the flatland problem. 
The user just needs to create an environment and a policy (solver) to solve the flatland problem. Rendering can be added as an option.
The policy can be a learned one or a manual written heuristic or even any other solver/idea. 
The observation can as well exchanged through the abstraction.
If reinforcement learning is not used, the observation can be replaced by the dummy observation.


### Example

Environments which can be used and are tested:

- [openAI cartpool](https://github.com/aiAdrian/flatland_solver_policy/blob/main/example/learning_policy_cartpool_example.py)
    - Learning policy
        - DDDQNPolicy
        - PPOPolicy
- [Flatland Problem Solver](https://github.com/aiAdrian/flatland_solver_policy/blob/main/example/learning_policy_flatland_example.py)
    - Learning policy
        - DDDQNPolicy
        - PPOPolicy
    - Heuristic policy
        - DeadLockAvoidancePolicy

Training / quality logging is done with tensorboard. Navigate to the example folder
and call ``tensorboard --logdir runs``

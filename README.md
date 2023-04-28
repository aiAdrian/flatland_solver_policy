# Flatland Solver

## Solving the Flatland problem

Policy abstraction makes it possible to solve the flatland problem in a very accurate way. The user does not have to
care about flatland integration.

The goal of this library is to allow the user to easily extend the abstract policy to solve the flatland problem.
The user just needs to create an environment and a policy (solver) to solve the flatland problem. Rendering can be added
as an option.
The policy can be a learned one or a manual written heuristic or even any other solver/idea.
The observation can as well exchanged through the abstraction.
If reinforcement learning is not used, the observation can be replaced by the dummy observation.

## One solver for multiple environments and policy

The **[BaseSolver](https://github.com/aiAdrian/flatland_solver_policy/blob/main/solver/base_solver.py)** requires an
**Environment** and a **[Policy](https://github.com/aiAdrian/flatland_solver_policy/blob/main/policy/policy.py)**. The
BaseRenderer can be optionally enabled.

```mermaid
graph TD;
    Environment ---> |required| BaseSolver;
    Policy ---> |required| BaseSolver;
    BaseRenderer ---> |optional| BaseSolver;
```

### Class diagram

The class diagram shows the most important classes and their dependencies, generalization and specialization of
Environment, BaseSolver and Policy.

```mermaid
classDiagram
       
    BaseSolver o-- Policy
    BaseSolver o-- BaseRenderer
    BaseSolver o-- Environment

    Environment <|-- cartpole : package(Gymnasium)
    Environment <|-- RailEnv : package(Flatland)
    RailEnv <|-- FlatlandDynamics : package(Flatland Railway Extension)
    
    BaseSolver <|-- FlatlandSolver
    BaseSolver <|-- CartpoleSolver
    FlatlandSolver <|-- FlatlandDynamicsSolver
    
    Policy  <|-- HeuristicPolicy

    HeuristicPolicy  <|-- DeadLockAvoidancePolicy : Flatland

    Policy  <|-- LearningPolicy

    LearningPolicy <|-- PPOPolicy : Environment
    LearningPolicy <|-- DDDQNPolicy : Environment

    class Environment {
        env.reset() state, info
        step(action) state_next, reward, terminal, info = 
    }

    class Policy{
      get_name()* str
      start_episode(train: bool)
      start_step(train: bool)
      start_act(handle: int, train: bool)
      act(handle, state, eps=0.)*
      end_act(handle: int, train: bool)
      step(handle, state, action, reward, next_state, done)*
      end_step(train: bool)
      end_episode(train: bool)
      load_replay_buffer(filename)
      test()
      reset(env: RailEnv)
      clone()* Policy
      save(filename)*
      load(filename)* 
    }

    class BaseRenderer {
        reset():
        render(episode, step, terminal)
    }

    class BaseSolver {
        get_name()* str
        set_renderer(renderer: BaseRenderer)
        set_policy(policy: Policy)
        activate_rendering()
        deactivate_rendering()
        render(episode: int, step: int, terminal: bool)
        reset() state, info
        run_step(env, policy, state, eps, info, training_mode) state_next, tot_reward, all_terminal, info
        update_state(state_next) state
        before_step_starts() bool
        after_step_ends() bool
        run_internal_episode( episode, env, policy, state, eps, info, training_mode) tot_reward[int]
        before_episode_starts()
        after_episode_ends()
        run_episode(episode, env, policy, eps, training_mode) tot_reward[int]
        do_training(max_episodes=2000)
        save_policy(filename: str)
        load_policy(filename: str)
    }
``` 

### Solver

The following flowchart explains the main flow of Solver.do_training() and the respective calls to the abstract policy
and environment methods. The implementation of the Policy significantly controls the environmental behavior. The
environment must have implemented the reset and step method. The reset method returns the initial state (observation)
and an info dict. The step method needs a dict with all actions (for each agent one action) and returns the next state (
observation), reward, done signals (terminate) and an info.

```mermaid
flowchart TD
    Env(Environment)
    Policy(Policy)
    Solver(BaseSolver)
    Policy ~~~ Root
    Root((do_training))
    Root ---> |training: on| C{Episode loop:\n more episodes?}
    C --------------> |no| End(( done ))
    C --> |yes : run_episode| D0(env.reset)
    D0 --> D2(policy.start_episode)
    D2 --> |run_internal_episode| E{Step loop:\n more steps?}
    F1 --> |collect actions| G{Agent loop:\n more agents?}
    E ---> |yes : run_step| F1(policy.start_step)
    G --->  |yes : get action for agent| G1(policy.start_act)
    G1 --> G2(policy.act)
    G2 --> G3(policy.end_act)
    G3 --> G
    G --> |no : all actions collected|F2(env.step)
    F2 --> F3(policy.step)
    F3 --> F4(policy.end_step)
    F6 --> E 
    F4 --> F6(render)

    E ---> |no| D4(policy.end_episode)
    D4 --> C

    
    style D2 fill:#ffe,stroke:#333,stroke-width:1px 
    style F1 fill:#ffe,stroke:#333,stroke-width:1px 
    style D4 fill:#ffe,stroke:#333,stroke-width:1px
    style G1 fill:#ffe,stroke:#333,stroke-width:1px 
    style G2 fill:#ffe,stroke:#333,stroke-width:1px 
    style G3 fill:#ffe,stroke:#333,stroke-width:1px 
    style F3 fill:#ffe,stroke:#333,stroke-width:1px 
    style F4 fill:#ffe,stroke:#333,stroke-width:1px 

    style Policy fill:#ffe,stroke:#333,stroke-width:1px 
    style D0 fill:#fcc,stroke:#333,stroke-width:1px,color:#300
    style F2 fill:#fcc,stroke:#333,stroke-width:1px,color:#300
    style Env fill:#fcc,stroke:#333,stroke-width:1px,color:#300        
```

### [Examples](https://github.com/aiAdrian/flatland_solver_policy/tree/main/example                                                  )

#### Flatland Dynamics


```python
def create_observation_builder() -> ObservationBuilder:
    return FlattenTreeObsForRailEnv(
        max_depth=3,
        predictor=ShortestPathPredictorForRailEnv(max_depth=50)
    )
```  


```python
def create_environment(obs_builder_object: Union[ObservationBuilder, None] = None,
                       number_of_agents: int = 10):
    flatland_environment_helper = FlatlandEnvironmentHelper(rail_env=FlatlandDynamics,
                                                            number_of_agents=number_of_agents,
                                                            random_seed=2341,
                                                            obs_builder_object=obs_builder_object)

    action_space = flatland_environment_helper.get_rail_env().action_space[0]

    obs_states, _ = flatland_environment_helper.get_rail_env().reset()
    observation_space = len(obs_states[0])

    return flatland_environment_helper.get_rail_env(), observation_space, action_space
```                                                                

```python
env, obs_space, act_space = create_environment(create_observation_builder())
solver = FlatlandDynamicsSolver(env)

solver.set_policy(PPOPolicy(obs_space, act_space))
solver.do_training(max_episodes=1000)
```                                                                

#### Environments

Environments which are tested:

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium):
   - [cartpole](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py)
- [Flatland](https://github.com/flatland-association/flatland-rl)
   - [RailEnv](https://github.com/flatland-association/flatland-rl/blob/main/flatland/envs/rail_env.py)
- [Flatland Railway Extension](https://github.com/aiAdrian/flatland_railway_extension)
   - [FlatlandDynamics](https://github.com/aiAdrian/flatland_railway_extension/blob/master/flatland_railway_extension/environments/FlatlandDynamics.py)

| Policy                    | cartpole | RailEnv    | Flatland Dynamics |
|---------------------------|----------|------------|-------------------|
| CartpoleAnalyticalPolicy  | **yes**  | no         | no                |
| DDDQNPolicy               | **yes**  | **yes**    | **yes**           |
| PPOPolicy                 | **yes**  | **yes**    | **yes**           |
| DeadLockAvoidancePolicy   | no       | **yes**    | **yes**           |

### Tensorboard 
Training / quality logging is done with tensorboard. Navigate to the example folder
and call ``tensorboard --logdir runs``

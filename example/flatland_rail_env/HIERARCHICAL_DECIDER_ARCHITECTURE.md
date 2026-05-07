# Hierarchische Decider-Architektur mit Spezialisten-Sub-Agents

> **Ziel:** Eine hierarchische MARL-Policy für Flatland, in der spezialisierte Sub-Module
> (Routing, Merging, Deadlock, Communication) Embeddings + Confidence-Scores liefern,
> die ein **Decider** zu einer einzigen, Flatland-kompatiblen Aktion fusioniert.
>
> **Constraints:**
> - Aktionsraum bleibt **5-way Flatland** (`DO_NOTHING`, `MOVE_LEFT`, `MOVE_FORWARD`, `MOVE_RIGHT`, `STOP_MOVING`)
> - **Keine** `handle` / `agent_id` / Position als Adressierung in der Kommunikation
> - Cross-Agent-Info nur über lernbare **Inhalts-Embeddings** + 3-Bit-Relevanz-Klassen
> - Maximal **4 Routenkandidaten** pro Agent
> - Sparse Attention auf **K=4** relevanteste Nachbarn

---

## 1. Verknüpfung Flatland-Züge ↔ Agents

In Flatland ist **jeder Zug genau ein Agent**:

| Flatland-Konzept | RL-Agent-Konzept |
|---|---|
| `env.agents[i]` (RailAgent) | Agent `i` mit eigener Observation, Policy, Action |
| `agent.position`, `agent.direction` | Lokaler Zustand auf Schienennetz |
| `agent.target` | Ziel auf Schienennetz |
| `env.step({i: action_i, ...})` | Multi-Agent-Step mit gemeinsamem Aktions-Dict |
| `agent.handle` (Index) | **Wird intern verwendet, aber NICHT als Comm-Adresse** |

### Wichtige Eigenschaften

- **Decentralized Execution, Centralized Training**: Jeder Agent hat seine **eigene Observation** und sein eigenes Forward-Pass-Embedding, aber **alle Agents teilen die Netzwerk-Gewichte** (parameter sharing). Critic kann optional zentralisiert sein.
- **Synchron**: Alle Agents agieren in jedem Step gleichzeitig; die Env wartet auf das vollständige Action-Dict.
- **Comm via Buffer**: Da Agents zeitgleich entscheiden, **lesen sie die `global_metric` der Vorrunde (`t-1`)** der anderen, nicht die aktuelle. Das macht Comm asynchron-kausal und vermeidet Zirkularität.

```
Pro Env-Step t:
  1. ENV gibt allen Agents ihre Observations (parallel)
  2. Jeder Agent A liest aus dem Comm-Buffer:
       global_metrics[Andere][t-1, t-2, t-3]   ← schon vorhanden, KEIN Sync nötig
  3. Jeder Agent berechnet eigenständig:
       Specialists → Fusion → Decider → action_t  + global_metric_out_t
  4. Comm-Buffer wird mit allen global_metric_out_t aktualisiert
  5. ENV.step({0: action_0, 1: action_1, ...})
```

---

## 2. Parameter Sharing – Ein globaler PPO-Agent für alle Züge

> **Kernprinzip:** Es gibt **genau ein** Netzwerk (ein Set Gewichte θ).
> **Alle N Flatland-Züge teilen dieses Netzwerk** (Standard-MARL-Pattern für homogene Agents,
> z.B. MAPPO, IPPO).

```
              ┌──────────────────────────────┐
              │   EIN gemeinsames Netzwerk   │
              │   (alle Specialists +        │
              │    Decider + Critic)         │
              │                              │
              │   θ = {θ_routing, θ_merging, │
              │        θ_deadlock, θ_comm,   │
              │        θ_decider}            │
              └──────────────┬───────────────┘
                             │
              ┌──────────────┼──────────────┬──────────────┐
              ▼              ▼              ▼              ▼
         ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
         │Agent 0 │    │Agent 1 │    │Agent 2 │    │Agent N │
         │(Zug 0) │    │(Zug 1) │    │(Zug 2) │    │(Zug N) │
         └────────┘    └────────┘    └────────┘    └────────┘
         eigene obs    eigene obs    eigene obs    eigene obs
         eigene LSTM-  eigene LSTM-  eigene LSTM-  eigene LSTM-
         hidden state  hidden state  hidden state  hidden state
```

### Was geteilt wird vs. was pro Agent ist

| Komponente | Geteilt | Pro Agent |
|---|---|---|
| **Netzwerk-Gewichte** θ | ✅ ein Set | – |
| **Forward-Pass-Code** | gleicher Code | – |
| **Observation** | – | ✅ eigene |
| **LSTM Hidden State** | – | ✅ eigener (pro Zug eine eigene Sequenz) |
| **Action** | – | ✅ eigenes Sample |
| **Reward** | – | ✅ eigener |
| **Trajektorie für PPO** | – | ✅ wird gesammelt |
| **Comm-Buffer** | env-weit gemeinsam | – (alle lesen den gleichen) |

### Was die Agents trotzdem unterschiedlich macht

Da jeder Zug eine **andere lokale Observation** sieht (andere Position, andere Nachbarn, anderes Ziel),
produziert das gleiche Netzwerk **andere Embeddings → andere Aktionen**.

- **Comm-Klassen** (ONCOMING/MERGING/LOCAL) sind aus der Sicht jedes Agents anders – derselbe Nachbar
  kann für A „oncoming" sein und für B ebenfalls „oncoming".
- **LSTM-Hidden** akkumuliert pro Zug die eigene Bewegungs-Historie.

### Training mit Parameter-Sharing (PPO) – API-konform zum Solver

Der Solver ruft am Policy-Objekt **exakt** die `Policy`-API aus
[policy/policy.py](../../policy/policy.py) auf:
`reset(env)`, `start_episode(train)`, `start_step(train)`, `act(handle, state, eps)`,
`step(handle, state, action, reward, next_state, done)`, `end_step(train)`, `end_episode(train)`.

Parameter Sharing bedeutet: **Eine** `DeciderPPOPolicy`-Instanz wird vom Solver für **alle** `handle`s
benutzt. Pro `handle` wird intern eine eigene Trajektorie gepuffert (Memory pro Handle, wie in
`MARL_ATTENTION_TEMPORAL_PPOPolicy`).

```python
class DeciderPPOPolicy(LearningPolicy):
    """
    EIN globales Netzwerk fuer ALLE Zuege (Parameter Sharing).
    Wird vom Solver pro handle aufgerufen, intern werden Trajektorien
    handle-getrennt gepuffert und am Episodenende gemeinsam aktualisiert.
    """

    # ---- Konstruktion: einmal pro Run, ein Set Gewichte ----
    def __init__(self, state_size, action_size, in_parameters):
        super().__init__()
        self.shared_net = SharedNetwork(...)   # Specialists + Decider + Critic
        self.optimizer  = torch.optim.Adam(self.shared_net.parameters(), lr=...)
        self.current_episode_memory = EpisodeMemory()  # pro handle getrennt
        self.comm_buffer = CommBuffer()                # env-weit, alle Handles

    # ---- Solver-API ----
    def reset(self, env):
        self.comm_buffer.reset(num_agents=len(env.agents))

    def start_episode(self, train: bool):
        self.current_episode_memory.reset()
        self.comm_buffer.reset_history()

    def start_step(self, train: bool):
        # vor jedem env.step() aufgerufen -> z.B. comm-snapshot freezen
        self.comm_buffer.freeze_read_view()

    def act(self, handle: int, state, eps=0.):
        """
        state ist temporal: [(obs_t-2,...), (obs_t-1,...), (obs_t,...)]
        SHARED_NETWORK wird mit handle's eigener Sicht aufgerufen.
        Aktion = Flatland 5-way (DO_NOTHING/L/F/R/STOP).
        """
        with torch.no_grad():
            comm_in = self.comm_buffer.read_for(handle)            # K=4 sparse
            self_ctx = self.shared_net.lstm(state, handle)
            specialists_out = self.shared_net.specialists(state, comm_in, self_ctx)
            action_logits, value, gm_out = self.shared_net.decider(specialists_out)
            action_logits = mask_illegal(action_logits, state)     # action mask
            dist = Categorical(logits=action_logits)
            action = dist.sample()

        # global_metric_out fuer naechsten Step in Buffer schreiben
        self.comm_buffer.write(handle, gm_out.cpu().numpy())
        return action.item()

    def step(self, handle, state, action, reward, next_state, done):
        """Pro handle Transition speichern - identisch zu bestehender PPO-Policy."""
        transition = (state, action, reward, next_state, done)
        self.current_episode_memory.push_transition(handle, transition)

    def end_step(self, train: bool):
        pass

    def end_episode(self, train: bool):
        if not train:
            return
        # ALLE Handle-Trajektorien gemeinsam in den PPO-Update einspeisen.
        all_transitions = self.current_episode_memory.collect_all_handles()
        self._ppo_update(all_transitions)        # PPO + Aux-Losses (BCE Deadlock, ...)
        self.current_episode_memory.reset()

    # ---- Persistence ----
    def save(self, filename): torch.save(self.shared_net.state_dict(), filename)
    def load(self, filename): self.shared_net.load_state_dict(torch.load(filename))
    def get_name(self): return self.__class__.__name__
```

**Wichtig für API-Kompatibilität:**

| Methode | Wer ruft sie? | Verhalten in `DeciderPPOPolicy` |
|---|---|---|
| `reset(env)` | Solver bei Env-Reset | `comm_buffer` re-allokieren auf `len(env.agents)` |
| `start_episode(train)` | Solver pro Episode | Episoden-Memory leeren, Comm-History reset |
| `start_step(train)` | Solver pro Env-Step (vor act) | Comm-Buffer Read-View einfrieren (alle lesen `t-1`) |
| `act(handle, state, eps)` | Solver pro Agent pro Step | Forward-Pass + Action sampling + `gm_out` in Buffer schreiben |
| `step(handle, ...)` | Solver pro Agent pro Step (nach env-step) | Transition in handle-Memory schreiben |
| `end_step(train)` | Solver pro Env-Step (Ende) | – |
| `end_episode(train)` | Solver am Episodenende | **gemeinsamer** PPO-Update über alle Handles |
| `save/load/get_name` | Solver für Checkpoints | Standard |

Das bestehende `MARL_ATTENTION_TEMPORAL_PPOPolicy` folgt **exakt** diesem Muster
(siehe [marl_attention_temporal_mappo.py](marl_attention_temporal_mappo.py)) –
`DeciderPPOPolicy` erbt von `LearningPolicy` und implementiert die gleichen Hooks.

**Effekt:** N Züge → N-mal so viele Trainings-Samples pro Episode → schnelleres Lernen,
weil ein Netzwerk gleichzeitig von allen Zügen profitiert.

### Vorteile / Limitierungen

| ✅ Vorteile | ⚠️ Limitierungen |
|---|---|
| Skaliert auf beliebige Agentenzahl ohne neue Parameter | Alle Agents müssen homogen sein (✓ in Flatland: alle sind Züge) |
| Mehr Daten pro Update | Kein hart agent-spezifisches Verhalten – aber Comm-Mechanismus + Obs-Unterschiede lösen das implizit |
| Generalisiert auf Episoden mit anderer Agentenzahl | – |
| Standard in MARL (MAPPO, IPPO, MADDPG mit Sharing) | – |

> **Kurz:** Ein PPO-Agent = ein Set Netzwerk-Gewichte. Alle N Züge teilen dieses Netzwerk.
> Unterschiedliches Verhalten entsteht ausschließlich durch unterschiedliche Observations,
> unterschiedliche LSTM-Hidden-States und unterschiedliche Comm-Sichten.

Das matched das bestehende `MARL_ATT_DecisionPointPolicy`-Schema – wir behalten Parameter Sharing bei,
nur die **innere Architektur** (Specialists + Decider) ist neu.

---

## 3. Gesamtbild – Ein Step für **einen** Agent

```
═══════════════════════════════════════════════════════════════════════════════
                        STEP t   für Agent A
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  ENV / OBSERVATION BUILDER                                              │
  │                                                                         │
  │  ┌────────────────────┐   ┌────────────────────┐  ┌─────────────────┐  │
  │  │ DecisionPointObs   │   │ Route-Enumerator   │  │ Sparse-Selector │  │
  │  │ (Switch-Logik)     │   │ (max 4 Routen)     │  │ K=4 Nachbarn    │  │
  │  │ → 48D base         │   │ → 4×8D routes      │  │ + Klassen       │  │
  │  └────────┬───────────┘   └─────────┬──────────┘  └────────┬────────┘  │
  └───────────┼─────────────────────────┼──────────────────────┼───────────┘
              │                         │                      │
              ▼                         ▼                      ▼
   ╔═════════════════════════════════════════════════════════════════════╗
   ║                    PER-AGENT NETWORK                                ║
   ║                                                                     ║
   ║  ┌──────────────────────┐                                           ║
   ║  │  LSTM (eigene Obs    │   ◄── Self-History (3-5 Steps)            ║
   ║  │  über 3-5 Steps)     │                                           ║
   ║  │  → self_ctx 64D      │                                           ║
   ║  └──────────┬───────────┘                                           ║
   ║             │                                                       ║
   ║   ┌─────────┴─────────────────────┬─────────────────────┐           ║
   ║   ▼                               ▼                     ▼           ║
   ║ ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────┐    ║
   ║ │  ROUTING    │   │  MERGING    │   │  DEADLOCK   │   │  COMM  │    ║
   ║ │ Specialist  │   │ Specialist  │   │ Specialist  │   │ Spec.  │    ║
   ║ │             │   │             │   │             │   │        │    ║
   ║ │ Input:      │   │ Input:      │   │ Input:      │   │ Input: │    ║
   ║ │ obs[0-21]   │   │ obs[22-29]  │   │ obs[42],    │   │ K=4    │    ║
   ║ │ + 4 routes  │   │ + obs[43-47]│   │ alle Risks, │   │ Nachb. │    ║
   ║ │             │   │             │   │ self_ctx    │   │ Hist 3 │    ║
   ║ │ Output:     │   │ Output:     │   │ Output:     │   │ Output │    ║
   ║ │ route_emb   │   │ merge_emb   │   │ dl_emb      │   │ comm_  │    ║
   ║ │ 32D         │   │ 32D         │   │ 16D         │   │ emb 32D│    ║
   ║ │ + L/F/R     │   │ + wait_pres │   │ + p_dl_1    │   │ + gate │    ║
   ║ │   logits    │   │ + priority  │   │ + p_dl_3    │   │        │    ║
   ║ │ + conf      │   │             │   │             │   │        │    ║
   ║ └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └────┬───┘    ║
   ║        ▼                 ▼                 ▼               ▼        ║
   ║   ┌──────────────────────────────┐                                  ║
   ║   │      FUSION (Concat)         │                                  ║
   ║   │  base + 4 emb + scores       │                                  ║
   ║   │  → 162D                      │                                  ║
   ║   └──────────────┬───────────────┘                                  ║
   ║                  ▼                                                  ║
   ║   ┌──────────────────────────────┐                                  ║
   ║   │     DECIDER (MLP 128D)       │                                  ║
   ║   │   ┌────────────┬───────────┐ │                                  ║
   ║   │   │   Actor    │  Critic   │ │                                  ║
   ║   │   │ Logits[5]  │  Value    │ │                                  ║
   ║   │   └─────┬──────┴───────────┘ │                                  ║
   ║   │   ┌─────▼──────────────┐     │                                  ║
   ║   │   │  Action Mask       │     │                                  ║
   ║   │   │  (illegal=-inf)    │     │                                  ║
   ║   │   └─────┬──────────────┘     │                                  ║
   ║   │         ▼                    │                                  ║
   ║   │   Flatland Action            │                                  ║
   ║   │   {DO_NOTHING, L, F, R, STOP}│                                  ║
   ║   └──────────────┬───────────────┘                                  ║
   ║   ┌──────────────┴────────────────┐                                 ║
   ║   │  global_metric_out (16D)      │ ─── fließt in Comm-Buffer       ║
   ║   │  (Decider-Hidden → Linear)    │     für ALLE anderen Agenten    ║
   ║   └───────────────────────────────┘     in Step t+1                 ║
   ╚═════════════════════════════════════════════════════════════════════╝
              │
              ▼
        ┌──────────────────┐
        │  Flatland ENV    │
        │  step()          │
        └──────────────────┘
```

---

## 4. Spezialisten im Detail

### 4.1 RoutingSpecialist – „Welcher Branch ist gut?"

| Aspekt | Wert |
|---|---|
| **Input** | `base_obs[0-21]` (decision_type + 3 Branch-Blöcke) + `routes[4×8D]` |
| **Output** | `route_emb[32D]` + soft `route_logits[3]` (L/F/R) + `route_confidence[1]` |
| **Aux-Loss (optional)** | Cross-Entropy gegen `shortest_path_hint[1-3]` (Bootstrap, abschaltbar) |
| **Aktivität** | Hoch wenn `decision_type & 2` (Routing-Switch), sonst niedrige Confidence |

### 4.2 MergingSpecialist – „Wer hat Vorfahrt?"

| Aspekt | Wert |
|---|---|
| **Input** | `base_obs[22-29]` (merge fwd/bwd) + `[42]` (local_dl) + `[43-47]` (coord) + `comm_emb` |
| **Output** | `merge_emb[32D]` + `wait_pressure[1]` + `priority_score[1]` |
| **Aux-Loss** | BCE gegen Heuristik `wait_intent > 0.5` (Bootstrap, abschaltbar nach ~500 Eps) |
| **Aktivität** | Hoch wenn `decision_type & 4` (Merge-Switch) |

### 4.3 DeadlockSpecialist – „Kommt es zum Stillstand?"

| Aspekt | Wert |
|---|---|
| **Input** | Alle Risk-Features + `self_ctx` (LSTM) + `comm_emb` |
| **Output** | `dl_emb[16D]` + `p_deadlock_1step[1]` + `p_deadlock_3step[1]` |
| **Aux-Loss** | **BCE** gegen 1-Step + 3-Step Lookahead-Label aus echtem Env-Rollout |
| **Wichtig** | Direkter Gradient unabhängig von PPO → garantiertes Lernsignal für Comm |

### 4.4 CommSpecialist – „Welche Nachrichten zähle ich?"

| Aspekt | Wert |
|---|---|
| **Input** | K=4 sparse Nachbarn × `global_metric[16D]` × 3 Time-Steps |
| **Mechanik** | Multi-Head Attention (Q=`self_ctx`, K,V=Nachbar-Metrics) + Temporal-Attention über 3 Steps |
| **Output** | `comm_emb[32D]` + `gate_score[1]` |
| **Loss** | Indirekt über Decider/Deadlock-Aux-Loss → Comm muss nützliche Infos liefern |

---

## 5. Inter-Agent Communication

```
═══════════════════════════════════════════════════════════════════════════════
                      INTER-AGENT COMMUNICATION
═══════════════════════════════════════════════════════════════════════════════

  Step t-1 (Vergangenheit)              Step t (Jetzt)
  ─────────────────────────             ─────────────────────────

  Agent A → global_metric_A_t-1 ┐
  Agent B → global_metric_B_t-1 │
  Agent C → global_metric_C_t-1 ├─►  COMM-BUFFER (env-weit)
  Agent D → global_metric_D_t-1 │      pro Agent X gespeichert:
  Agent E → global_metric_E_t-1 ┘      letzte 3 Steps × N Agenten

                                              │
                                              ▼
                                    ┌───────────────────────┐
                                    │  Sparse-Selector      │
                                    │  (für Agent A)        │
                                    │                       │
                                    │  Filtert auf K=4      │
                                    │  relevanteste Nachb.  │
                                    │  per Klassifikation:  │
                                    │   • ONCOMING (Gegner) │
                                    │   • MERGING (mündend) │
                                    │   • LOCAL (nah, kein  │
                                    │     direkter Konflikt)│
                                    │                       │
                                    │  KEIN handle/id!      │
                                    │  Nur 3-bit Class +    │
                                    │  Distanz + Inhalt     │
                                    └───────────┬───────────┘
                                                ▼
                                    ┌───────────────────────┐
                                    │  CommSpecialist       │
                                    │  (für Agent A)        │
                                    │                       │
                                    │  Multi-Head-Attention │
                                    │  Q = self_ctx_A       │
                                    │  K,V = global_metrics │
                                    │       der K=4 Nachb.  │
                                    │  + Temporal Attention │
                                    │    über 3 Steps       │
                                    │                       │
                                    │  → comm_emb_A (32D)   │
                                    │  → gate_score_A (1D)  │
                                    └───────────┬───────────┘
                                                ▼
                                       In Decider-Fusion (Agent A)
```

### Was wird ausgetauscht (`global_metric`, 16D):

```
[0-7]   intent_embed       (frei lernbar, aus Decider-Hidden)
[8]     p_deadlock_1step   ◄── DeadlockSpecialist
[9]     p_deadlock_3step   ◄── DeadlockSpecialist
[10]    wait_pressure      ◄── MergingSpecialist
[11]    priority_score     ◄── MergingSpecialist
[12-14] route_logits soft  ◄── RoutingSpecialist
[15]    progress_norm      ◄── eigener Distanzfortschritt
```

**NICHT enthalten:** `handle` / `id` / `position` / `agent_index`.

---

## 6. Lernsignale – Wer bekommt warum Gradient?

```
PPO-Loss (Decider-Output)
    ├──► Decider
    ├──► Fusion-Layer
    ├──► RoutingSpec
    ├──► MergingSpec
    ├──► DeadlockSpec
    ├──► CommSpec
    └──► LSTM-Encoder

BCE Aux-Loss (Deadlock 1-Step + 3-Step Lookahead)
    ├──► DeadlockSpec   ◄── DIREKTES SIGNAL, unabhängig von PPO
    ├──► CommSpec       ◄── Comm fließt in Deadlock-Input
    └──► LSTM-Encoder

Cross-Entropy Aux-Loss (Routing vs shortest_path_hint, optional)
    └──► RoutingSpec    ◄── Bootstrap, abschaltbar nach ~500 Eps

BCE Aux-Loss (Merging wait_pressure vs Heuristik, optional)
    └──► MergingSpec    ◄── Bootstrap, abschaltbar

Critic-Loss (V vs Returns)
    └──► alle (durch Fusion-Backprop)
```

---

## 7. Drei Zeit-Ebenen

| Zeit-Ebene | Mechanismus | Zweck |
|---|---|---|
| Innerhalb 1 Step (Forward) | Specialist → Decider (Fusion) | Entscheidung „was tue ich?" |
| Letzte 3-5 Steps pro Agent | LSTM (eigene Obs) | „Habe ich mich bewegt? Steh ich seit 3 Steps?" |
| Letzte 3 Steps global Comm | Temporal Attention auf Comm-Buffer | „Sendet Nachbar X seit 2 Steps WAIT?" → robust gegen Spike-Noise |
| Episode (PPO) | Returns + GAE | Langfrist-Belohnung |

---

## 8. Datenflüsse zusammengefasst

| Pfad | Quelle | Ziel | Inhalt |
|---|---|---|---|
| **Lokale Obs** | Flatland-Env | Specialists | 48D base + 4×8D routes |
| **Self-History** | Eigene letzte Obs | LSTM → Specialists | Bewegungs-Trends |
| **Cross-Agent (t-1)** | Comm-Buffer | CommSpecialist | K=4 × 3 × 16D Nachrichten |
| **Action** | Decider | Flatland-Env | 5-way action |
| **global_metric_out (t)** | Decider-Hidden | Comm-Buffer | 16D pro Agent |
| **Gradient PPO** | Env-Reward | alle Module | Standard PPO |
| **Gradient Aux** | Lookahead-Labels | Deadlock/Routing/Merging | Direkte BCE/CE |

---

## 9. Datei-Struktur (Implementation)

```
flatland_solver_policy/example/flatland_rail_env/
│
├── marl_attention_temporal_observation/
│   ├── decision_point_observation.py        ◄── BLEIBT (Switch-Logik)
│   └── hierarchical_routes_observation.py   ◄── NEU
│       └── HierarchicalRoutesObservation
│           ├── erbt von DecisionPointObservation
│           ├── + enumerate_routes (max 4)
│           └── + classify_neighbors (sparse K=4)
│
├── decider_policy.py                        ◄── NEU
│   ├── RoutingSpecialist
│   ├── MergingSpecialist
│   ├── DeadlockSpecialist
│   ├── CommSpecialist
│   ├── DeciderActorCritic
│   ├── CommBuffer
│   └── DeciderPPOPolicy
│
└── marl_attention_temporal.py               ◄── ERWEITERT
    └── + create_decider_agent()
```

---

## 10. Trainings-Strategie

1. **Phase 0 – Isolierter Test** (ohne Curriculum)
   - 300 Episoden, 2-3 Agenten, kleines Grid
   - Aux-Losses aktiv (Deadlock-BCE muss konvergieren < 0.3)
   - PPO mit moderaten Hyperparametern

2. **Phase 1 – Bootstrap mit Heuristik-Aux**
   - Routing-CE & Merging-BCE als Bootstrap aktiv
   - Wenn Done-Rate > 50 %, Bootstrap abschalten

3. **Phase 2 – Curriculum aktivieren**
   - Bestehende 3-Phasen-Curriculum übernehmen
   - Nur PPO + Deadlock-Aux

4. **Phase 3 – Stabilisierung**
   - LR-Decay, Entropy-Decay
   - Comm-Gate-Monitoring (sollte > 0.05 aktiv sein)

---

## 11. Erwartete Vorteile gegenüber bisheriger Architektur

| Punkt | Bisher (`MARL_ATT_DecisionPointPolicy`) | Neu (Decider) |
|---|---|---|
| Lernsignal Comm | Nur indirekt via PPO | **Direkt** via DeadlockSpec-BCE |
| Spezialisierung | Eine Policy für alles | **4 Köpfe** lernen je eigene Aufgabe |
| Robustheit Sub-Versagen | Bricht ein | Decider kann ignorieren (Confidence) |
| Plateau-Bruch | Schwierig | Aux-Losses brechen Plateaus |
| Aktionsraum | Flatland 5-way ✓ | Flatland 5-way ✓ (unverändert) |

---

## 11b. Skalierbarkeit auf beliebige Agentenzahl

> **Antwort: ja – die Policy ist agentenzahl-unabhängig.**
> Ein einziges Set Netzwerk-Gewichte θ funktioniert für `N ∈ {1, 2, 3, ..., 50, ...}`
> ohne Architekturänderung.

### Warum sie frei skaliert

Die Netzwerk-Gewichte θ haben **keine Dimension, die von N abhängt**:

| Komponente | Input-Shape | N drin? |
|---|---|---|
| Specialists (Routing/Merging/Deadlock) | `[batch, 48D]` + `[batch, 4×8D]` (Routen) | ❌ |
| LSTM | `[batch, T, 48D]` (eigene Historie) | ❌ |
| CommSpecialist | `[batch, K=4, 16D]` | ❌ **K ist fix**, nicht N |
| Decider (MLP) | `[batch, 162D]` | ❌ |

Die einzige Stelle wo „andere Agenten" reinkommen ist der **Comm-Buffer**, und der wird durch
den **Sparse-Selector auf K=4** zusammengedampft – egal ob 2 oder 50 Züge im Env sind.

### Was bei verschiedenen N tatsächlich passiert

```
N=2:  Sparse-Selector findet 1 Nachbar → padding auf K=4 mit zero-mask
N=5:  Sparse-Selector findet 4 Nachbarn → exakt K=4
N=20: Sparse-Selector wählt Top-4 aus 19 → bleibt K=4
```

→ Netzwerk sieht immer die gleiche Input-Form → ein θ funktioniert für alle N.

### Drei Skalier-Regeln (Constraint-Check)

| Regel | Status im Plan | Wo |
|---|---|---|
| **1.** Keine N-abhängigen Layer-Größen | ✅ | Alle Specialist-Inputs sind fix |
| **2.** Cross-Agent-Info via Sparse-Pooling (fixe K) | ✅ | K=4 Selector |
| **3.** Keine `handle`/`agent_id`-Embeddings | ✅ | Comm nutzt nur Inhalt + 3-Bit-Klasse |

### Praktische Konsequenzen

- **Curriculum-Training:** N ∈ {2, 3, 4, 5, ...} ohne Architektur-Änderung
- **Inference:** Auf N=5 trainiert → auf N=10 oder N=20 anwendbar
- **Zero-Shot Transfer:** Technisch möglich; in der Praxis leichte Performance-Drops bei stark
  abweichenden Konfliktdichten erwartbar

### Limitierungen

- Bei sehr großem N (z.B. 50) wird K=4 evtl. zu wenig Info → K erhöhen oder Multi-Class Selector
- Comm-Buffer-Speicher wächst linear mit N; Forward-Pass bleibt aber O(K) → konstant pro Agent

---

## 12. KPI-Erwartung VOR dem Bau (kalibrierte Prognose)

> Setting: 30×40 Grid, 3 Cities, gleicher Curriculum-Endzustand wie bisher.
> KPI: **Done-Rate** über `10 Episoden × N Agenten` für `N ∈ {1, 2, 3, 4, 5}`.
> Baseline: bisherige `MARL_ATT_DecisionPointPolicy` mit DLA-Shield + 3-Phasen-Curriculum.

### Erwartete Done-Rate

| N Agents | Bisher (Best Case) | Decider neu (erwartet) | Decider neu (Stretch) |
|---:|---:|---:|---:|
| 1 | ~95 % | **97–100 %** | 100 % |
| 2 | ~80 % | **85–92 %** | 95 % |
| 3 | ~60 % | **70–80 %** | 88 % |
| 4 | ~50 % | **55–68 %** | 78 % |
| 5 | ~42 % | **48–62 %** | 72 % |
| **Avg** | **~65 %** | **~71–80 %** | **~87 %** |

**Konservativer Punktschätzwert** nach 2000 Episoden Curriculum: **Avg ≈ 72 %**
(+7 pp absolut über Baseline; löst N=5-Engpass von 42 → 55 %).

### Begründung der Spannen

**Warum besser als bisher:**
- DeadlockSpec mit BCE-Aux-Loss bricht das bekannte Comm-Lernsignal-Problem (Gate ~0.04, CommW ~3e-4)
- Spezialisten-Spezialisierung sollte Plateau bei N=4,5 anheben (bisheriger Hauptengpass)
- Action-Mask + Decider verhindert ungültige Routing-Aktionen

**Warum nicht 95 %+ überall:**
- Sparse-Selector mit K=4 ist eine Approximation – bei N=5 verliert man bereits Info
- Aux-Losses brauchen Rollout-Daten → erste ~300 Episoden eher schlechter als Baseline
- Comm-Buffer mit `t-1`-Read kann bei sehr schnellen Konflikten zu spät kommen
- Single-Track-Sektionen mit Head-on-Konflikt bleiben prinzipiell schwer

### Erwartungs-Trajektorie über Training

| Episodes | Erwartete Avg-Done (N=1..5) | Was passiert |
|---|---|---|
| 0–200 | 15–25 % | Aux-Losses initialisieren, Bootstrap |
| 200–600 | 35–50 % | Specialists differenzieren sich |
| 600–1500 | 55–70 % | Comm wird gelernt, Decider lernt zu kombinieren |
| 1500–2500 | 68–80 % | **Konvergenz-Bereich** |
| > 2500 | Stretch nur mit größerem Net / mehr Cities-Variation | – |

### Erfolgskriterien („besser als bisher")

| Kriterium | Schwelle |
|---|---|
| **Avg Done-Rate** über 10×[1..5] | **≥ 70 %** (vs. ~65 % bisher) |
| Done-Rate bei N=5 | **≥ 50 %** (vs. ~42 %) |
| **DeadlockSpec BCE-Loss** | konvergiert auf **< 0.30** |
| **Comm-Gate** | aktiv **> 0.10** in Phase 2 (vs. 0.04 bisher) |
| Stabilität (max-min Done über letzte 100 Eps bei N=5) | **≤ 15 pp** Schwankung |

### Risiken die das KPI killen können

| Risiko | Wahrscheinlichkeit | Mitigation |
|---|---|---|
| Decider lernt Specialists zu ignorieren (Collapse zu „nur base_obs") | ~50 % | Specialist-Dropout-Regularisierung |
| Comm-Buffer-Latenz (t-1 zu spät bei dichtem Verkehr) | ~30 % | Schreiben in `start_step` zwingend vor `act` |
| Aux-Loss überlagert PPO zu früh | ~25 % | Aux-Weight-Decay nach 800 Eps |

### Falls deutlich > 85 % gewünscht

Zusätzlich nötig (nicht in Phase 0–3 enthalten):
- Centralized Critic statt Parameter-shared
- Größeres Hidden (256 → 512)
- Belief-State-Tracking statt nur LSTM

> **Entscheidung erst nach** dem ersten Decider-Training, basierend auf realen Zahlen.









---
Update 07.05.2026
---


# =============================================================================
# 📊 OBSERVATION FEATURES DOCUMENTATION
# =============================================================================
# 
# Das MARL-Modell erhält für jeden Agenten eine 72D Observation pro Zeitschritt,
# die in 3 zeitliche Frames gepuffert wird (TEMPORAL_WINDOW = 3).
# Der Agent sieht somit die letzten 3 Timesteps, um Bewegungen & Intentions zu erkennen.
#
# ═══════════════════════════════════════════════════════════════════════════
# 1. BASE OBSERVATION (48D) - DecisionPointObservation
# ═══════════════════════════════════════════════════════════════════════════
#
# [0] DECISION TYPE (1D) - normalized to [0, 1]
#     Kategorie der aktuellen Situation:
#     • 0 = Normaler Fortschritt (1 Weg verfügbar)
#     • 1 = Ready-to-depart (Agent startet gerade)
#     • 2 = Switch-Entscheidung (>1 Weg an Junction)
#     • 4 = Merge-Point (Weg wird hinten verlassen)
#     • 6 = Switch + Merge combined
#     • 8 = Done / Agent am Ziel
#
# [1-3] SHORTEST PATH HINT (3D) - one-hot [left, forward, right]
#     Optimal Routing-Tipp vom Dijkstra Distance-Map:
#     Welche Richtung führt am schnellsten zum Ziel?
#     Gibt direktes Guidance-Signal für den Actor ohne credit assignment.
#
# [4-9] BRANCH LEFT (6D) - Route nach links an Switch:
#     [0] progress_gain:     0→1, wie viel näher zum Ziel? (normalized distance reduction)
#     [1] deadlock_signal:   0→1, Deadlock erkannt? (0.85=soft, 1.0=hard)
#     [2] switches_norm:     0→1, Wie viele weitere Switches im Pfad? normalized
#     [3] branch_dist_norm:  0→1, Distanz zum Ziel? (normalized by max_reachable)
#     [4] target_found:      0 oder 1, Ziel erreichbar in diesem Branch?
#     [5] abort_flag:        0 oder 1, Route blockiert/deadlock?
#
# [10-15] BRANCH FORWARD (6D) - identische Layout wie LEFT
# [16-21] BRANCH RIGHT (6D)   - identische Layout wie LEFT
#     Die 3 Branch-Blöcke geben per-direction routing hints. Jeder enthält
#     potenzielle Deadlock-Signale, Distanz-Info, Fortschritts-Anzeige.
#     Der Agent lernt hier: "Linker Weg = hohes Deadlock-Risiko" etc.
#
# [22-25] MERGE FORWARD SUB-BLOCK (4D):
#     [0] deadlock_signal:   Deadlock hinter mir? (Andere Agenten-Konflikt?)
#     [1] switches_norm:     Schalter im Rückbereich
#     [2] target_found:      Ziel erreichbar wenn ich NICHT abbiege?
#     [3] abort_flag:        Bewegung nach vorne ist unmöglich?
#
# [26-29] MERGE BACKWARD SUB-BLOCK (4D) - identische Layout
#     Erkennt, wenn Agent am Eintritt zu Merge-Punkt steht.
#     Signalisiert: "Andere Agenten warten hinter mir, gib Platz"
#
# [30-36] STATE ONE-HOT (7D) - TrainState encoding:
#     Position:  [READY_TO_DEPART, ON_MAP_SLOW, ON_MAP_MEDIUM, ON_MAP_FAST, MALFUNCTION, MALFUNCTION_OFF_MAP, DONE]
#     Wert:      Nur eine Position = 1.0, Rest = 0.0
#     Wissen:    Agentenstatus - spawnt gerade? läuft normal? blockiert? fertig?
#
# [37-41] LAST ACTION ONE-HOT (5D) - RailEnvActions encoding:
#     Position:  [DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING]
#     Wert:      Letzte Aktion wiedergeben für Action-Konsistenz
#     Wissen:    "Ich bin gerade nach vorne gefahren" - für Temporal-Differenzierung
#
# [42] LOCAL DEADLOCK FLAG (1D) - 0 oder 1
#     Erkennung: Agent steht in "lokalen Deadlock" Situation:
#     • nur 1 Ausweg verfügbar
#     • anderer Agent sitzt kopfüber in diesem Ausweg
#     • => Beide können sich nicht bewegen
#     Kritisches Signal für Koordination / Shielding
#
# [43] COORDINATION WAIT INTENT (1D) - normalized [0, 1]
#     Berechnete "Bereitschaft zu warten" Signal:
#     wait_intent = 0.45*local_deadlock + 0.30*conflict_pressure + 0.20*ttc_risk + 0.10*cycle_risk + 0.10*(1.0-progress_best)
#     Verwendet: Deadlock-Status, Konflikt-Druck, Time-to-Collision, Zyklenrisiko
#     Interpretation: "Ich sollte wahrscheinlich stoppen und warten"
#
# [44] COORDINATION GO INTENT (1D) - normalized [0, 1]
#     Berechnete "Bereitschaft zu fahren" Signal:
#     go_intent = progress_best * right_of_way * (1.0 - 0.75*wait_intent)
#     Interpretation: "Ich habe Vorfahrt und guter Fortschritt - los geht's!"
#
# [45] COORDINATION PRIORITY / RIGHT-OF-WAY (1D) - normalized [0, 1]
#     Basiert auf: Distanz zum Ziel vs. andere Agenten
#     Mehr Priorität = kürzere Distanz zum eigenen Ziel
#     Soft-Signal für "wer darf fahren" ohne zentrale Verwaltung
#
# [46] COORDINATION CONFLICT PRESSURE (1D) - normalized [0, 1]
#     Gesamtdruckindex von allen Konflikt-Quellen:
#     conflict_pressure = 0.45*branch_risk + 0.20*merge_risk + 0.20*ttc_risk + 0.15*cycle_risk
#     Hoch = viele potenzielle Verkehrskonflikte in der Nähe
#
# [47] COORDINATION YIELD HINT (1D) - normalized [0, 1]
#     "Ich sollte wahrscheinlich ausweichen/stoppen" Signal:
#     yield_hint = wait_intent * (1.0 - priority) * (0.7 + 0.3*cycle_risk)
#     Interpretation: "Wenn ich keine Priorität habe UND andere warten, weich aus"
#
# ═══════════════════════════════════════════════════════════════════════════
# 2. HIERARCHICAL NEIGHBOR BLOCK (24D) - Top-4 Relevant Agents
# ═══════════════════════════════════════════════════════════════════════════
#
# [48-71] SPARSE NEIGHBOR BLOCK (4 agents x 6D each = 24D)
#
# Per Neighbor (6D Layout):
#   [0] EXISTS_FLAG:         0 oder 1 - ist dieser Slot belegt? (im Local Radius?)
#   [1] CLASS_ONCOMING:      0 oder 1 - Kopfüber-Konflikt erkannt?
#   [2] CLASS_MERGING:       0 oder 1 - Merging-Konflikt? (Andere fahren in meine Route)
#   [3] CLASS_LOCAL:         0 oder 1 - Lokaler Deadlock zusammen?
#   [4] DISTANCE_NORMALIZED: 0→1 - Wie weit weg? (max_dist = LOCAL_RADIUS = 6)
#   [5] TTC_NORMALIZED:      0→1 - Time-to-Collision (bei aktuellen Geschwindigkeiten)
#
# Sortierung: Nach Relevanz-Score (höchste zuerst)
#   relevance = 0.2*decision_strength + 1.0*deadlock + 0.6*conflict_pressure + ...
#   → Die gefährlichsten Nachbarn stehen zuerst im Array
#
# Design-Philosophie:
#   • KEINE agent_id / handle / position exposing!
#   • Nur relative Features (Distanz, Klassifizierung, Risiko)
#   • Local Radius = 6 Felder → Agent sieht nur unmittelbare Umgebung
#   • Top-K Filtering → Begrenzte Attention auf K=4 Nachbarn
#   • Ermöglicht dezentralisierte Koordination ohne global state
#
# ═══════════════════════════════════════════════════════════════════════════
# 3. TEMPORAL WRAPPER (3 Frames)
# ═══════════════════════════════════════════════════════════════════════════
#
# INPUT STRUCTURE: [(obs_{t-2}, opponents_{t-2}), (obs_{t-1}, opponents_{t-1}), (obs_t, opponents_t)]
#
# TEMPORAL_WINDOW = 3 bedeutet:
#   Frame 0: 2 Steps in der Vergangenheit
#   Frame 1: 1 Step in der Vergangenheit
#   Frame 2: aktueller Step
#
# VELOCITY FEATURES (EMERGENT, nicht explizit berechnet):
#   Der LSTM-Encoder SIEHT NUR die 3 rohen 72D Observation-Vektoren (obs_t-2, obs_t-1, obs_t).
#   Diese werden zuerst in 128D Embeddings projiziert [obs_encoder: Linear→LN→LeakyReLU].
#   Der LSTM verarbeitet dann: emb_t-2 → emb_t-1 → emb_t (3×128D Sequenz)
#   KEINE expliziten Deltas! Der LSTM muss selbst lernen, dass:
#      • branch_dist_norm sinkt (0.8 → 0.7 → 0.6) = Agent fährt näher zum Ziel
#      • decision_type ändert sich (0 → 2) = Agent nähert sich Switch
#      • wait_intent steigt = Situation wird kritischer
#   Das ist schwächer als explizite Deltas, aber recheneffizienter.
#
# EARLY-EPISODE ZERO-PADDING:
#   In den ersten 2 Steps wird [0D vector] eingefügt, damit der Agent nicht
#   nur Nullsequenzen während Warm-up sieht (würde zero-velocity-Bias erzeugen).
#
# MULTI-AGENT OPPONENT TRACKING:
#   Jeder Agent sieht seine Top-4 Nachbarn temporal:
#   → Kann lernen: "Dieser Nachbar war 1 Step weg, jetzt 0.3 Schritte → Frontal-Kollisions-Risiko!"
#
# ═══════════════════════════════════════════════════════════════════════════
# 4. AGENT KNOWLEDGE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
#
# ✅ LOKALES WISSEN (Deterministisch aus Flatland):
#    • Meine Position, Richtung, Ziel
#    • Schienennetz Topologie (wohin kann ich fahren?)
#    • Distanz-Map zum Ziel (Dijkstra)
#    • Letzte Aktion, aktueller Status
#    • Lokale Deadlock-Erkennung
#
# ✅ NACHBARN-WISSEN (Probabilistisch, Top-4):
#    • Konflikt-Typ (Kopfüber, Merging, Lokal-Deadlock)
#    • Relative Distanz
#    • Time-to-Collision Estimate
#    • NICHT: Namen, IDs, exakte Positionen (Privacy)
#
# ✅ ROUTING-GUIDANCE (Policy-Independent):
#    • Shortest Path Hint von Dijkstra (1-3D signal)
#    • Branch-wise Deadlock/Progress per direction
#    • Ziel-Erreichbarkeit pro Branch
#
# ✅ COORDINATION SOFT-SIGNALS (Emergent, Learned):
#    • Wait/Go Intents (Dezentralisiert, kein Master)
#    • Right-of-Way Hints (Basiert auf Distanz + Konflikt)
#    • Conflict Pressure (Umgebungs-Stress-Level)
#
# ✅ TEMPORAL CONTEXT (Sequence-Based):
#    • 3-Frame History → Velocity & Trend-Erkenntnisse
#    • LSTM sieht: "Bin ich schneller geworden?" "Drehe ich ab?"
#    • Impliziter Action-Konsistenz-Druck
#
# ❌ NICHT SICHTBAR (Privacy/Decentralization):
#    • Andere Agenten exakte Position, Ziel, Absicht
#    • Globale Agenten-IDs (nur Relevanz-Ranking)
#    • Zentrale Dispatcher-Befehle
#    • Zukünftige Pläne anderer Agenten
#
# ═══════════════════════════════════════════════════════════════════════════
# 5. DESIGN RATIONALE FÜR 0.136→? PLATEAU-BREAK
# ═══════════════════════════════════════════════════════════════════════════
#
# PROBLEM: Policy lernt nur 1-2 Agenten pro Episode durchzulassen
#          (13-14% = ~0.7/5 Agenten), nicht alle 5 → Plateau
#
# ROOT CAUSES:
#   1. Reward-Signal zu sparsam (nur ALL_DONE_BONUS wenn ALLE im Ziel)
#   2. KL-Guards/Ratio-Guards blockieren zu viele Updates (Pw=0.26)
#   3. Keine inkrementellen Zwischen-Ziele
#
# FIXES IN DIESER VERSION:
#   ✨ INDIVIDUAL_DONE_BONUS = 1.5
#      → Jeder Agent der ins Ziel kommt, bekommt sofort +1.5 Reward
#      → Bricht "einer reicht" Plateau durch positive Reinforcement
#      → Gleichzeitig: ALL_DONE_BONUS = 18.0 belohnt Teamplay
#
#   ⚡ AGGRESSIVE KL/RATIO SETTINGS
#      → ppo_target_kl: 0.05 → 0.10 (2x weniger restriktiv)
#      → ratio_guard_soft: 1.10 → 1.20
#      → ratio_guard_hard: 1.20 → 1.40
#      → Ziel: Pw >= 0.50 (statt 0.26) → mehr Lernupdates pro Batch
#
#   🔊 EXPLORATION BOOST
#      → weight_entropy: 0.012 → 0.035 (3x)
#      → max_eps_random: 0.03 → 0.10
#      → surrogate_eps_clip: 0.15 → 0.22
#      → Ziel: Policy exploriert aggressiver neue Koordinations-Patterns
#
# EXPECTED OUTCOME:
#   • Batch-Akzeptanz Pw: 0.26 → 0.50-0.60 (mehr Updates)
#   • Done-Ratio: 0.136 → 0.20-0.30 (inkrementeller Progress)
#   • Grad_Norm: 0.86 → weiterhin ~0.8-0.9 (Gradienten fließen)
#   • Episode 2040+: done-ratio sollte kontinuierlich steigen



# Log-Out description während dem lernen
Hier ist eine klare Lesehilfe zu deinem PPO-Log.

Was bedeutet jede Spalte?

Loss: Gesamtziel, das minimiert wird. Mischung aus Policy, Value, Entropy, Aux, Comm.
P_Loss: Policy-Update-Term (Clipped PPO). Klein und wechselnd um 0 ist normal.
V_Loss: Critic-Fehler (Wertfunktion). Je kleiner/stabiler, desto besser kann PPO lernen.
E_Loss: Entropie-Term (negativ, weil als -entropy geloggt). Mehr negativ = mehr Exploration.
Adiv: Action-Diversity-Term. Bei dir 0.0000, weil deaktiviert (korrekt bei Forward-dominanter Domain).
AuxDL: Aux-Deadlock-Loss (BCE). Sinkend ist gut, zeigt besseres Deadlock-Signal-Lernen.
C_Loss: Communication-Regularizer/aux-Komponente. Sollte stabil bleiben, nicht explodieren.
Adv: mean±std: rohe Advantages vor Normierung. Wichtig ist, dass std nicht kollabiert.
Ratio: PPO-Wichtigkeit ( r_t = \exp(\log \pi_\theta - \log \pi_{\theta_{old}}) ). Ideal nahe 1.
KL: Distanz alte vs neue Policy. Zentrale Stabilitätsmetrik.
Clip: PPO-Clip-Epsilon.
Pw: effektives Policy-Gewicht nach Stabilitäts-Guards (wenn sehr niedrig, lernt Actor kaum).
H: Entropie (wenn geloggt). Bei 5 Aktionen ist Maximum ( \log(5)\approx1.61 ).
Was ist „gut“ (Faustwerte für dein Setup)?

Ratio: meist in etwa 0.9–1.1, kurze Ausreißer okay.
KL: grob < 0.03–0.05 stabil; > 0.08 häufig ist kritisch.
V_Loss: sollte über Zeit sinken oder wenigstens nicht trendmäßig steigen.
AuxDL: sollte langsam sinken.
Pw: nicht dauerhaft stark gedrosselt (sonst kaum Policy-Fortschritt).
Done-Ratio: der wichtigste KPI auf Episode-Ebene, muss langfristig steigen.
Was sieht man in deinem konkreten Log?

Adiv=0.0000: passt zur Entscheidung, Forward nicht global zu bestrafen.
V_Loss liegt oft bei ~0.67–0.90: nicht katastrophal, aber noch recht hoch.
Ratio hat mehrere harte Peaks (bis ~1.47): zu große Policy-Sprünge in einzelnen Batches.
KL hat öfter hohe Peaks (~0.10–0.14): klares Stabilitätsproblem in Teilen des Updates.
P_Loss bleibt klein: normal, aber mit hohem KL/Ratio heißt das eher „sprunghaft statt sauber“.
AuxDL ~0.34–0.53: okay, aber noch nicht „satt“ gelernt.
Wo liegen aktuell die Probleme?

Hauptproblem ist Update-Stabilität (KL/Ratio-Spikes), nicht Entropie.
Wenn diese Spikes oft kommen, wird der Actor durch Guards gedrosselt und kommt schwer aus lokalen Optima.
Das passt zu deinem beobachteten Verhalten: zeitweise Fortschritt, dann wieder Plateau.
Kurzregel zum schnellen Lesen künftig

Erst Done-Ratio und Action stats.
Dann KL + Ratio checken.
Wenn KL häufig >0.08 oder Ratio oft >1.25/<0.8, ist Stabilität das Bottleneck.
Danach V_Loss-Trend prüfen (Critic gut genug?).
Wenn du willst, mache ich dir als nächsten Schritt ein kompaktes Ampel-Schema direkt im Logformat (GRÜN/GELB/ROT pro Batch), damit du beim Laufen sofort siehst, ob ein Update gesund ist.

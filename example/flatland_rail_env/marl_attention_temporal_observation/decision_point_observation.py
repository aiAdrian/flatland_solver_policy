"""
DecisionPointObservation: 70D Observation für Multi-Agent Railway Scheduling
==============================================================================

ARCHITEKTUR-ÜBERSICHT
───────────────────

Diese Klasse generiert für jeden Agenten eine **35-dimensionale Merkmalsrepräsentation**
(Modus A Pure: nur essenzielle Features), zusammen mit einer vollständigen lokalen
Baum-Struktur für trainierbare Encoder (GNN/Transformer/LSTM).

    Flatland Rail Grid
           ↓
    Agent [pos, dir, target]
           ↓
    DecisionPointObservation.get(handle)
    ├─→ [0-5] Immediate Context: is_switch, direction_hint, is_merge, local_deadlock
    ├─→ tree_payload = _local_search(pos, dir, depth=5)
    │   ├─ nodes: [pos, dir, depth, num_transitions, deadlock_risk, agents_encountered]
    │   ├─ edges: [src→dst, rel_dir, agents_on_edge]
    │   └─ seen_agents: [sorted opponent IDs from tree search]
    ├─→ [6-28] State/Action Memory: train_state, last_action, priority_rank, cell_type
    └─→ [29-34] Tree Statistics: mean_deadlock, max_deadlock, conflict_density, branching_ratio
           ↓
    (raw_features[35], opponent_agents)
           ↓
    TemporalMultiAgentObservation (3-step buffer)
    ├─ Nutzt opponent_agents zum Ranking (Top-K nach Relevanz)
    ├─ Nutzt tree_payload für strukturierte Agenten-Kontext
    └─ Returns: [(ego_features, relevant_opponent_features), ...]
           ↓
    MAPPO Policy Network
    ├─ LSTM: (batch, 3, 35) → (batch, 64) hidden state
    ├─ Attention: ego-Features + opponent-Features → cross-attention weights
    └─ Heads: 64-dim → (5-dim action logits, 1-dim value estimate)
           ↓
    Action Sampling & Training

MODUS A PURE — Trainable Encoder Architektur:
───────────────────────────────────────────────
Modus A Pure bedeutet:
- Nur essenzielle 35D Features für immediate context + state/action memory
- Alle Branch-Metrics (Modus B) wurden entfernt
- Alle Merge-Heuristics (Modus B) wurden entfernt
- Trainable Encoder hat direkten Zugriff auf tree_payload
  → nodes/edges können mit Graph Neural Network (GNN) oder Transformer verarbeitet werden
  → Encoder lernt selbst optimal, wie man Branching-Struktur nutzt
  
Vorteil:
- Saubere Separation: Handcrafted Features ↔ Learnable Tree Encoding
- Encoder ist nicht limited durch handcrafted Branch Features
- Trainable Encoder kann novel patterns entdecken
- Einfacher zu erweitern (z.B. mit Graph-Attention)


DATENFLUSS: Von DecisionPointObservation zu MAPPO Training
──────────────────────────────────────────────────────────

1. get(handle) wird pro Agent & Timestep aufgerufen
2. Lokale Baumsuche (DFS, depth=5) sammelt strukturierte Daten:
   - nodes: Position, Richtung, Deadlock-Risiko, sichtbare Agenten
   - edges: Verbindungen, relative Richtungen, Agenten auf Kanten
   - seen_agents: Sortierte Liste entdeckter Gegner
   - visited_states: Alle besuchten (pos, dir, depth)-Tupel
   
3. tree_payload wird in env.dev_tree_dict[handle] gespeichert
   → Vollständig verfügbar für trainierbare Graph-Encoder
   → Encoder (z.B. Graph-Attention) könnte direkt nodes/edges nutzen
   
4. 35D raw_features werden zurückgegeben:
   → [0-5]: Immediate Context (is_switch, merge, deadlock)
   → [6-28]: State/Action Memory (agent state, last action, priority, cell type)
   → [29-34]: Tree-Statistiken (aggregierte Deadlock/Conflict-Metriken)
   
5. seen_agents wird an agent.cur_opp_agent_handles übergeben
   → TemporalMultiAgentObservation nutzt diese zum Ranking
   → "Welche der sichtbaren Gegner sind relevant für diesen Timestep?"
   
6. TemporalMultiAgentObservation puffert 3 Timesteps:
   → Input: [(ego_35D, opp_agents_list), ...] pro Timestep
   → Output: [(ego_t-2, opponent_t-2), (ego_t-1, opponent_t-1), (ego_t, opponent_t)]
   → LSTM eingang: (3, 35) für ego, Top-K (3, 35) pro opponent
   
7. LSTM/Transformer encodiert: (3, 35) → 64-dim hidden state
   → Erfasst zeitliche Muster (Deadlock-Risiko steigt? Gegner näher?)
   
8. Policy-Head erzeugt Action-Logits; Value-Head schätzt Zukunftswert
   
9. PPO-Update optimiert die Gewichte basierend auf Rewards

OPTIONAL: Trainable Graph-Encoder für Tree-Struktur
─────────────────────────────────────────────────────
Statt features [29-34] (aggregierte Statistiken) könnte ein trainable Graph-Encoder
die nodes/edges direkt verarbeiten:

  tree_payload.nodes  ──┐
  tree_payload.edges  ──┼─→ Graph Attention / GNN ──→ 64-dim tree_embedding
  tree_payload.seen_agents ──┘
  
Dann: ego_35D + tree_embedding → Policy/Value Heads
Vorteil:
- Encoder lernt optimal Branching-Struktur (nicht handcrafted)
- Gegner-Kontext direkt aus edges kodiert
- Flexibel für neue Szenarien


FEATURE-LAYOUT: 40D Dimensionen (Modus A Pure)
────────────────────────────────────────────

Modus A nutzt ausschließlich:
1. Immediate Context [0-5]
2. State/Action Memory [6-28]
3. Tree Statistics [29-34]

Alle Branch-Metrics und Merge-Heuristics werden vom trainable Encoder
aus tree_payload.nodes/edges direkt verarbeitet.

[0-5]       Switch & Merge Topology (6 Features) — Immediate Context
  [0]       is_switch: 1.0 falls Agent auf Switch, sonst 0
  [1-3]     shortest_path_hint: One-Hot für beste Richtung (L/F/R)
  [4]       is_merge: 1.0 falls nächster Schritt vor Merge
  [5]       local_deadlock: Binary Korridorblockade-Risiko

[6-28]      State & Action Memory (23 Features)
  [6-12]    train_state: One-Hot (7 states: READY, MOVING, ...)
  [13-17]   last_action: One-Hot (5 actions: DN, L, F, R, STOP)
  [18]      priority_rank: Normalisiert (0-1) nach verbleibender Distanz
  [19-23]   cell_type: One-Hot (5 types: OUTSIDE, SWITCH, MERGE, FWD, DONE)
  [24-28]   transitions: Pattern-Bits für Zelltyp-Übergänge

[29-34]     Tree Statistics (6 Features) ← TRAINABLE ENCODER INPUT
  [29]      mean_deadlock_risk: Durchschnitt aus lokaler Baumsuche
  [30]      confirmed_deadlock: 1.0 falls kritische Blockade erkannt
  [31]      mean_deadlock_risk: (Duplikat für Redundanz)
  [32]      max_deadlock_risk: Worst-case in lokalem Fenster
  [33]      conflict_density: Agenten-Begegnungen pro Knoten
  [34]      branching_ratio: Durchschnittliche Übergänge pro Knoten

[41-63]     State & Action Memory (23 Features)
  [41-47]   train_state: One-Hot (7 states: READY, MOVING, ...)
  [48-52]   last_action: One-Hot (5 actions: DN, L, F, R, STOP)
  [53]      priority_rank: Normalisiert (0-1) nach verbleibender Distanz
  [54-58]   cell_type: One-Hot (5 types: OUTSIDE, SWITCH, MERGE, FWD, DONE)
  [59-63]   transitions: Pattern-Bits für Zelltyp-Übergänge

[64-69]     Tree Statistics (6 Features) ← TRAINABLE ENCODER INPUT
  [64]      mean_deadlock_risk: Durchschnitt aus lokaler Baumsuche
  [65]      confirmed_deadlock: 1.0 falls kritische Blockade erkannt
  [66]      mean_deadlock_risk: (Duplikat für Redundanz)
  [67]      max_deadlock_risk: Worst-case in lokalem Fenster
  [68]      conflict_density: Agenten-Begegnungen pro Knoten
  [69]      branching_ratio: Durchschnittliche Übergänge pro Knoten


TREE PAYLOAD: Struktur der Baumsuche-Ausgabe
──────────────────────────────────────────────

tree_payload = {
    "nodes": [
        {
            "pos": (row, col),
            "dir": direction,
            "depth": depth_in_search,
            "num_transitions": count_outgoing_edges,
            "deadlock_risk": float 0.0-1.0,
            "agents_encountered": [agent_id_1, agent_id_2, ...],
            "has_oncoming": bool,
            "incoming_agents": [agent_id_3, ...],
            "backward_inflow_count": int
        },
        ...
    ],
    "edges": [
        {
            "src_pos": (r, c),
            "src_dir": direction,
            "dst_pos": (r', c'),
            "dst_dir": direction',
            "src_depth": int,
            "dst_depth": int,
            "rel_dir_bin": 0|1|2,  # Left/Forward/Right
            "edge_len_cells": 1,
            "agents_on_edge": [agent_ids],
            "has_oncoming_edge": bool
        },
        ...
    ],
    "seen_agents": [sorted_agent_ids],  ← GEGNER in lokaler Umgebung entdeckt!
    "visited_states": [(r, c, dir, depth), ...]
}

Speicherort: env.dev_tree_dict[handle] 
→ Verfügbar für Encoders, die strukturierte Tree-Daten nutzen


SEEN_AGENTS: Gegner-Erkennung in der lokalen Baumsuche
────────────────────────────────────────────────────────

tree_payload["seen_agents"] ist eine **sortierte Liste aller Agent-IDs**, die während
der lokalen Tiefensuche entdeckt wurden. Diese agents werden in zwei Kontexten gefunden:

1. **Im Knoten** (Agent befindet sich auf einer Position):
   - nodes[i]["agents_encountered"]: Gegner auf Knoten (i)
   - nodes[i]["incoming_agents"]: Gegner, die in diesen Knoten einfahren können
   
2. **Auf Kanten** (Agent bewegt sich zwischen Knoten):
   - edges[j]["agents_on_edge"]: Gegner auf der Kante zwischen (i) → (j)

Diese werden alle gesammelt in tree_payload["seen_agents"] → sortierte List[int]

**Datenfluss**:
  tree_payload["seen_agents"]
         ↓
  get() Rückgabe: (70D features, opponent_agents)
         ↓
  agent.cur_opp_agent_handles = opponent_agents
         ↓
  TemporalMultiAgentObservation:
    - Nutzt opponent_agents zum Ranking (welche Gegner sind relevant?)
    - Puffert Top-K Gegner über 3 Timesteps
    - Returns: [(ego_features, [relevant_opponent_handles]), ...]
         ↓
  MAPPO Multi-Agent Attention:
    - Ego-Features → Policy/Value Heads
    - Opponent-Features → Cross-Attention Module
    - Policy lernt: "Gibt es Gegner hier? Wie nah sind sie?"

**Trainbar?**
- seen_agents werden NICHT direkt in 70D-Features kodiert (keine One-Hots)
- Stattdessen: tree_payload.nodes/edges enthalten Agenten-Kontext
- TemporalMultiAgentObservation macht das Ranking
- MAPPO-Attention macht die Gewichtsanpassung
→ **Vollständig trainierbar über End-to-End MAPPO!**

**Warum nicht in [0-69] kodiert?**
- 70D ist schon eng (Agent-IDs sind dynamisch, Agent-Anzahl variabel)
- Stattdessen: Strukturelle Agenten-Information in tree_payload
  - nodes[i].agents_encountered → "Agenten auf Knoten i"
  - edges[j].agents_on_edge → "Agenten auf Kante j"
- Trainable Encoder entscheidet selbst, wie relevant diese sind
- TemporalMultiAgentObservation kümmert sich um Cross-Agent Ranking


TREE PAYLOAD: Struktur der Baumsuche-Ausgabe
──────────────────────────────────────────────

tree_payload = {
    "nodes": [
        {
            "pos": (row, col),
            "dir": direction,
            "depth": depth_in_search,
            "num_transitions": count_outgoing_edges,
            "deadlock_risk": float 0.0-1.0,
            "agents_encountered": [agent_id_1, agent_id_2, ...],
            "has_oncoming": bool,
            "incoming_agents": [agent_id_3, ...],
            "backward_inflow_count": int
        },
        ...
    ],
    "edges": [
        {
            "src_pos": (r, c),
            "src_dir": direction,
            "dst_pos": (r', c'),
            "dst_dir": direction',
            "src_depth": int,
            "dst_depth": int,
            "rel_dir_bin": 0|1|2,  # Left/Forward/Right
            "edge_len_cells": 1,
            "agents_on_edge": [agent_ids],
            "has_oncoming_edge": bool
        },
        ...
    ],
    "seen_agents": [sorted_agent_ids],  ← GEGNER in lokaler Umgebung entdeckt!
    "visited_states": [(r, c, dir, depth), ...]
}

Speicherort: env.dev_tree_dict[handle] 
→ Verfügbar für Encoders, die strukturierte Tree-Daten nutzen


INTEGRATION MIT MAPPO TRAINING (35D Features + Tree Payload)
────────────────────────────────────────────────────────────

Die 35D Features + tree_payload fließen in folgender Pipeline:

1. Temporal Buffer (TemporalMultiAgentObservation):
   3 Timesteps à 35D → (3, 35) Tensor für LSTM-Eingabe
   (Tree-Payload kann zusätzlich über env.dev_tree_dict[handle] abgerufen werden)

2. LSTM-Encoder:
   (batch, 3, 35) → (batch, 64) hidden state
   → Erfasst zeitliche Trends (Deadlock-Risk steigt?)

3. Optional: Graph-Encoder für tree_payload
   tree_payload.nodes/edges → GNN → 64-dim tree_embedding
   ego_features + tree_embedding → consolidated representation

4. Multi-Agent Attention:
   Ego-Features + Top-K Opponent-Features → Cross-attention
   → "Agent 5 kommt näher, erhöhe Vorsicht"

5. Policy & Value Heads:
   64-dim hidden state → (5-dim action logits, 1-dim value estimate)

6. PPO Objective:
   L = L_policy + λ_v * L_value + β * L_entropy
   
   L_policy: Ratio-Clipping (verhindert zu große Policy-Sprünge)
   L_value:  MSE zwischen geschätztem & tatsächlichem Return
   L_entropy: Bonus für Exploration


MODUS A PURE — Trainable Tree Encoder Architektur
────────────────────────────────────────────────

**Current Implementation (Modus A Pure):**
- [0-34]: Handcrafted aber differentiable Features (essenziel)
- [0-5]: Immediate context (is_switch, direction_hint, is_merge, local_deadlock)
- [6-28]: Agent state/action memory (train_state, last_action, priority, cell_type, transitions)
- [29-34]: Aggregierte Tree-Statistiken (mean_deadlock, max_deadlock, conflict_density, branching_ratio)
- tree_payload: Vollständige lokale Baum-Struktur (nodes/edges/seen_agents)
  → Verfügbar für trainable Graph-Encoder, z.B.:
     - Graph Attention Networks (GAT)
     - Message-Passing Neural Networks (MPNN)
     - Transformer mit strukturiertem Input

**Trainable Processing:**
1. 35D features + tree_payload → LSTM/Transformer Encoder
2. Tree-Struktur kann optional durch GNN aufbereitet werden
3. Policy + Value Heads lernen end-to-end mit PPO

**Vorteil Modus A Pure:**
- Handcrafted Features sind minimal und essenziel
- Tree-Struktur ist interpretierbar und vollständig verfügbar
- Trainable Encoder hat maximale Flexibilität
- Klare Separation: Feature Engineering ↔ Learning

**ALLE Modus B CODE (deterministic TreeLSTM, Branch Features, Merge Heuristics) ENTFERNT.**


OPTIMIERUNGSTECHNIKEN (Modus A Pure)
──────────────────────────────────────

1. Feature-Skalierung: Alle Features normalisiert [0,1]
   → Stabilere Gradienten, schnelleres Lernen

2. Soft Deadlock-Encoding: sigmoid-ähnliche Funktion
   → 1/(1 + dl_distance/2.5)
   → Nähe zu Deadlock wird sanft stärker signalisiert

3. Priority Ranking: Relative Distanz zum Ziel
   → Agenten "wetteifern" fair, keine dominanten Agenten

4. Tree Statistics Normalisierung:
   - conflict_density = min(1.0, agents_on_node / 2.0)
   - branching_ratio = min(1.0, num_transitions / 3.0)
   → Raw tree-Metriken werden softmax-normalized für Stabilität
"""

from typing import List

import numpy as np
import os

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax
from .decision_point_utils import DecisionPointUtils


_UNREACHABLE = -1.0


class DecisionPointObservation(ObservationBuilder):
    """
    Beobachtungs-Builder für Flatland (Modus A Pure — Trainable Encoder).
    
    Generiert 35D-Features + tree_payload(nodes/edges) für trainierbare Encoder-Integration.
    
    Modus A Pure (All Handcrafted Branch/Merge Features Removed):
    - [0-5]: Immediate Context (is_switch, direction_hint, is_merge, local_deadlock)
    - [6-28]: State/Action Memory (train_state, last_action, priority, cell_type, transitions)
    - [29-34]: Tree Statistics (mean_deadlock, max_deadlock, conflict_density, branching_ratio)
    - tree_payload.nodes/edges: Direkt verfügbar für trainable Graph-Encoder (GNN/Transformer)
    
    Alle Modus B Code (deterministische Branch Navigation, Merge Heuristics) wurde entfernt.
    
    Methoden:
    - __init__: Initialisiert die Klasse (Modus A).
    - set_env: Setzt die Umgebung.
    - reset: Initialisiert die Agentenkarte.
    - _local_search: Tiefensuche (5 Schritte) → nodes/edges-Struktur.
    - get: Gibt (35D features, opponent_agents) zurück.
    - get_many: Batch-Version von get().
    
    OUTPUT-FORMAT:
    get(handle) → (raw_features[35D], opponent_agents)
                  + env.dev_tree_dict[handle] = tree_payload mit nodes/edges/seen_agents
    """

    OBS_SIZE = 155  # Modus A Pure: 35 base + 15 nodes × 8 node features (6 node + 2 edge)
    NODE_DIM = 8    # Features per tree node: 6 node + 2 incoming edge (see _serialize_tree_nodes)
    MAX_NODES = 15  # Max nodes in DFS sequence (15 × 8 = 120D; padding with zeros)
    FEATURE_GROUPS_DOC = [
        ("[0]",    "is_switch",       "1.0 if agent is on switch cell"),
        ("[1-3]",  "hint_L/F/R",      "shortest-path direction hint (one-hot)"),
        ("[4]",    "is_merge",        "1.0 if agent is one step before merge node"),
        ("[5]",    "local_deadlock",  "binary corridor blockage/deadlock risk"),
        ("[6-12]", "st_0..st_6",      "TrainState one-hot (READY, MOVING, ..., DONE)"),
        ("[13-17]","act_DN/L/F/R/S",  "last saved action one-hot"),
        ("[18]",   "priority_rank",   "normalized rank by remaining distance"),
        ("[19-23]","ct_*",            "current cell-type one-hot"),
        ("[24-28]","tr_*",            "5 selected transitions"),
        ("[29]",   "mean_deadlock",   "mean deadlock risk from tree search"),
        ("[30]",   "confirmed_deadlock","1.0 if confirmed corridor deadlock"),
        ("[31]",   "mean_deadlock_dup","mean deadlock risk (duplicate for stability)"),
        ("[32]",   "max_deadlock",    "max deadlock risk in local window"),
        ("[33]",   "conflict_density","agent encounters per node"),
        ("[34]",   "branching_ratio", "mean transitions per node"),
        ("[35-154]","tree_nodes[15×8]",
         "15 DFS-ordered nodes: [dl_risk, norm_trans, oncoming, inflow, norm_depth, "
         "has_agents, incoming_rel_dir, edge_has_agents]"),
    ]

    def __init__(self,
                 debug: bool = False,
                 search_depth: int = 5,
                 observation_profile: str = "local_tree_encoder",
                 use_trainable_tree_encoder: bool = True):
        super().__init__()
        if debug:
            os.environ["DEBUG_OBSERVATION"] = "1"
        # Core observation configuration used throughout get()/local search.
        self.search_depth = max(1, int(search_depth))
        self.observation_profile = observation_profile
        self.use_trainable_tree_encoder = bool(use_trainable_tree_encoder)
        # Local-tree search control to avoid branch explosion at higher depths.
        # Up to depth 1: expand all transitions.
        # From depth >= 2: always keep shortest-path branch and sample side branches.
        self.local_search_random_start_depth = 2
        self.local_search_max_side_branches = 1
        self.local_search_distance_bias = 2.0
        # Optional advanced controls for deeper searches.
        self.local_search_mode = "stochastic"  # stochastic | mcts
        self.local_search_mcts_rollouts = 6
        self.local_search_mcts_horizon = 4
        self.local_search_ucb_c = 1.2
        self.local_search_contract_depth = 7
        self.local_search_max_nodes = 48
        self.local_search_min_nodes = 24
        self.local_search_adaptive_budget = True
        self.local_search_adaptive_branch_bonus = 6
        self.local_search_adaptive_conflict_bonus = 8
        self.local_search_adaptive_depth_bonus = 2
        self.local_search_deadlock_probe_depth = 6
        self.local_search_deadlock_max_states = 64
        self.local_tree_clip_features = True
        self.env = None
        self.agent_map = None
        self._print_feature_layout_doc()

    @staticmethod
    def _serialize_tree_nodes(
        tree_data: list,
        tree_edges: list = None,
        depth_limit: int = 5,
        clip_to_unit: bool = True,
    ) -> np.ndarray:
        """Serialize DFS-ordered tree nodes + incoming edge features into obs[35:155].

        Nodes must be in DFS pre-order (as produced by _local_search via frontier.pop()).
        Each node carries 8D including the incoming edge so the LSTM can reconstruct
        the branching topology from the sequence order alone.

        Returns:
            np.float32 array of shape (MAX_NODES * NODE_DIM,) = (120,)
            DFS pre-order; padding rows beyond len(tree_data) are zero.

        Node feature layout (8D per node):
            [0] deadlock_risk        float 0-1
            [1] norm_transitions     num_transitions / 3.0
            [2] has_oncoming         binary 0/1  (oncoming agent at this node)
            [3] norm_inflow          backward_inflow_count / 2.0
            [4] norm_depth           depth / depth_limit (5.0)
            [5] has_agents           1 if agents_encountered else 0
            [6] incoming_rel_dir     rel_dir_bin / 2.0: 0=left, 0.5=fwd, 1=right (0.5 for root)
            [7] edge_has_agents      1 if agents on incoming edge else 0  (0 for root)
        """
        MAX_N = DecisionPointObservation.MAX_NODES
        NODE_D = DecisionPointObservation.NODE_DIM  # 8
        arr = np.zeros(MAX_N * NODE_D, dtype=np.float32)

        # Build lookup: (dst_pos, dst_dir) → most-informative incoming edge
        edge_lookup = {}
        if tree_edges:
            for e in tree_edges:
                key = (e["dst_pos"], int(e["dst_dir"]))
                existing = edge_lookup.get(key)
                # Keep edge with agents if present (more informative)
                if existing is None or len(e.get("agents_on_edge", [])) > len(existing.get("agents_on_edge", [])):
                    edge_lookup[key] = e

        for i, node in enumerate(tree_data[:MAX_N]):
            base = i * NODE_D
            depth = int(node.get("depth", 0))
            arr[base + 0] = float(node.get("deadlock_risk", 0.0))
            arr[base + 1] = min(1.0, float(node.get("num_transitions", 1)) / 3.0)
            arr[base + 2] = 1.0 if node.get("has_oncoming", False) else 0.0
            arr[base + 3] = min(1.0, float(node.get("backward_inflow_count", 0)) / 2.0)
            safe_depth_limit = max(1, int(depth_limit))
            arr[base + 4] = min(1.0, depth / float(safe_depth_limit))
            arr[base + 5] = 1.0 if len(node.get("agents_encountered", [])) > 0 else 0.0
            # Incoming edge features (root at depth=0 has no incoming edge)
            if depth > 0:
                key = (node["pos"], int(node["dir"]))
                incoming = edge_lookup.get(key)
                if incoming is not None:
                    arr[base + 6] = float(incoming.get("rel_dir_bin", 1)) / 2.0
                    arr[base + 7] = 1.0 if len(incoming.get("agents_on_edge", [])) > 0 else 0.0
                else:
                    arr[base + 6] = 0.5  # default: forward if edge not found
            else:
                arr[base + 6] = 0.5  # root: no turn (encode as forward)
        if bool(clip_to_unit):
            np.clip(arr, 0.0, 1.0, out=arr)
        return arr

    def set_env(self, env):
        super().set_env(env)
        self.env = env

    def reset(self):
        self.agent_map = np.zeros((self.env.height, self.env.width), dtype=np.int32) - 1

    @staticmethod
    def _dir_to_rel_bin(current_dir: int, next_dir: int) -> int:
        """Map absolute next direction to relative bin: left=0, forward=1, right=2, other=3."""
        if current_dir is None or next_dir is None:
            return 3
        delta = (int(next_dir) - int(current_dir)) % 4
        if delta == 3:
            return 0
        if delta == 0:
            return 1
        if delta == 1:
            return 2
        return 3

    def _safe_distance(self, handle, position, direction, distance_map, default=np.inf):
        if distance_map is None:
            return default
        try:
            return float(distance_map[handle, position[0], position[1], direction])
        except Exception:
            return default

    def _mcts_rollout_score(self, handle, start_pos, start_dir, start_depth, horizon, distance_map):
        """Small Monte-Carlo rollout score for one root branch.

        Uses a light UCT-style policy over local successor choices to keep
        selection robust while staying compute-bounded.
        """
        if self.env is None or self.env.rail is None:
            return -1e9

        pos = start_pos
        direction = int(start_dir)
        score = 0.0
        max_steps = max(1, int(horizon))

        for _ in range(max_steps):
            dist = self._safe_distance(handle, pos, direction, distance_map)
            if np.isfinite(dist):
                score += 1.0 / (1.0 + dist)
            else:
                score -= 0.05

            if self.agent_map is not None:
                try:
                    other_idx = int(self.agent_map[pos])
                    if other_idx != -1 and other_idx != handle:
                        score -= 0.7
                        other_dir = self.env.agents[other_idx].direction
                        if other_dir is not None and DecisionPointUtils.is_opposite_direction(direction, other_dir):
                            score -= 0.8
                except Exception:
                    pass

            try:
                transitions = self.env.rail.get_transitions(*pos, direction)
            except Exception:
                score -= 0.5
                break

            choices = [nd for nd in range(4) if transitions[nd]]
            if not choices:
                score -= 1.0
                break

            # Soft distance-biased stochastic rollout policy.
            cand = []
            for nd in choices:
                np_pos = get_new_position(pos, nd)
                ndist = self._safe_distance(handle, np_pos, nd, distance_map)
                cand.append((nd, np_pos, ndist))
            dvals = np.array([c[2] if np.isfinite(c[2]) else 1000.0 for c in cand], dtype=np.float64)
            dmin = float(np.min(dvals))
            closeness = 1.0 / (1.0 + np.maximum(0.0, dvals - dmin))
            alpha = max(0.1, float(self.local_search_distance_bias))
            weights = np.power(closeness, alpha)
            wsum = float(np.sum(weights))
            probs = (weights / wsum) if np.isfinite(wsum) and wsum > 0 else np.full(len(cand), 1.0 / len(cand))
            pick = int(np.random.choice(len(cand), p=probs))
            direction = int(cand[pick][0])
            pos = cand[pick][1]

        # Small depth penalty so shorter informative branches are slightly preferred.
        score -= 0.02 * float(start_depth)
        return score

    def _contract_corridor_segment(self, handle, pos, direction, depth, depth_limit):
        """Compress linear corridor steps into one edge after contract depth.

        Stops contraction at decision points, conflicts, or depth limit.
        """
        if self.env is None or self.env.rail is None:
            return pos, int(direction), 1

        contract_depth = max(0, int(self.local_search_contract_depth))
        if depth < contract_depth:
            return pos, int(direction), 1

        cur_pos = pos
        cur_dir = int(direction)
        edge_len = 1

        while (depth + edge_len) < depth_limit:
            if self.agent_map is not None:
                try:
                    other_idx = int(self.agent_map[cur_pos])
                    if other_idx != -1 and other_idx != handle:
                        break
                except Exception:
                    break

            try:
                transitions = self.env.rail.get_transitions(*cur_pos, cur_dir)
            except Exception:
                break

            next_dirs = [nd for nd in range(4) if transitions[nd]]
            if len(next_dirs) != 1:
                break

            nd = int(next_dirs[0])
            next_pos = get_new_position(cur_pos, nd)
            cur_pos = next_pos
            cur_dir = nd
            edge_len += 1

            if edge_len >= 6:
                break

        return cur_pos, cur_dir, edge_len

    def _compute_adaptive_node_budget(
        self,
        handle,
        start_pos,
        start_dir,
        depth_limit,
        transition_cache,
        incoming_degree_cache,
    ):
        """Compute per-step node budget for local search.

        The budget is reduced in simple scenes and increased near conflicts,
        while staying bounded in [min_nodes, max_nodes].
        """
        max_nodes = max(8, int(getattr(self, "local_search_max_nodes", 48)))
        min_nodes = max(8, int(getattr(self, "local_search_min_nodes", 24)))
        if min_nodes > max_nodes:
            min_nodes = max_nodes

        if not bool(getattr(self, "local_search_adaptive_budget", True)):
            return max_nodes

        def _get_transitions_cached(pos, direction):
            key = (int(pos[0]), int(pos[1]), int(direction))
            if key in transition_cache:
                return transition_cache[key]
            trans = self.env.rail.get_transitions(int(pos[0]), int(pos[1]), int(direction))
            transition_cache[key] = trans
            return trans

        budget = min_nodes
        try:
            root_trans = _get_transitions_cached(start_pos, start_dir)
            branch_count = int(fast_count_nonzero(root_trans))
        except Exception:
            branch_count = 1

        branch_bonus_unit = max(0, int(getattr(self, "local_search_adaptive_branch_bonus", 6)))
        budget += branch_bonus_unit * max(0, branch_count - 1)

        depth_bonus_unit = max(0, int(getattr(self, "local_search_adaptive_depth_bonus", 2)))
        budget += depth_bonus_unit * max(0, int(depth_limit) - 6)

        start_key = (int(start_pos[0]), int(start_pos[1]))
        try:
            if start_key in incoming_degree_cache:
                in_deg = incoming_degree_cache[start_key]
            else:
                in_deg = self._incoming_degree(start_pos, transition_cache=transition_cache)
                incoming_degree_cache[start_key] = in_deg
        except Exception:
            in_deg = 0

        if in_deg > 1:
            conflict_bonus = max(0, int(getattr(self, "local_search_adaptive_conflict_bonus", 8)))
            budget += conflict_bonus

        if self.agent_map is not None:
            try:
                start_agent = int(self.agent_map[start_pos])
                if start_agent != -1 and start_agent != handle:
                    budget += max(0, int(getattr(self, "local_search_adaptive_conflict_bonus", 8)))
            except Exception:
                pass

        return max(min_nodes, min(max_nodes, int(budget)))

    def _select_local_search_branches(self, handle, depth, current_pos, transitions, distance_map):
        """Select branches for local search with depth-aware stochastic pruning.

        Strategy:
        - depth < random_start_depth: keep all valid branches
        - depth >= random_start_depth:
          1) always keep shortest branch (distance-map)
          2) sample limited side branches with short-branch bias
        """
        candidates = []
        for next_dir in range(4):
            if not transitions[next_dir]:
                continue
            next_pos = get_new_position(current_pos, next_dir)
            dist = np.inf
            if distance_map is not None:
                try:
                    dist = float(distance_map[handle, next_pos[0], next_pos[1], next_dir])
                except Exception:
                    dist = np.inf
            candidates.append((next_dir, next_pos, dist))

        if len(candidates) <= 1:
            return candidates

        if depth < int(self.local_search_random_start_depth):
            return candidates

        finite_dists = [c[2] for c in candidates if np.isfinite(c[2])]
        fallback_large = (max(finite_dists) + 1.0) if finite_dists else 1.0
        normalized = []
        for c in candidates:
            d = c[2] if np.isfinite(c[2]) else fallback_large
            normalized.append((c[0], c[1], d))

        normalized.sort(key=lambda x: x[2])
        shortest = normalized[0]
        side = normalized[1:]

        k_side = min(int(self.local_search_max_side_branches), len(side))
        if k_side <= 0:
            return [shortest]

        # Optional MCTS-lite root action selection (flat UCT at current node).
        if str(getattr(self, "local_search_mode", "stochastic")).lower() == "mcts":
            rollout_budget = max(1, int(getattr(self, "local_search_mcts_rollouts", 6)))
            rollout_horizon = max(1, int(getattr(self, "local_search_mcts_horizon", 4)))
            ucb_c = max(0.01, float(getattr(self, "local_search_ucb_c", 1.2)))

            stats = {}
            for cand in normalized:
                stats[cand[0]] = {"visits": 0, "value": 0.0, "cand": cand}

            for _ in range(rollout_budget):
                total_visits = sum(v["visits"] for v in stats.values()) + 1
                best_dir = None
                best_ucb = -1e18
                for dkey, rec in stats.items():
                    v = rec["visits"]
                    mean = (rec["value"] / v) if v > 0 else 0.0
                    ucb = mean + ucb_c * np.sqrt(np.log(float(total_visits)) / float(v + 1))
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_dir = dkey

                selected = stats[best_dir]["cand"]
                roll_score = self._mcts_rollout_score(
                    handle=handle,
                    start_pos=selected[1],
                    start_dir=selected[0],
                    start_depth=depth + 1,
                    horizon=rollout_horizon,
                    distance_map=distance_map,
                )
                stats[best_dir]["visits"] += 1
                stats[best_dir]["value"] += float(roll_score)

            ordered = sorted(
                normalized,
                key=lambda c: ((stats[c[0]]["value"] / max(1, stats[c[0]]["visits"])), -c[2]),
                reverse=True,
            )
            chosen = ordered[: 1 + k_side]
            if shortest[0] not in [c[0] for c in chosen]:
                chosen = [shortest] + chosen[:k_side]
            return chosen

        dvals = np.array([s[2] for s in side], dtype=np.float64)
        dmin = float(np.min(dvals))
        closeness = 1.0 / (1.0 + np.maximum(0.0, dvals - dmin))
        alpha = max(0.1, float(self.local_search_distance_bias))
        weights = np.power(closeness, alpha)
        wsum = float(np.sum(weights))
        if wsum <= 0.0 or not np.isfinite(wsum):
            probs = np.full(len(side), 1.0 / len(side), dtype=np.float64)
        else:
            probs = weights / wsum

        idx = np.random.choice(len(side), size=k_side, replace=False, p=probs)
        chosen_side = [side[int(i)] for i in np.atleast_1d(idx)]
        return [shortest] + chosen_side

    def _local_search(self, handle, start_pos, start_dir, depth_limit):
        """Run a bounded local graph search around one agent and emit tree payload.

        Purpose:
            Build a structured, variable-size neighborhood graph that captures
            switch/merge topology, nearby agents, and deadlock cues for the
            trainable tree encoder path.

        Args:
            handle: current ego agent id.
            start_pos: agent position (row, col).
            start_dir: agent direction in {0,1,2,3}.
            depth_limit: maximum exploration depth in rail-cell steps.

        Search mechanics:
            1) Frontier-based traversal with best-depth pruning per state
               (state = position + direction). A state is expanded only if it is
               reached at a strictly better (smaller) depth.
            2) Adaptive node budget via `_compute_adaptive_node_budget(...)` limits
               total expanded nodes per call to keep runtime bounded.
            3) At each expanded state, transitions are read once via cache, then
               candidate branches are selected by `_select_local_search_branches(...)`.
               The shortest-path successor is always retained; side branches are
               sampled/ranked depending on local-search mode.
            4) Optional corridor contraction via `_contract_corridor_segment(...)`
               compresses linear tracks into one edge while preserving edge length
               (`edge_len_cells`).
            5) Deadlock signal per node is computed by `_calculate_deadlock_risk(...)`
               and adjusted by oncoming and backward-inflow bonuses.

        Returns:
            dict with keys:
              - nodes: list[dict], one entry per visited local state
                fields: pos, dir, depth, num_transitions, deadlock_risk,
                        agents_encountered, has_oncoming, incoming_agents,
                        backward_inflow_count
              - edges: list[dict], directed local transitions
                fields: src_pos/src_dir/src_depth, dst_pos/dst_dir/dst_depth,
                        rel_dir_bin, edge_len_cells, agents_on_edge,
                        has_oncoming_edge
              - seen_agents: sorted list[int] of all opponents observed in nodes,
                incoming scans, or edges
              - visited_states: list[(row, col, dir, depth)] in expansion order

        Notes:
            - Output size is intentionally variable (node/edge counts differ per
              timestep and agent).
            - On any severe failure, a safe empty payload is returned.
        """
        try:
            if start_pos is None or start_dir is None or self.env is None or self.env.rail is None:
                print(f"[Warn] _local_search: Ungültige Startdaten für Agent {handle}.")
                return {"nodes": [], "edges": [], "seen_agents": [], "visited_states": []}
            best_depth_by_state = {}
            frontier = [(start_pos, start_dir, 0)]
            tree_nodes = []
            tree_edges = []
            seen_agents = set()
            visited_states = []
            transition_cache = {}
            incoming_degree_cache = {}
            incoming_agents_cache = {}
            deadlock_cache = {}

            def _get_transitions_cached(pos, direction):
                key = (int(pos[0]), int(pos[1]), int(direction))
                if key in transition_cache:
                    return transition_cache[key]
                trans = self.env.rail.get_transitions(int(pos[0]), int(pos[1]), int(direction))
                transition_cache[key] = trans
                return trans

            distance_map = None
            try:
                distance_map = self.env.distance_map.get()
            except Exception:
                distance_map = None
            max_nodes = self._compute_adaptive_node_budget(
                handle=handle,
                start_pos=start_pos,
                start_dir=start_dir,
                depth_limit=depth_limit,
                transition_cache=transition_cache,
                incoming_degree_cache=incoming_degree_cache,
            )
            while frontier:
                current_pos, current_dir, depth = frontier.pop()
                state_key = (int(current_pos[0]), int(current_pos[1]), int(current_dir))
                prev_best = best_depth_by_state.get(state_key)
                if depth > depth_limit or (prev_best is not None and depth >= prev_best):
                    continue
                if len(tree_nodes) >= max_nodes:
                    break
                best_depth_by_state[state_key] = int(depth)
                visited_states.append((int(current_pos[0]), int(current_pos[1]), int(current_dir), int(depth)))
                try:
                    transitions = _get_transitions_cached(current_pos, current_dir)
                except Exception as e:
                    print(f"[Warn] _local_search: Fehler bei get_transitions: {e}")
                    continue
                num_transitions = fast_count_nonzero(transitions)
                agents_encountered = []
                has_oncoming = False
                incoming_agents = []
                if self.agent_map is not None:
                    try:
                        agent_idx = self.agent_map[current_pos]
                        if agent_idx != -1 and agent_idx != handle:
                            agents_encountered.append(agent_idx)
                            seen_agents.add(int(agent_idx))
                            other_dir = self.env.agents[agent_idx].direction
                            if other_dir is not None and DecisionPointUtils.is_opposite_direction(current_dir, other_dir):
                                has_oncoming = True
                    except Exception as e:
                        print(f"[Warn] _local_search: Fehler bei agent_map: {e}")
                try:
                    pos_key = (int(current_pos[0]), int(current_pos[1]))
                    if pos_key in incoming_degree_cache:
                        in_deg = incoming_degree_cache[pos_key]
                    else:
                        in_deg = self._incoming_degree(current_pos, transition_cache=transition_cache)
                        incoming_degree_cache[pos_key] = in_deg

                    if in_deg > 1:
                        ia_key = (pos_key[0], pos_key[1], int(handle))
                        if ia_key in incoming_agents_cache:
                            incoming_agents = incoming_agents_cache[ia_key]
                        else:
                            incoming_agents = self._incoming_agent_handles(
                                current_pos,
                                handle,
                                transition_cache=transition_cache,
                            )
                            incoming_agents_cache[ia_key] = incoming_agents
                        for a in incoming_agents:
                            seen_agents.add(int(a))
                except Exception as e:
                    print(f"[Warn] _local_search: Fehler bei incoming-agent scan: {e}")
                try:
                    if state_key in deadlock_cache:
                        base_risk = deadlock_cache[state_key]
                    else:
                        base_risk = self._calculate_deadlock_risk(
                            handle,
                            current_pos,
                            current_dir,
                            max_depth=int(getattr(self, "local_search_deadlock_probe_depth", 6)),
                            max_states=int(getattr(self, "local_search_deadlock_max_states", 64)),
                            transition_cache=transition_cache,
                        )
                        deadlock_cache[state_key] = base_risk
                except Exception as e:
                    print(f"[Warn] _local_search: Fehler bei _calculate_deadlock_risk: {e}")
                    base_risk = 1.0
                inflow_bonus = 0.15 * min(2, len(incoming_agents))
                adjusted_risk = min(1.0, base_risk + (0.5 if has_oncoming else 0.0) + inflow_bonus)
                node_info = {
                    "pos": current_pos,
                    "dir": current_dir,
                    "depth": depth,
                    "num_transitions": num_transitions,
                    "deadlock_risk": adjusted_risk,
                    "agents_encountered": agents_encountered,
                    "has_oncoming": has_oncoming,
                    "incoming_agents": incoming_agents,
                    "backward_inflow_count": len(incoming_agents),
                }
                tree_nodes.append(node_info)
                selected = self._select_local_search_branches(
                    handle=handle,
                    depth=depth,
                    current_pos=current_pos,
                    transitions=transitions,
                    distance_map=distance_map,
                )
                for next_dir, next_pos, _dist in selected:
                    final_pos, final_dir, edge_len = self._contract_corridor_segment(
                        handle=handle,
                        pos=next_pos,
                        direction=next_dir,
                        depth=depth + 1,
                        depth_limit=depth_limit,
                    )
                    next_depth = min(int(depth_limit), int(depth + edge_len))
                    if next_depth <= depth:
                        next_depth = depth + 1
                    edge_agents = []
                    if self.agent_map is not None:
                        try:
                            nidx = self.agent_map[final_pos]
                            if nidx != -1 and nidx != handle:
                                edge_agents.append(int(nidx))
                                seen_agents.add(int(nidx))
                        except Exception as e:
                            print(f"[Warn] _local_search: Fehler bei edge-agent scan: {e}")
                    tree_edges.append({
                        "src_pos": current_pos,
                        "src_dir": int(current_dir),
                        "dst_pos": final_pos,
                        "dst_dir": int(final_dir),
                        "src_depth": int(depth),
                        "dst_depth": int(next_depth),
                        "rel_dir_bin": self._dir_to_rel_bin(current_dir, next_dir),
                        "edge_len_cells": int(edge_len),
                        "agents_on_edge": edge_agents,
                        "has_oncoming_edge": bool(len(edge_agents) > 0),
                    })
                    frontier.append((final_pos, final_dir, next_depth))
            return {
                "nodes": tree_nodes,
                "edges": tree_edges,
                "seen_agents": sorted(seen_agents),
                "visited_states": visited_states,
            }
        except Exception as e:
            print(f"[Warn] _local_search: Schwerwiegender Fehler: {e}")
            return {"nodes": [], "edges": [], "seen_agents": [], "visited_states": []}

    def _calculate_deadlock_risk(self, handle, pos, direction, max_depth=6, max_states=64, transition_cache=None):
        """Defensive Deadlock-Risk-Berechnung: Gibt bei Fehlern Risiko=1.0 zurück."""
        try:
            if pos is None or direction is None or self.env is None or self.env.rail is None:
                print(f"[Warn] _calculate_deadlock_risk: Ungültige Eingaben für Agent {handle}.")
                return 1.0

            if transition_cache is None:
                transition_cache = {}

            def _get_transitions_cached(cell_pos, cell_dir):
                key = (int(cell_pos[0]), int(cell_pos[1]), int(cell_dir))
                if key in transition_cache:
                    return transition_cache[key]
                trans = self.env.rail.get_transitions(int(cell_pos[0]), int(cell_pos[1]), int(cell_dir))
                transition_cache[key] = trans
                return trans

            visited = set()
            frontier = [(pos, direction, 0)]
            deadlock_risk = 0.0
            while frontier:
                if len(visited) >= max(8, int(max_states)):
                    break
                current_pos, current_dir, depth = frontier.pop()
                if depth > max(1, int(max_depth)):
                    continue
                if (current_pos, current_dir) in visited:
                    continue
                visited.add((current_pos, current_dir))
                try:
                    transitions = _get_transitions_cached(current_pos, current_dir)
                except Exception as e:
                    print(f"[Warn] _calculate_deadlock_risk: Fehler bei get_transitions: {e}")
                    deadlock_risk += 1.0
                    continue
                num_transitions = fast_count_nonzero(transitions)
                if num_transitions == 0:
                    deadlock_risk += 1.0
                elif num_transitions > 1:
                    deadlock_risk += 0.5
                for next_dir in range(4):
                    if transitions[next_dir]:
                        next_pos = get_new_position(current_pos, next_dir)
                        frontier.append((next_pos, next_dir, depth + 1))
            return min(deadlock_risk / 10.0, 1.0)
        except Exception as e:
            print(f"[Warn] _calculate_deadlock_risk: Schwerwiegender Fehler: {e}")
            return 1.0

    @staticmethod
    def getObservationSize() -> int:
        return DecisionPointObservation.OBS_SIZE

    @classmethod
    def _print_feature_layout_doc(cls):
        if os.getenv("DEBUG_OBSERVATION", "0") == "1":
            print(">> DecisionPointObservation (Modus A Pure) — 35D Feature-Layout:")
            for idx, name, desc in cls.FEATURE_GROUPS_DOC:
                print(f"   {idx:<8} {name:<18} {desc}")

    @staticmethod
    def _encode_detect_deadlock(raw: float) -> float:
        return 1.0 if raw > 0 else 0.0

    @staticmethod
    def _encode_deadlock_signal(deadlock_distance: float) -> float:
        if deadlock_distance is None or deadlock_distance <= 0:
            return 0.0
        # Steeper decay: nearby deadlocks become more prominent, which helps
        # the policy separate "slightly risky" from "immediate danger".
        return min(1.0, 1.0 / (1.0 + deadlock_distance / 2.5))

    @staticmethod
    def _cell_type_index_from_decision_type(decision_type: int) -> int:
        if decision_type & 8:
            return 4
        if decision_type == 1:
            return 0
        if decision_type & 2:
            return 3
        if decision_type & 4:
            return 2
        return 1

    def _is_switch_at_current_cell(self, pos, direction) -> bool:
        """True if the agent stands on a switching cell right now."""
        transitions = self.env.rail.get_transitions(*pos, direction)
        return fast_count_nonzero(transitions) > 1

    def _incoming_degree(self, cell_pos, transition_cache=None) -> int:
        """Count incoming directed edges to a cell by local 4-neighborhood scan."""
        if transition_cache is None:
            transition_cache = {}

        def _get_transitions_cached(pos, direction):
            key = (int(pos[0]), int(pos[1]), int(direction))
            if key in transition_cache:
                return transition_cache[key]
            trans = self.env.rail.get_transitions(int(pos[0]), int(pos[1]), int(direction))
            transition_cache[key] = trans
            return trans

        incoming_edges = set()
        for prev_dir in range(4):
            prev_pos = get_new_position(cell_pos, (prev_dir + 2) % 4)
            if prev_pos[0] < 0 or prev_pos[0] >= self.env.height or prev_pos[1] < 0 or prev_pos[1] >= self.env.width:
                continue
            for d in range(4):
                trans = _get_transitions_cached(prev_pos, d)
                for nd in range(4):
                    if not trans[nd]:
                        continue
                    npos = get_new_position(prev_pos, nd)
                    if npos == cell_pos:
                        incoming_edges.add((prev_pos[0], prev_pos[1], d, nd))
        return len(incoming_edges)

    def _incoming_agent_handles(self, cell_pos, handle_exclude: int, transition_cache=None) -> list:
        """Collect agents that can enter cell_pos through an incoming directed edge."""
        if self.agent_map is None:
            return []

        if transition_cache is None:
            transition_cache = {}

        def _get_transitions_cached(pos, direction):
            key = (int(pos[0]), int(pos[1]), int(direction))
            if key in transition_cache:
                return transition_cache[key]
            trans = self.env.rail.get_transitions(int(pos[0]), int(pos[1]), int(direction))
            transition_cache[key] = trans
            return trans

        found = set()
        for prev_dir in range(4):
            prev_pos = get_new_position(cell_pos, (prev_dir + 2) % 4)
            if prev_pos[0] < 0 or prev_pos[0] >= self.env.height or prev_pos[1] < 0 or prev_pos[1] >= self.env.width:
                continue
            try:
                agent_idx = int(self.agent_map[prev_pos])
            except Exception:
                continue
            if agent_idx == -1 or agent_idx == handle_exclude:
                continue
            try:
                a_dir = self.env.agents[agent_idx].direction
                if a_dir is None:
                    continue
                trans = _get_transitions_cached(prev_pos, a_dir)
                for nd in range(4):
                    if not trans[nd]:
                        continue
                    if get_new_position(prev_pos, nd) == cell_pos:
                        found.add(agent_idx)
                        break
            except Exception:
                continue
        return sorted(found)

    def _is_pre_merge_one_exit(self, pos, direction, transitions) -> bool:
        """True if agent is exactly one step before a merge/conflict node with one current exit.

        Semantics for DAG-style routing:
        - current cell: exactly one usable outgoing edge for the current heading
        - next cell: true merge node, i.e. receives multiple incoming edges and
          has a single onward edge for the arriving orientation
        """
        if fast_count_nonzero(transitions) != 1:
            return False
        ndir = fast_argmax(transitions)
        if not transitions[ndir]:
            return False
        # "Nur forward" am aktuellen Knoten: kein Links/Rechts-Entscheid mehr möglich.
        if ndir != direction:
            return False
        next_pos = get_new_position(pos, ndir)
        if next_pos[0] < 0 or next_pos[0] >= self.env.height or next_pos[1] < 0 or next_pos[1] >= self.env.width:
            return False
        in_deg = self._incoming_degree(next_pos)
        if in_deg <= 1:
            return False
        next_transitions_arrival = self.env.rail.get_transitions(*next_pos, ndir)
        return fast_count_nonzero(next_transitions_arrival) == 1

    def _decision_type_at_position(self, pos, direction, target) -> int:
        if pos == target:
            return 8
        transitions = self.env.rail.get_transitions(*pos, direction)
        decision_type = 0
        if self._is_switch_at_current_cell(pos, direction):
            decision_type += 2
        if self._is_pre_merge_one_exit(pos, direction, transitions):
            decision_type += 4
        return decision_type

    def get(self, handle: int = 0):
        """
        Generiere 35D Observation für einen einzelnen Agenten (Modus A Pure).
        
        Pipeline:
        1. Immediate Context [0-5]: is_switch, direction_hint, is_merge, local_deadlock
        2. Lokale Baumsuche (DFS depth=5) → tree_payload
        3. State/Action Memory [6-28]: agent state, last action, priority, cell type
        4. Tree Statistics [29-34]: aggregierte Deadlock/Conflict-Metriken
        5. Speichere tree_payload in env.dev_tree_dict[handle]
        6. Rückgabe: (35D features, opponent_agents_list)
        
        WICHTIG: Alle Branch-Metrics [6-29] und Merge-Heuristics [30-40] wurden entfernt.
        Trainable Encoder nutzen tree_payload direkt (nodes/edges/seen_agents).
        
        Args:
            handle: Agent ID
            
        Returns:
            (raw_features[35], opponent_agents): 
                - raw_features: np.float32[35] normalisiert [0,1]
                - opponent_agents: List[int] sichtbare Gegner aus tree_payload
                
        Side Effects:
            - env.dev_tree_dict[handle] = tree_payload mit vollständiger Struktur
              (nodes/edges/seen_agents für trainable encoder)
            - agent.cur_opp_agent_handles = opponent_agents
        """
        raw_features = np.zeros(DecisionPointObservation.OBS_SIZE, dtype=np.float32)
        try:
            agent = self.env.agents[handle]
            pos = agent.position if agent.position is not None else agent.initial_position
            direction = agent.direction if agent.direction is not None else agent.initial_direction
            target = agent.target
            if pos is None or target is None or direction is None:
                print(f"[Warn] get: Ungültige Agenten-Startdaten für {handle}.")
                return (raw_features, [])
            distance_map = self.env.distance_map.get()
            curr_dist_raw = distance_map[handle, pos[0], pos[1], direction]
            curr_reachable = bool(np.isfinite(curr_dist_raw))
            max_dist = self._max_dist
            # IMPORTANT: np.inf distance is a valid planning state in Flatland.
            # We do not log warnings for it; instead we encode it in features below.
            curr_dist_norm = (float(curr_dist_raw) / max_dist) if curr_reachable else 1.0
            try:
                transitions = self.env.rail.get_transitions(*pos, direction)
            except Exception as e:
                print(f"[Warn] get: Fehler bei get_transitions: {e}")
                transitions = [0, 0, 0, 0]
            # Lokale Suche → Baum-Payload für trainierbare Encoder-Integration
            tree_payload = self._local_search(handle, pos, direction, self.search_depth)
            tree_data = tree_payload.get("nodes", [])
            local_search_seen_agents = set(tree_payload.get("seen_agents", []))
            
            merge_switch = False
            try:
                merge_switch = self._is_pre_merge_one_exit(pos, direction, transitions)
            except Exception as e:
                print(f"[Warn] get: Fehler bei _is_pre_merge_one_exit: {e}")
            decision_type = 0
            if agent.state.name == "READY_TO_DEPART":
                decision_type = 1
            else:
                if self._is_switch_at_current_cell(pos, direction):
                    decision_type += 2
                if merge_switch:
                    decision_type += 4
            if agent.state.name == "DONE":
                decision_type = 8
            raw_features[0] = 1.0 if (decision_type & 2) else 0.0
            try:
                raw_features[1:4] = self._shortest_path_action_hint(handle, pos, direction, transitions, distance_map)
            except Exception as e:
                print(f"[Warn] get: Fehler bei _shortest_path_action_hint: {e}")
                raw_features[1:4] = 0.0
            raw_features[4] = 1.0 if (decision_type & 4) else 0.0
            try:
                raw_features[5] = self._encode_detect_deadlock(self._detect_deadlock(handle, pos, direction))
            except Exception as e:
                print(f"[Warn] get: Fehler bei _detect_deadlock: {e}")
                raw_features[5] = 0.0
            all_distance = []
            for idx, a in enumerate(self.env.agents):
                apos = a.position if a.position is not None else a.initial_position
                adir = a.direction if a.direction is not None else a.initial_direction
                if apos is None or adir is None:
                    adist = np.inf
                else:
                    try:
                        adist = float(distance_map[a.handle, apos[0], apos[1], adir])
                    except Exception as e:
                        print(f"[Warn] get: Fehler bei distance_map für Agent {a.handle}: {e}")
                        adist = np.inf
                all_distance.append((a.handle, adist, idx))
            all_distance.sort(key=lambda x: (x[1], x[2]))
            value_to_rank = {}
            next_rank = 1
            handle_to_rank = {}
            for h, dist, _ in all_distance:
                if dist not in value_to_rank:
                    value_to_rank[dist] = next_rank
                    next_rank += 1
                handle_to_rank[h] = value_to_rank[dist]
            priority_rank = float(handle_to_rank.get(handle, next_rank)) / next_rank
            
            opp_agents = set()
            opp_agents.update(local_search_seen_agents)
            for other in self.env.agents:
                if other.handle == handle:
                    continue
                other_pos = other.position if other.position is not None else other.initial_position
                if other_pos == pos:
                    opp_agents.add(other.handle)
            
            # State/Action Memory [6-28]
            state_value = int(agent.state.value)
            if 0 <= state_value <= 6:
                raw_features[6 + state_value] = 1.0
            
            # Last-action one-hot with behavior-aware fallback.
            sa = None
            on_map = getattr(agent, "position", None) is not None
            if on_map and agent.action_saver.is_action_saved:
                sa = int(agent.action_saver.saved_action)
            elif on_map:
                is_moving = bool(getattr(agent, "moving", False))
                if not is_moving:
                    sa = 4  # STOP_MOVING
                else:
                    old_dir = getattr(agent, "old_direction", None)
                    cur_dir = getattr(agent, "direction", None)
                    if old_dir is None or cur_dir is None:
                        sa = 2  # MOVE_FORWARD
                    else:
                        delta = (int(cur_dir) - int(old_dir)) % 4
                        if delta == 0:
                            sa = 2  # FORWARD
                        elif delta == 1:
                            sa = 3  # RIGHT
                        elif delta == 3:
                            sa = 1  # LEFT
                        else:
                            sa = 2
            if sa is not None and 0 <= sa <= 4:
                raw_features[13 + sa] = 1.0
            
            raw_features[18] = priority_rank
            curr_idx = self._cell_type_index_from_decision_type(decision_type)
            raw_features[19 + curr_idx] = 1.0
            
            _TR_SLOTS = {(1, 1): 24, (1, 2): 25, (1, 3): 26, (3, 1): 27, (2, 1): 28}
            next_decision_type = decision_type
            try:
                if decision_type & 8:
                    next_decision_type = 8
                elif decision_type == 1:
                    next_decision_type = self._decision_type_at_position(pos, direction, target)
                else:
                    if fast_count_nonzero(transitions) > 0:
                        forward_dir = fast_argmax(transitions)
                        next_pos = get_new_position(pos, forward_dir)
                        next_decision_type = self._decision_type_at_position(next_pos, forward_dir, target)
            except Exception as e:
                print(f"[Warn] get: Fehler bei next_decision_type: {e}")
                next_decision_type = decision_type
            next_idx = self._cell_type_index_from_decision_type(next_decision_type)
            tr_slot = _TR_SLOTS.get((curr_idx, next_idx), None)
            if tr_slot is not None:
                raw_features[tr_slot] = 1.0
            
            try:
                raw_features[30] = 1.0 if DecisionPointUtils.is_local_deadlock(self.env, agent, self.agent_map) else 0.0
            except Exception as e:
                print(f"[Warn] get: Fehler bei is_local_deadlock: {e}")
                raw_features[30] = 0.0
            
            if not curr_reachable:
                raw_features[5] = max(raw_features[5], 0.5)
                raw_features[18] = min(raw_features[18], 0.25)
            
            try:
                if not hasattr(self.env, 'dev_tree_dict'):
                    self.env.dev_tree_dict = {}
                self.env.dev_tree_dict[handle] = tree_payload
            except Exception as e:
                print(f"[Warn] get: Fehler bei dev_tree_dict.update: {e}")
            
            # Tree Statistics [29-34] + serialized nodes [35-154] for LocalTreeEncoder
            if tree_data:
                try:
                    _dl = [n["deadlock_risk"] for n in tree_data]
                    _cf = [min(1.0, len(n.get("agents_encountered", [])) / 2.0) for n in tree_data]
                    _br = [min(1.0, n.get("num_transitions", 1) / 3.0) for n in tree_data]
                    raw_features[29] = float(np.mean(_dl))
                    # [30] already set above with confirmed_deadlock
                    raw_features[31] = float(np.mean(_dl))
                    raw_features[32] = float(np.max(_dl))
                    raw_features[33] = float(np.mean(_cf))
                    raw_features[34] = float(np.mean(_br))
                    # Serialize tree nodes into obs[35:155] for LocalTreeEncoder
                    tree_flat = self._serialize_tree_nodes(
                        tree_data,
                        tree_payload.get("edges", []),
                        depth_limit=self.search_depth,
                        clip_to_unit=bool(getattr(self, "local_tree_clip_features", True)),
                    )
                    raw_features[35:35 + len(tree_flat)] = tree_flat
                except Exception as e:
                    print(f"[Warn] get: Fehler bei tree_data-Statistiken: {e}")
            
            agent.cur_opp_agent_handles = sorted(opp_agents)
            return (raw_features, agent.cur_opp_agent_handles)
        except Exception as e:
            print(f"[Warn] get: Schwerwiegender Fehler für Agent {handle}: {e}")
            return (raw_features, [])

    def get_many(self, handles: list = None):
        if handles is None:
            handles = list(range(len(self.env.agents)))

        self.agent_map = np.zeros((self.env.height, self.env.width), dtype=np.int32) - 1
        for agent in self.env.agents:
            if agent.position is not None:
                self.agent_map[agent.position] = agent.handle

        distance_map = self.env.distance_map.get()
        finite = distance_map[np.isfinite(distance_map)]
        if finite.size > 0:
            self._max_dist = max(float(np.max(finite)), 1.0)
        else:
            self._max_dist = 1.0

        for agent in self.env.agents:
            if not hasattr(agent, 'opp_agent_handles'):
                agent.opp_agent_handles = []
            if not hasattr(agent, 'cur_opp_agent_handles'):
                agent.cur_opp_agent_handles = []

        result = []
        for handle in handles:
            obs_self, obs_others = self.get(handle)
            result.append((obs_self, obs_others))

        for agent in self.env.agents:
            agent.opp_agent_handles = agent.cur_opp_agent_handles

        return result

    @staticmethod
    def _normalise_distance(value: float, max_dist: float) -> float:
        if value is None or value == _UNREACHABLE:
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if max_dist <= 0:
            return 0.0
        return float(np.clip(value / max_dist, 0.0, 1.0))

    @staticmethod
    def _normalise_count(value: float) -> float:
        if value is None or value < 0:
            return 0.0
        return float(value) / (float(value) + 8.0)

    def _shortest_path_action_hint(self, handle, pos, direction, transitions, distance_map):
        """Compute which direction (L/F/R) is best according to distance map."""
        best_hint = [0.0, 0.0, 0.0]
        min_dist = np.inf
        best_idx = None
        for idx, rel in enumerate((-1, 0, 1)):
            ndir = (direction + rel) % 4
            if transitions[ndir]:
                npos = get_new_position(pos, ndir)
                dist = distance_map[handle, npos[0], npos[1], ndir]
                if np.isfinite(dist) and dist < min_dist:
                    min_dist = dist
                    best_idx = idx
        if best_idx is not None:
            best_hint[best_idx] = 1.0
        return best_hint

    def _detect_deadlock(self, handle, pos, direction):
        """Detect confirmed corridor blockage before the next switch."""
        return DecisionPointUtils.detect_corridor_blockage(
            self.env,
            self.agent_map,
            handle,
            pos,
            direction,
            {handle},
            16,
            0,
        )

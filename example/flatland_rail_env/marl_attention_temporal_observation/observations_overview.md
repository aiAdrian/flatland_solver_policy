# Flatland Multi-Agent Observations – Übersicht & Feature-Design

## Update Mai 2026 (aktueller Stand)

Dieses Dokument wurde für den aktuellen Trainingsstand aktualisiert.

### TreeLSTM + Decider (Lernpfad und Attention)

Dieser Abschnitt beschreibt die genaue Einbindung des Tree-Moduls und welche
Teile trainierbar sind.

#### 1) Datenfluss in der Observation

1. In `DecisionPointObservation._local_search(...)` wird ab der Agent-Position
   ein lokaler Suchbaum bis `search_depth` aufgebaut.
2. Pro besuchtem Knoten werden u. a. gespeichert:
   - `depth`
   - `num_transitions`
   - `deadlock_risk`
   - `agents_encountered`
   - `has_oncoming`
3. Diese Knotenliste (`tree_data`) geht in `TreeLSTM.aggregate(tree_data)`.
4. Das Ergebnis wird als `tree_ctx` in Feature-Slot **[64]** geschrieben.

#### 2) Was lernt und was lernt NICHT?

- **Nicht trainierbar (deterministisch):** `marl_attention_temporal_observation/tree_lstm.py`
  - Das Modul ist ein fester Numpy-Aggregator (gewichtete Mittelung),
    keine Torch-Parameter, kein Backprop.
- **Trainierbar:** `TreeSpecialist` in `decider_policy.py`
  - Liest `tree_ctx` aus Slot [64] plus Sicherheitskontext ([5], [65], [4], [30]).
  - Erzeugt `tree_emb` + `tree_conf` und wird über PPO (AdamW) mitoptimiert.

#### 3) Wie kommt Tree-Kontext in die Policy-Entscheidung?

In `DeciderNetwork.forward(...)`:

1. `tree_emb, tree_conf = self.tree(self_seq[:, -1, :])`
2. `tree_conf` wird an die Skalare angehängt.
3. `tree_emb` wird in den Fusionsvektor aufgenommen.
4. Der fusionierte Vektor geht in Actor/Critic.

Damit beeinflusst `tree_ctx` direkt die Action-Logits und den Value.

#### 4) Attention-Anbindung der gesehenen Agenten

- Alle in `_local_search` angetroffenen Agenten (`agents_encountered`) werden
  in `get(...)` zu `local_search_seen_agents` gesammelt.
- Diese Handles werden in `opp_agents` gemerged und als
  `agent.cur_opp_agent_handles` zurückgegeben.
- Der temporale Wrapper / Hierarchical-Observation baut daraus die
  Opponent-Liste für die Kommunikations-Features.
- Die Comm-Schicht (`CommSpecialist`, MultiheadAttention) nutzt diese
  Opponent-Informationen für die sparse Nachbarschafts-Attention.

Kurzfassung:
- Tree-Aggregator liefert das Signal (`tree_ctx`).
- TreeSpecialist lernt die Nutzung dieses Signals.
- Opponent-Handles aus lokaler Suche fließen in den Attention-Kontext.

### Hauptobservationen (aktuelle Versionen)

1. **`DecisionPointObservation`** — **66D** Decision-Point-basierte Beobachtung:
    - **[0-5]**: Basis-Switches & Entscheidungstypen (is_switch, hints, is_merge, local_deadlock)
    - **[6-29]**: Branch-Metriken (3 Zweige × 8 Features: progress, deadlock_signal mit Inverse-Decay, distance, abort, target, etc.)
    - **[30]**: decision_required Flag (nur bei MERGING/SWITCH gesetzt für Policy-Gating)
    - **[31-40]**: Merge-Metriken (forward/backward sub-blocks mit inverse-decay deadlock_signal)
    - **[41-47]**: TrainState one-hot (7 Dimensionen) — **FIX Mai 2026: Jetzt immer korrekt gesetzt**
    - **[48-52]**: Last action one-hot (5 Aktionen) — **CRITICAL FIX Mai 2026: Bedingungslos gesetzt, nicht mehr im Guard-Block**
    - **[53]**: priority_rank (normalisierter Rang nach Restdistanz)
    - **[54-58]**: Cell-Type one-hot (OUTSIDE, FORWARD_ONLY, MERGING, SWITCH, DONE)
    - **[59-63]**: Transitions one-hot (5 Übergänge zwischen Zelltypen)
    - **[64]**: Reserved (ungenutzter Platzhalter)
    - **[65]**: **Confirmed Deadlock Flag** — 1.0 wenn Rückstau im voraus erkannt via `DecisionPointUtils.detect_corridor_blockage()`, 0.0 sonst

2. **`TemporalMultiAgentObservation`** — Zeitfenster über `DecisionPointObservation`:
    - Liefert Sequenz: `[(obs_{t-2}, opp_{t-2}), (obs_{t-1}, opp_{t-1}), (obs_t, opp_t)]` — je 66D observation
    - Nutzt Temporal Transformer/LSTM für Bewegungsmuster-Erkennung
    - Handle-sicheres Mapping (keine Index/Handle-Verwechselung mehr)

### Was hat sich seit April 2026?

**Mai 2026 — Kritische Architektur-Fixes für Kooperation & Deadlock-Erkennung:**

1. **Deadlock-Signal Encoding (Issue #1)** ✅
  - **ALT:** `_encode_deadlock_signal()` konvertierte Distanzen zu Binary (0/1)
  - **NEU:** Inverse-Decay Encoding: `1.0 / (1.0 + distance/4.0)`
  - **Impact:** Policy kann jetzt zwischen "Deadlock 1 Schritt weg" (Warnung 1.0) vs "Deadlock 16 Schritte weg" (Warnung 0.06) unterscheiden
  - **Betroffen:** Features [7, 15, 23, 31, 36] (deadlock_signal in allen Blöcken)

2. **Opponent-Awareness Vollständigkeit (Issue #2)** ✅
  - **ALT:** Nur Agenten auf Branches wurden gesammelt (visited during DFS)
  - **NEU:** Agenten am **aktuellen Knoten** werden explizit hinzugefügt vor Branch-Enumeration
  - **Impact:** Head-on Konflikte und Merge-Blockaden werden nun erkannt
  - **Code:** Neue Loop über `self.env.agents` um pos == current_pos zu prüfen (Zeile ~260)

3. **Backward-Merge Enumeration Robustheit (Issue #3)** ✅
  - **ALT:** `for d in range(1, 4)` — skipped direction 0, early break
  - **NEU:** `for d in range(4)` — alle Richtungen, kein early break
  - **Impact:** Merge-Topologie wird vollständig erfasst
  - **Code:** Zeile ~306-321 (decision_point_observation.py)

4. **Deadlock-Timeout Logik Sicherheit (Issue #4)** ✅
  - **ALT:** Bei Timeout (s >= max_steps): `return 0` (als safe behandelt — falsch!)
  - **NEU:** Bei Timeout: `return s` (als Deadlock-Verdacht behandelt)
  - **Impact:** Lange Korridore (>128 Schritte) werden nicht falsch als sicher klassifiziert
  - **Code:** decision_point_utils.py Zeile ~161-162

5. **Action Features Bug Fix (Critical)** ✅
  - **ALT:** Features [48-52] waren **immer 0** weil Guard `if agent.action_saver.is_action_saved:` immer False
  - **NEU:** Features werden **bedingungslos gesetzt** (Default DO_NOTHING wenn nicht gespeichert)
  - **Impact:** Critic kann jetzt Aktionsverteilungen lernen statt nur Konstante 0 zu sehen
  - **Code:** decision_point_observation.py Zeile ~333-340

6. **TrainState Features Konsistenz** ✅
  - **ALT:** Features [41-47] waren oft 0 (State nicht korrekt gespeichert)
  - **NEU:** State wird immer aus `agent.state.value` gespeichert (keine bedingten Guards mehr)
  - **Impact:** Network sieht Agent States konsistent
   
7. **Policy Decision-Point Gating** ✅
  - Feature [30] `decision_required` wird jetzt **nur bei echten Entscheidungen (MERGING/SWITCH) gesetzt**
  - MARL_ATT_DecisionPointPolicy nutzt dies um deterministische MOVE_FORWARD (OUTSIDE/FORWARD_ONLY) zu erzwingen
  - Sparse Reward-Signal wird nicht mehr auf nicht-Entscheidungszellen verschwendet

**Fazit:** Alle 4 Hauptprobleme (binäres Deadlock-Signal, unvollständige Opponent-Erkennung, Merge-Enumeration, Timeout-Logik) wurden behoben. Zusätzlich 2 kritische Feature-Bugs (Action one-hot, TrainState).

---

**Dezember 2024 - April 2026 — Vorherige Updates:**

- **[65] Deadlock-Feature:** Recursive cycle detection (nicht nur "head-on")
  - Folgt Blockade-Ketten um Zyklen zu erkennen
  - Max 128 Schritte Lookahead auf obligatorischen Korridoren
  - Penalisiert in Reward-Sharern
  
- **DecisionPointUtils:** Zentrale Utility-Klasse für Deadlock-Logik
  - Statische Methoden: `is_local_deadlock()`, `detect_corridor_blockage()`, `is_opposite_direction()`

### Deadlock-Detection Details

**Szenario-Erkennung:**
1. Agent läuft in obligatorischen Korridor (nur 1 Übergang pro Zelle)
2. Trifft auf anderen Agenten
3. Check:
   - **Gegenrichtung?** → Sofort Deadlock erkannt
   - **Gleiche Richtung, aber selbst blockiert?** → Rekursiv den blockierenden Agenten prüfen
   - **Bereits gesehen?** → Zyklus erkannt → Deadlock

**False-Positive-Vermeidung:**
- Schalter/Branches unterbrechen die Blockade-Verfolgung
- Blockierter Agent mit Fluchtroute: KEIN Deadlock
- Max 128 Schritte: Verhindert Endlosschleifen im Code

> **Klassen im Überblick**
> | Klasse | Größe | Zweck |
> |---|---|---|
> | `ExperimentalObservation` | 30D | Basisbeobachtung für jeden Agenten |
> | `SimplifiedPathThreeTierObservation` | 57D | Drei Pfad-Tiers (links/geradeaus/rechts) |
> | `DecisionPointObservation` | **66D** | Entscheidungsbasierte Beobachtung an Weichen/Merges + Deadlock-Flag |
> | `TemporalMultiAgentObservation` | T × 66D | Zeitfenster über `DecisionPointObservation` |
> | `DecisionPointUtils` | — | Shared Deadlock-Detection-Logic |

---

## 1. ExperimentalObservation

**Klasse:** `ExperimentalObservation(ObservationBuilder)`  
**Feature-Größe:** 30D  
**Abhängigkeiten:** `RailroadSwitchAnalyser`, `WalkToNextDecisionPoint`

Basisbeobachtung je Agent. Liefert einen 30-dimensionalen Feature-Vektor mit Positionsinformationen, Richtung, Ziel und Agentenstatus. Wird von `TemporalMultiAgentObservation` und `SimplifiedPathThreeTierObservation` verwendet.

**Wichtige Methoden:**
- `get_pos_dir(agent)` – Gibt aktuelle Position und Richtung zurück (fallback auf `initial_position/direction`)
- `get_decision_point_observation(...)` – Berechnet die 30 Features für einen Agenten

---

## 2. SimplifiedPathThreeTierObservation

**Klasse:** `SimplifiedPathThreeTierObservation(ObservationBuilder)`  
**Feature-Größe:** 9 (Header) + 3 × 16 (Pfad-Tiers) = **57D**

Teilt die Umgebung in drei Pfadsegmente auf: links, geradeaus, rechts. Jedes Segment hat identische 16D Feature-Blöcke. Nur bei tatsächlich vorhandenen Pfaden (Transitions) werden die Blöcke befüllt – sonst Nullvektor.

**Struktur:**
- **Header [0–8]:** Agentenstatus, Richtung, one-hot Best-Path-Hinweis, Anzahl Agenten
- **Links [9–24]:** 16 Features für den linken Pfad (nur bei Switch aktiv)
- **Geradeaus [25–40]:** 16 Features für den Vorwärtspfad (immer aktiv)
- **Rechts [41–56]:** 16 Features für den rechten Pfad (nur bei Switch aktiv)

---

## 3. DecisionPointObservation

**Klasse:** `DecisionPointObservation(ObservationBuilder)`  
**Feature-Größe:** **66D** (erweitert Mai 2026)
**Rückgabe von `get(handle)`:** `(features: np.array[66], opp_agent_handles: list[int])`

Spezialisierte Beobachtung für die drei zentralen Entscheidungssituationen im Flatland:

| `decision_type` | Wert (Bit) | Situation |
|---|---|---|
| Normal / immer geradeaus | `0` | Kein Entscheidungspunkt |
| Start | `1` | Agent ist `READY_TO_DEPART` |
| Switch | `2` (Bit) | Agent steht auf einer Weiche (mehrere Transitionen möglich) |
| Merge/Crossing | `4` (Bit) | Agent ist **eine Zelle vor** einer Einmündungs-Weiche |
| Switch + Merge | `6` | Beide Bits gesetzt |
| Done | `8` | Agent hat Ziel erreicht |

> Bits sind kombinierbar: `decision_type & 2` prüft Switch, `decision_type & 4` prüft Merge.

---

### Feature-Layout (66 Features — Mai 2026 Update)

#### Block A – Switch-Analyse (31-47, nur wenn `decision_type & 2`)

Für jede der drei Richtungen (L/F/R) ein **8er-Block**:

| Index | Base | Name | Beschreibung |
|---|---|---|---|
| 6-13 | 6 | `sw_L_*` | Linker Zweig: progress_gain, deadlock_signal, switches_norm, dist_norm, target_found, abort, deadlock_ahead, valid |
| 14-21 | 14 | `sw_F_*` | Geradeaus-Zweig: [wie Links] |
| 22-29 | 22 | `sw_R_*` | Rechter Zweig: [wie Links] |

**WICHTIG - Mai 2026 Update:** Die `deadlock_signal` Features (Indizes 7, 15, 23) verwenden jetzt **Inverse-Decay-Encoding** statt Binary:
```python
deadlock_signal = 1.0 / (1.0 + deadlock_distance / 4.0)
# Deadlock 1 Schritt weg → 1.0 (höchste Warnung)
# Deadlock 4 Schritte weg → 0.2
# Deadlock 16 Schritte weg → 0.06
# Kein Deadlock → 0.0
```
Dies ermöglicht dem Netzwerk, Deadlocks nach Entfernung zu differenzieren, statt sie als Binary zu behandeln.

#### Block B – Basis-Infos (0-5, immer)

| Index | Name | Beschreibung |
|---|---|---|
| 0 | `is_switch` | Binary: 1.0 wenn an Weiche, else 0.0 |
| 1-3 | `hint_L/F/R` | One-hot: optimale Richtung laut distance_map |
| 4 | `is_merge` | Binary: 1.0 wenn Merge-Zone voraus, else 0.0 |
| 5 | `local_deadlock` | Binary: 1.0 wenn Deadlock am aktuellen Knoten erkannt, else 0.0 |

#### Block C – Merge/Crossing-Analyse (31-40, nur wenn `decision_type & 4`)

| Index | Name | Beschreibung |
|---|---|---|
| 31 | `mgF_dl_sig` | Merge Forward: deadlock_signal (Inverse-Decay wie oben) |
| 32 | `mgF_switches` | Merge Forward: normalized switch count |
| 33 | `mgF_target` | Merge Forward: target found? (0/1) |
| 34 | `mgF_abort` | Merge Forward: DFS abbruch flag |
| 35 | `mgF_dl_ahead` | Merge Forward: deadlock_ahead binary |
| 36 | `mgB_dl_sig` | Merge Backward: deadlock_signal |
| 37 | `mgB_switches` | Merge Backward: normalized switch count |
| 38 | `mgB_target` | Merge Backward: target found? (0/1) |
| 39 | `mgB_abort` | Merge Backward: DFS abort flag |
| 40 | `mgB_dl_ahead` | Merge Backward: deadlock_ahead binary |

**WICHTIG - Mai 2026 Update:** Backward-Merge Enumeration wurde erweitert um alle 4 Richtungen zu prüfen (vorher nur 1-3), um Merge-Topologie vollständig zu erfassen.

#### Block D – Agent State (41-47, One-hot)

| Index | State | Beschreibung |
|---|---|---|
| 41 | `st_0` | READY_TO_DEPART |
| 42 | `st_1` | MALFUNCTION_OFF_MAP |
| 43 | `st_2` | MOVING |
| 44 | `st_3` | STOPPED |
| 45 | `st_4` | MALFUNCTION |
| 46 | `st_5` | WAITING |
| 47 | `st_6` | DONE |

**WICHTIG - Mai 2026 Fix:** Diese Features waren früher oft konstant 0 weil `agent.state` nicht korrekt gespeichert war. Nun werden sie immer gesetzt (Default auf st_0 falls unbekannt).

#### Block E – Last Action (48-52, One-hot)

| Index | Action | Beschreibung |
|---|---|---|
| 48 | `act_DN` | DO_NOTHING |
| 49 | `act_L` | MOVE_LEFT |
| 50 | `act_F` | MOVE_FORWARD |
| 51 | `act_R` | MOVE_RIGHT |
| 52 | `act_S` | STOP_MOVING |

**CRITICAL FIX - Mai 2026:** Diese Features waren früher **immer 0** weil der Guard `if agent.action_saver.is_action_saved:` immer False war. Jetzt:
- Default zu `act_DN` (Index 48 = 1.0) wenn keine Aktion gespeichert
- Features werden **bedingungslos gesetzt** (nicht mehr im Guard-Block)
- Dies ermöglicht dem Critic, Aktionsverteilungen zu lernen

#### Block F – Dezisionsmerkmale & Nebenfunktionen (53-65)

| Index | Name | Beschreibung |
|---|---|---|
| 53 | `priority_rank` | Normalized rank by remaining path distance [0,1] |
| 54-58 | `ct_*` | Cell Type One-hot: OUTSIDE, FORWARD_ONLY, MERGING, SWITCH, DONE |
| 59-63 | `tr_*` | 5 selected transitions: FWD→FWD, FWD→MRG, FWD→SWI, SWI→FWD, MRG→FWD |
| 64 | `reserved` | Placeholder (unused, kept for backward compatibility) |
| 65 | `deadlock` | **Confirmed deadlock flag: 1.0 wenn Rückstau im voraus erkannt, 0.0 sonst** |

#### Block G – Decision Required (30, zentral für Policy-Gating)

| Index | Name | Beschreibung |
|---|---|---|
| 30 | `decision_required` | **1.0 nur wenn (decision_type & 2) OR (decision_type & 4), else 0.0** |

**WICHTIG - Mai 2026 Policy Integration:** Dieses Feature wird von MARL_ATT_DecisionPointPolicy verwendet um zu entscheiden, ob echte Entscheidungen getroffen werden (MERGING/SWITCH) oder ob deterministische MOVE_FORWARD (OUTSIDE/FORWARD_ONLY) angewendet wird.

---

### DFS-Logik: `_navigate_direction`

Das Herzstück der Observation. Führt eine rekursive Tiefensuche (DFS) ab einer Startposition durch.

**Signatur:**
```python
_navigate_direction(handle, start_pos, start_dir, target, backward_trace, max_steps=100)
-> (dist, deadlock_flag, num_switches, seen_agents, abort_flag, target_found_flag, visited)
```

**Globaler DFS-Controller (pro Aufruf geteilt über alle Rekursionen):**
| Feld | Bedeutung |
|---|---|
| `count` | Gesamtzahl besuchter Zellen (Abbruch bei ≥ `max_steps`) |
| `visited` | Menge aller `(pos, dir)` Paare (Zyklenerkennung) |
| `seen_agents` | Alle auf dem Pfad gesehenen Agenten-Handles + **NEUE Mai 2026: auch Agenten am aktuellen Knoten** |

**Abbruchbedingungen (in Reihenfolge):**
1. `count >= max_steps` → `abort=1`
2. `pos == target` (nur forward) → `target_found=1`
3. `(pos, dir) in visited` → Zyklus, `deadlock=-1, abort=1`
4. `curr_dist == inf` (nur forward) → kein Pfad, `deadlock=-1, abort=1`

**Deadlock-Erkennung im DFS:**
- Agent auf nächster Zelle fährt **entgegengesetzt** → `deadlock=1` (Gegenverkehr, Stopp)
- Agent auf nächster Zelle ist **derselbe Agent** (handle==self) → `deadlock=2` (Selbst-Block bei Rückwärtssuche)

**Switch-Verhalten (mehrere Transitionen):**
- **Forward-Modus:** Alternativen werden nach `distance_map` sortiert, die **beste nicht-deadlockende** wird gewählt (Backtracking)
- **Backward-Modus:** **Alle** Alternativen werden verfolgt und die Ergebnisse **gemittelt** (modelliert Unsicherheit über fremde Entscheidungen)

**Lokale Deadlock-Erkennung (`_detect_deadlock`):**  
Wird separat aufgerufen (unabhängig von DFS). Prüft ob ein direkt benachbarter Agent entgegenkommt **und** beide Agenten nur vorwärts können (`forward_only`). Schreibt Ergebnis in `features[41]`.

---

### Methoden-Übersicht

| Methode | Beschreibung |
|---|---|
| `get(handle)` | 42D Feature-Vektor + Liste gegnerischer Agent-Handles |
| `get_many(handles)` | Ruft `get()` für alle Handles, aktualisiert `agent_map` und `opp_agent_handles` |
| `_shortest_path_action_hint(...)` | One-hot [links, geradeaus, rechts] für optimale Richtung via `distance_map` |
| `_navigate_direction(...)` | Rekursive DFS für Pfadmetriken |
| `_detect_deadlock(...)` | Lokale Deadlock-Prüfung (direkte Nachbarn, forward-only) |

---

## 4. TemporalMultiAgentObservation

**Klasse:** `TemporalMultiAgentObservation(ObservationBuilder)`  
**Feature-Größe:** T × 42D (default T=3, konfigurierbar via `temporal_window`)  
**Rückgabe von `get_many(handles)`:** Liste pro Agent, je T Einträge `(obs_42D, [list of opp_obs_42D])`

Hüllt eine beliebige Basis-Observation (Standard: `DecisionPointObservation`) in ein Zeitfenster. Pro Zeitschritt wird die aktuelle Beobachtung gespeichert und eine Sequenz der letzten T Schritte zurückgegeben. Fehlende History wird durch Wiederholen des ältesten Eintrags aufgefüllt.

```
Zeitschritt t:   seq = [(obs_t-2, opp_t-2), (obs_t-1, opp_t-1), (obs_t, opp_t)]
                         ↑ ältester                               ↑ aktuellster
```

**Gegner-Beobachtungen (`obs_others`):**  
Für jeden in `opp_agent_handles` gelisteten Agenten wird dessen 42D Basisobservation direkt aus dem aktuellen `get_many`-Ergebnis gelesen (kein separater Aufruf).

**Basis-Observation konfigurierbar:**
```python
TemporalMultiAgentObservation(temporal_window=3, base_obs='DecisionPointObservation')
# oder: base_obs=ExperimentalObservation()
# oder: base_obs=SimplifiedPathThreeTierObservation
```

---

## Zusammenfassung: Klassen-Vergleich

| Eigenschaft | `ExperimentalObservation` | `DecisionPointObservation` | `TemporalMultiAgentObservation` |
|---|---|---|---|
| Größe | 30D | **66D** (Mai 2026: erweitert) | T × 66D |
| Entscheidungslogik | Keine | DFS an Weichen/Merges | via Basis-Obs |
| Gegner-Info | Nein | Handles in Rückgabe | Obs-Vektoren aller Gegner |
| Zeitliche Tiefe | Nein | Nein | Ja (T Schritte) |
| Velocity-Features | Nein (statisch) | Nein (statisch) | Emergent (LSTM lernt aus Sequenz) |


### Kontext & Zielsetzung

- **Flatland-rl** ist eine Multi-Agenten-Umgebung zur Simulation von Zugverkehr und Konfliktlösung auf Schienennetzen. Die DecisionPointObservation abstrahiert die wichtigsten Entscheidungspunkte (Start, Weiche, Merge/Crossing) in einen RL-tauglichen, strukturierten Feature-Vektor.
- Die Features werden dynamisch und kontextsensitiv befüllt: Nur die für die aktuelle Entscheidungssituation relevanten Blöcke sind aktiv, alle anderen sind 0.
- Die Entscheidungslogik basiert auf rekursiver Tiefensuche (DFS) zur Pfadanalyse, Deadlockerkennung und Zielprüfung.


### Feature-Tabelle: DecisionPointObservation (alle 42 Features einzeln)

| Index | Name                        | Beschreibung                                                                 | decision_type (gesetzt bei)         |
|-------|-----------------------------|------------------------------------------------------------------------------|-------------------------------------|
| 0     | decision_type               | Entscheidungssituation (0=normal, 1=Start, 2=Switch, 4=Merge, 8=DONE)        | alle                                |
| 1     | onehot_left                 | 1, wenn links optimal (laut distance_map), sonst 0                           | Switch, Merge/Crossing              |
| 2     | onehot_forward              | 1, wenn geradeaus optimal, sonst 0                                           | Switch, Merge/Crossing              |
| 3     | onehot_right                | 1, wenn rechts optimal, sonst 0                                              | Switch, Merge/Crossing              |
| 4     | left_curr_dist              | Aktuelle Distanz (vor Schritt links)                                         | Switch                              |
| 5     | left_deadlock               | Deadlock-Flag nach Schritt links                                             | Switch                              |
| 6     | left_switches               | Anzahl Weichen nach Schritt links                                            | Switch                              |
| 7     | left_delta_dist             | Distanzdifferenz zum Ziel nach Schritt links                                 | Switch                              |
| 8     | left_target_found           | Ziel erreicht nach Schritt links                                             | Switch                              |
| 9     | left_abort                  | Abbruch-Flag nach Schritt links                                              | Switch                              |
| 10    | forward_curr_dist           | Aktuelle Distanz (vor Schritt geradeaus)                                     | Switch                              |
| 11    | forward_deadlock            | Deadlock-Flag nach Schritt geradeaus                                         | Switch                              |
| 12    | forward_switches            | Anzahl Weichen nach Schritt geradeaus                                        | Switch                              |
| 13    | forward_delta_dist          | Distanzdifferenz zum Ziel nach Schritt geradeaus                             | Switch                              |
| 14    | forward_target_found        | Ziel erreicht nach Schritt geradeaus                                         | Switch                              |
| 15    | forward_abort               | Abbruch-Flag nach Schritt geradeaus                                          | Switch                              |
| 16    | right_curr_dist             | Aktuelle Distanz (vor Schritt rechts)                                        | Switch                              |
| 17    | right_deadlock              | Deadlock-Flag nach Schritt rechts                                            | Switch                              |
| 18    | right_switches              | Anzahl Weichen nach Schritt rechts                                           | Switch                              |
| 19    | right_delta_dist            | Distanzdifferenz zum Ziel nach Schritt rechts                                | Switch                              |
| 20    | right_target_found          | Ziel erreicht nach Schritt rechts                                            | Switch                              |
| 21    | right_abort                 | Abbruch-Flag nach Schritt rechts                                             | Switch                              |
| 22    | merge_deadlock_fwd          | Deadlock-Flag nach Schritt vorwärts (Merge/Crossing)                         | Merge/Crossing (forward)            |
| 23    | merge_switches_fwd          | Anzahl Weichen nach Schritt vorwärts (Merge/Crossing)                        | Merge/Crossing (forward)            |
| 24    | merge_target_found_fwd      | Ziel erreicht nach Schritt vorwärts (Merge/Crossing)                         | Merge/Crossing (forward)            |
| 25    | merge_abort_fwd             | Abbruch-Flag nach Schritt vorwärts (Merge/Crossing)                          | Merge/Crossing (forward)            |
| 26    | merge_deadlock_bwd          | Deadlock-Flag nach Schritt rückwärts (Merge/Crossing)                        | Merge/Crossing (backward)           |
| 27    | merge_switches_bwd          | Anzahl Weichen nach Schritt rückwärts (Merge/Crossing)                       | Merge/Crossing (backward)           |
| 28    | merge_target_found_bwd      | Ziel erreicht nach Schritt rückwärts (Merge/Crossing)                        | Merge/Crossing (backward)           |
| 29    | merge_abort_bwd             | Abbruch-Flag nach Schritt rückwärts (Merge/Crossing)                         | Merge/Crossing (backward)           |
| 30    | agent_state_ready_to_depart | One-hot: Agent ist READY_TO_DEPART                                           | alle                                |
| 31    | agent_state_malfunction_off_map | One-hot: Agent ist MALFUNCTION_OFF_MAP                                  | alle                                |
| 32    | agent_state_moving          | One-hot: Agent ist MOVING                                                    | alle                                |
| 33    | agent_state_stopped         | One-hot: Agent ist STOPPED                                                   | alle                                |
| 34    | agent_state_malfunction     | One-hot: Agent ist in MALFUNCTION                                            | alle                                |
| 35    | agent_state_done            | One-hot: Agent ist DONE                                                      | alle                                |
| 36    | agent_state_other           | One-hot: Sonstiger Status                                                    | alle                                |
| 37    | last_action_left            | One-hot: Letzte Aktion war links                                              | alle                                |
| 38    | last_action_forward         | One-hot: Letzte Aktion war geradeaus                                         | alle                                |
| 39    | last_action_right           | One-hot: Letzte Aktion war rechts                                            | alle                                |
| 40    | last_action_stop            | One-hot: Letzte Aktion war stop                                              | alle                                |
| 41    | last_action_other           | One-hot: Letzte Aktion war sonstiges                                         | alle                                |

### Feature-Befüllung & Speziallogik

- **decision_type** wird immer gesetzt und kodiert die aktuelle Entscheidungssituation.
- Die Blöcke für Switch und Merge/Crossing werden nur bei passender Situation befüllt, alle anderen Felder bleiben 0.
- Die Features für jede Richtung werden per rekursiver DFS mit Backtracking berechnet (`_navigate_direction`), inkl. Deadlock- und Cycle-Erkennung.
- Unerreichbare Zellen (`np.inf`) werden durch $2 \times$ aktuelle Distanz ersetzt, NaN/Inf werden durch -1 ersetzt.
- Deadlocks werden erkannt, wenn ein entgegenkommender Agent auf dem Pfad ist.
- **abort**-Flags werden gesetzt, wenn die maximale Schrittzahl (max_steps) überschritten wird.
- **target_found** wird gesetzt, wenn das Ziel auf dem Pfad erreicht wird.


### Entscheidungsfindung & DFS-Logik (Algorithmus-Details)

Die Methode `_navigate_direction` ist das Herzstück der Entscheidungslogik. Sie implementiert eine rekursive Tiefensuche (Depth-First Search, DFS), um für jede relevante Richtung (links, geradeaus, rechts, rückwärts) den Pfad zum Ziel zu analysieren und dabei Deadlocks, Weichen, Zielerreichung und Abbruchbedingungen zu erkennen.
### MERGING Decision-Point Logic (Mai 2026 Architektur)

**Szenario: Agent vor Merge-Punkt**

Ein Agent ist eine Zelle vor einer Weiche, in die ein anderer Agent einfädelt. Das ist `decision_type & 4 = MERGING`.

Die Observation liefert:
- **Forward Evaluate:** Wenn **ich** geradeaus fahre, what happens?
  - [31-35]: Deadlock Risk, Switches, Target, Abort, Deadlock_Ahead
- **Backward Evaluate:** Wenn der **andere Agent** einfädelt, what is sein Risk?
  - [36-40]: Sein Deadlock Risk, Switches, Target, Abort, Deadlock_Ahead

**Mai 2026 Improvement:** Features [36] `mgB_dl_sig` nutzt jetzt **Inverse-Decay**, nicht Binary:
- Wenn gegner Agent 1 Schritt bis Deadlock: Warnung = 1.0 (sehr hoch!)
- Wenn gegner Agent 8 Schritte bis Deadlock: Warnung = 0.33 (weniger kritisch)
- Wenn gegner Agent safe: Warnung = 0.0

**Kooperations-Potential:**
- Agent mit niedriger Forward-Risk (low [31]) kann **DO_NOTHING** machen um anderer Agent durchzulassen
- Agent mit niedriger Backward-Risk (low [36]) kann aggressiv fahren
- **Policy lernt:** Wenn `mgB_dl_sig[36] > 0.8` → machen Sie Platz! DO_NOTHING
- **Policy lernt:** Wenn `swF_dl_sig[15] > 0.8` → fahren Sie nicht geradeaus, Deadlock voraus!

#### Opponent-Awareness für Kooperation (Mai 2026 Fix #2)

Die Returned `opp_agent_handles` Liste wird jetzt **vollständig gefüllt:**
1. Agenten auf allen 3 Branches (L/F/R) — von DFS besucht
2. **NEU:** Agenten am **aktuellen Knoten** — können sofort kollidieren
3. **NEU:** Agenten am **nächsten Knoten im Merge** — können blockieren

Die `TemporalMultiAgentObservation` wrapper nutzt diese Liste um die 66D Observations der K=4 wichtigsten Gegner zu sammeln.

**Result:** Attention Transformer sieht nicht nur Pfad-Informationen, sondern die **echten Gegner-Agenten** die Konflikte verursachen.

---


#### Algorithmus in verständlicher Prosa

Die rekursive Tiefensuche (DFS) zur Entscheidungsfindung funktioniert wie folgt:

1. **Start und Zielprüfung:** Die Suche beginnt an der aktuellen Position und prüft zunächst, ob das Ziel bereits erreicht ist. Ist dies der Fall, wird dies sofort als Erfolg gemeldet.
2. **Abbruchbedingungen:** Die Suche wird abgebrochen, wenn eine maximale Suchtiefe überschritten wird (um endlose Rekursionen zu verhindern) oder wenn die aktuelle Position und Richtung bereits besucht wurden (Zyklenerkennung).
3. **Deadlock-Erkennung:** Trifft die Suche auf einen entgegenkommenden oder blockierenden Agenten, wird dies als Deadlock erkannt und entsprechend im Ergebnis markiert.
4. **Weichen und Alternativen:** An jeder Weiche (Switch) prüft die Suche alle möglichen Weiterführungen (z.B. links, geradeaus, rechts, rückwärts). Für jede Alternative wird die Suche rekursiv fortgesetzt. So werden alle potenziellen Pfade analysiert.
5. **Backtracking und Auswahl:** Nach der Analyse aller Alternativen wählt die Suche das beste Ergebnis aus (z.B. den Pfad mit minimalem Deadlock-Risiko oder kürzester Distanz).
6. **Feature-Befüllung:** Die Ergebnisse der Suche – Distanz zum Ziel, Deadlock-Status, Anzahl durchlaufener Weichen, Zielerreichung und Abbruch-Flag – werden für jede Richtung in die entsprechenden Features des Beobachtungsvektors eingetragen.
7. **Cycle Prevention:** Um Endlosschleifen zu vermeiden, merkt sich die Suche alle bereits besuchten (Position, Richtung)-Paare und prüft vor jedem Schritt, ob sie erneut betreten werden.

Diese Vorgehensweise stellt sicher, dass alle relevanten Entscheidungsalternativen im Schienennetz berücksichtigt werden, Deadlocks und Konflikte realistisch erkannt werden und die resultierenden Features robust und RL-tauglich sind.

#### Warum ist diese DFS-Logik ideal für Flatland-RL?

- **Realistische Konflikterkennung:** Die DFS erkennt echte Deadlocks und Konflikte mit anderen Agenten, was für Multi-Agenten-Planung und RL-Policies essenziell ist.
- **Flexible Pfadanalyse:** Durch Backtracking an Weichen werden alle Alternativen geprüft, sodass Policies nicht in lokale Minima laufen und komplexe Entscheidungssituationen abgebildet werden.
- **Effizient und sicher:** Die maximale Suchtiefe verhindert endlose Rekursionen und hält die Berechnung effizient und RL-tauglich.
- **Zyklenerkennung:** Cycle Prevention ist wichtig, da das Schienennetz Zyklen enthalten kann – so werden Endlosschleifen vermieden.
- **Konfliktlösung:** Die DFS speichert alle gesehene Agenten auf dem Pfad, was für Prioritätsentscheidungen und Konfliktlösung genutzt werden kann.
- **RL-taugliche Features:** Die so gewonnenen Features (Distanz, Deadlock, Ziel, Weichen, Abbruch) sind direkt für RL nutzbar und ermöglichen Policies, auf komplexe Situationen zu reagieren.
- **Disjunkte Feature-Blöcke:** Nur die für die aktuelle Entscheidungssituation relevanten Features werden befüllt, was die Policy-Entwicklung vereinfacht und Overfitting reduziert.

**Fazit:**
Die rekursive DFS-Logik in `_navigate_direction` bildet die reale Entscheidungsstruktur im Flatland-Setting ab, erkennt Deadlocks und Konflikte, prüft alle Alternativen und liefert robuste, RL-taugliche Features für jede relevante Richtung. Damit ist sie optimal geeignet, um Policies für komplexe Multi-Agenten-Szenarien im Bahnnetz zu trainieren.

### Flatland-Kontext & RL-Tauglichkeit

- Die DecisionPointObservation abstrahiert die komplexen Entscheidungssituationen im Flatland-Setting in einen RL-tauglichen, disjunkten Feature-Vektor.
- Sie erkennt und kodiert alle relevanten Entscheidungs- und Konfliktpunkte, sodass Policies gezielt auf diese Situationen reagieren können.
- Die Beobachtung ist effizient, flexibel und für Einzel- wie Multi-Agenten-Szenarien geeignet.

### Methodenüberblick

- `get(handle)`: Erzeugt den Feature-Vektor für einen einzelnen Agenten.
- `get_many(handles)`: Erzeugt die Beobachtungen für mehrere Agenten gleichzeitig.
- `_shortest_path_action_hint(...)`: Berechnet, welche Richtung (links, geradeaus, rechts) entlang des kürzesten Pfads zum Ziel optimal ist.
- `_navigate_direction(...)`: Führt die rekursive Tiefensuche durch, um Pfadmetriken, Deadlocks und Zielerreichung zu bestimmen.

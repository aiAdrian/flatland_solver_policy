# Flatland Multi-Agent Observations – Übersicht & Feature-Design

## Update April 2026 (aktueller Stand)

Dieses Dokument wurde auf den aktuellen Trainingsstand erweitert.

### Was wurde neu eingebaut?

1. **Optionaler LSTM-Encoder im MAPPO-Stack (Python-only)**
     - In `marl_attention_temporal_mappo.py` gibt es jetzt zwei Encoder-Varianten mit gleicher Schnittstelle:
         - `TemporalTransformerEncoder`
         - `TemporalLSTMEncoder` (neu)
     - Beide unterstützen:
         - `forward_agent(temporal_seq, handle)`
         - `forward_batch(temporal_sequences)`

2. **Encoder-Auswahl über Parameter**
     - Der Parameter-Tuple `MARL_ATTENTION_TEMPORAL_MAPPO_Param` enthält jetzt zusätzlich:
         - `encoder_type`
     - Gültige Werte:
         - `'transformer'`
         - `'lstm'`

3. **Aktive Konfiguration im Experiment**
     - In `marl_attention_temporal.py` ist aktuell gesetzt:
         - `encoder_type='lstm'`
     - Damit läuft das Training derzeit **aktiv mit LSTM**.

### Warum ist das relevant für die Observation?

- `TemporalMultiAgentObservation` liefert eine zeitliche Sequenz pro Agent:
    - `[(obs_t-2, opp_t-2), (obs_t-1, opp_t-1), (obs_t, opp_t)]`
- Der neue LSTM-Encoder verarbeitet genau diese Sequenz und bildet daraus einen robusteren Zeitkontext.
- Dadurch werden aufeinanderfolgende Situationen (Annähern, Warten, Konfliktaufbau) besser nutzbar als bei rein statischer Einzelbeobachtung.

### Technische Integration (Kurz)

- **Keine C++-Abhängigkeit**: vollständig Python/PyTorch-basiert.
- **Kein API-Bruch**:
    - Actor/Critic-Trainingspfad bleibt gleich.
    - Nur die Encoder-Instanz wird je nach `encoder_type` gewählt.
- **Fallback-Verhalten**:
    - Wenn `encoder_type` nicht gesetzt ist, bleibt Standard auf `'transformer'`.

### Konfigurationsbeispiel

```python
ppo_param = MARL_ATTENTION_TEMPORAL_MAPPO_Param(
        hidden_size=128,
        batch_size=512,
        learning_rate=3e-4,
        discount=0.99,
        gae_lambda=0.97,
        use_gpu=True,
        max_episodes_in_training_memory=50,
        k_epochs=3,
        batch_fraction=0.4,
        max_batches_per_training=12,
        temporal_window=3,
        encoder_type='lstm',
)
```

### Erwartete Wirkung im Training

- Besseres Ausnutzen zeitlicher Muster in Entscheidungspunkten.
- Stabilere lokale Entscheidungen bei 4-5 Agenten (weniger chaotische Umschaltungen).
- Solider Kompromiss aus Einfachheit und Effektivität ohne zusätzliche Over-Engineering-Schichten.

> **Klassen im Überblick**
> | Klasse | Größe | Zweck |
> |---|---|---|
> | `ExperimentalObservation` | 30D | Basisbeobachtung für jeden Agenten |
> | `SimplifiedPathThreeTierObservation` | 57D | Drei Pfad-Tiers (links/geradeaus/rechts) |
> | `DecisionPointObservation` | 42D | Entscheidungsbasierte Beobachtung an Weichen/Merges |
> | `TemporalMultiAgentObservation` | T × 42D | Zeitfenster über `DecisionPointObservation` |

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
**Feature-Größe:** **42D**  
**Rückgabe von `get(handle)`:** `(features: np.array[42], opp_agent_handles: list[int])`

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

### Feature-Tabelle (42 Features)

#### Block A – Allgemein (immer befüllt)

| Index | Name | Beschreibung |
|---|---|---|
| 0 | `decision_type` | Entscheidungstyp (siehe Tabelle oben) |
| 1 | `hint_left` | One-hot: linke Richtung ist optimal (kürzester Pfad via `distance_map`) |
| 2 | `hint_forward` | One-hot: Geradeaus ist optimal |
| 3 | `hint_right` | One-hot: rechte Richtung ist optimal |

#### Block B – Switch-Analyse (nur wenn `decision_type & 2`)

Für jede der drei Richtungen (links/geradeaus/rechts) ein 6er-Block. Nicht erreichbare Richtung → alle Werte `-1`.

| Index | rel_dir | Name | Beschreibung |
|---|---|---|---|
| 4 | links | `left_curr_dist` | Aktuelle Distanz zum Ziel **vor** dem Schritt |
| 5 | links | `left_deadlock` | Deadlock-Flag aus DFS (0=frei, 1=Gegenverkehr, 2=Selbst-Block) |
| 6 | links | `left_switches` | Anzahl Weichen auf dem DFS-Pfad |
| 7 | links | `left_dist` | Maximale DFS-Distanz entlang des Pfades |
| 8 | links | `left_target_found` | 1 wenn Ziel auf diesem Pfad erreicht |
| 9 | links | `left_abort` | 1 wenn DFS wegen `max_steps` abgebrochen |
| 10 | gerade | `fwd_curr_dist` | Aktuelle Distanz zum Ziel |
| 11 | gerade | `fwd_deadlock` | Deadlock-Flag |
| 12 | gerade | `fwd_switches` | Anzahl Weichen |
| 13 | gerade | `fwd_dist` | Maximale DFS-Distanz |
| 14 | gerade | `fwd_target_found` | Ziel gefunden |
| 15 | gerade | `fwd_abort` | DFS abgebrochen |
| 16 | rechts | `right_curr_dist` | Aktuelle Distanz zum Ziel |
| 17 | rechts | `right_deadlock` | Deadlock-Flag |
| 18 | rechts | `right_switches` | Anzahl Weichen |
| 19 | rechts | `right_dist` | Maximale DFS-Distanz |
| 20 | rechts | `right_target_found` | Ziel gefunden |
| 21 | rechts | `right_abort` | DFS abgebrochen |

#### Block C – Merge/Crossing-Analyse (nur wenn `decision_type & 4`)

| Index | Name | Beschreibung |
|---|---|---|
| 22 | `merge_deadlock_fwd` | Deadlock vorwärts (nächste Zelle Richtung Weiche) |
| 23 | `merge_switches_fwd` | Anzahl Weichen vorwärts |
| 24 | `merge_target_fwd` | Ziel auf Vorwärtspfad erreicht |
| 25 | `merge_abort_fwd` | DFS abgebrochen (vorwärts) |
| 26 | `merge_deadlock_bwd` | Deadlock rückwärts (Pfad des einmündenden Agenten) |
| 27 | `merge_switches_bwd` | Anzahl Weichen rückwärts |
| 28 | `merge_target_bwd` | Ziel auf Rückwärtspfad (immer 0 bei backward_trace) |
| 29 | `merge_abort_bwd` | DFS abgebrochen (rückwärts) |

> **Hinweis Merge-Rückwärts:** Die rückwärtige DFS (`backward_trace=True`) mittelt die Ergebnisse **aller Alternativen** an Weichen (statt die beste zu nehmen). Dies modelliert die Unsicherheit, welchen Weg ein anderer Agent nehmen wird.

#### Block D – Agentenstatus (immer befüllt, One-hot via `agent.state.value`)

| Index | State-Wert | Name |
|---|---|---|
| 30 | 0 | `state_WAITING` |
| 31 | 1 | `state_READY_TO_DEPART` |
| 32 | 2 | `state_MALFUNCTION_OFF_MAP` |
| 33 | 3 | `state_MOVING` |
| 34 | 4 | `state_STOPPED` |
| 35 | 5 | `state_MALFUNCTION` |
| 36 | 6 | `state_DONE` |

#### Block E – Letzte Aktion (One-hot via `agent.action_saver.saved_action`)

| Index | Action-Wert | Name |
|---|---|---|
| 37 | 0 | `action_DO_NOTHING` |
| 38 | 1 | `action_MOVE_LEFT` |
| 39 | 2 | `action_MOVE_FORWARD` |
| 40 | 3 | `action_MOVE_RIGHT` |
| 41 | 4 | `action_STOP_MOVING` ⚠️ |

> ⚠️ **Index 41 Konflikt:** Feature [41] wird zuerst mit dem lokalen Deadlock-Flag (`_detect_deadlock`) beschrieben und danach ggf. durch `STOP_MOVING` (action=4) überschrieben. Effektiv enthält [41] entweder `1.0` (Aktion war STOP) oder den Deadlock-Wert (wenn Aktion nicht gespeichert oder nicht STOP).

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
| `seen_agents` | Alle auf dem Pfad gesehenen Agenten-Handles |

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
| Größe | 30D | 42D | T × 42D |
| Entscheidungslogik | Keine | DFS an Weichen/Merges | via Basis-Obs |
| Gegner-Info | Nein | Handles in Rückgabe | Obs-Vektoren aller Gegner |
| Zeitliche Tiefe | Nein | Nein | Ja (T Schritte) |
| Velocity-Features | Nein | Nein | Nein |


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
